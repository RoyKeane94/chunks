import logging
import os

from django.conf import settings
from django.db import connection
from pgvector.django import CosineDistance

from apps.transcripts.models import AtomicPhrase, Chunk, Claim, Proposition

logger = logging.getLogger(__name__)


def _similarity_to_max_distance(similarity_threshold: float) -> float:
    return 1 - similarity_threshold


def get_retrieval_config():
    threshold = float(
        os.environ.get("RETRIEVAL_SIMILARITY_THRESHOLD")
        or settings.RETRIEVAL_SIMILARITY_THRESHOLD
    )
    phrase_threshold = float(
        os.environ.get("RETRIEVAL_PHRASE_SIMILARITY_THRESHOLD")
        or settings.RETRIEVAL_PHRASE_SIMILARITY_THRESHOLD
    )
    top_k = int(
        os.environ.get("RETRIEVAL_TOP_K")
        or settings.RETRIEVAL_TOP_K
    )
    return threshold, phrase_threshold, top_k


def _layer_hits(model, query_embedding, max_distance, layer_name, extra_fields=()):
    rows = (
        model.objects.annotate(distance=CosineDistance("embedding", query_embedding))
        .filter(distance__lte=max_distance)
        .order_by("distance")
        .values_list("id", "chunk_id", "distance", *extra_fields)
    )
    hits = []
    for row in rows:
        item_id, chunk_id, distance = row[:3]
        extra = row[3:]
        hit = {
            "layer": layer_name,
            "distance": distance,
            "chunk_id": chunk_id,
            "item_id": item_id,
        }
        if extra_fields:
            for field_name, value in zip(extra_fields, extra):
                hit[field_name] = value
        hits.append(hit)
    return hits


def search_layers(query_embedding, threshold, phrase_threshold, top_k):
    """
    Search chunk, proposition, claim, and phrase embeddings in parallel.
    Phrases use phrase_threshold; other layers use threshold.
    Returns up to top_k chunks (best hit per chunk, ordered by similarity).
    """
    phrase_max_distance = _similarity_to_max_distance(phrase_threshold)
    layer_max_distance = _similarity_to_max_distance(threshold)

    candidates = []

    for row in (
        Chunk.objects.annotate(distance=CosineDistance("embedding", query_embedding))
        .filter(distance__lte=layer_max_distance)
        .order_by("distance")
        .values_list("id", "distance")
    ):
        chunk_id, distance = row
        candidates.append({
            "layer": "chunk",
            "distance": distance,
            "chunk_id": chunk_id,
            "item_id": chunk_id,
            "highlight_start": None,
            "highlight_end": None,
            "matched_content": None,
        })

    proposition_hits = _layer_hits(
        Proposition,
        query_embedding,
        layer_max_distance,
        "proposition",
        ("start_char", "end_char", "content"),
    )
    for hit in proposition_hits:
        hit["highlight_start"] = hit.pop("start_char")
        hit["highlight_end"] = hit.pop("end_char")
        hit["matched_content"] = hit.pop("content")
    candidates.extend(proposition_hits)

    claim_hits = _layer_hits(
        Claim,
        query_embedding,
        layer_max_distance,
        "claim",
        ("start_char", "end_char", "content"),
    )
    for hit in claim_hits:
        hit["highlight_start"] = hit.pop("start_char")
        hit["highlight_end"] = hit.pop("end_char")
        hit["matched_content"] = hit.pop("content")
    candidates.extend(claim_hits)

    phrase_hits = _layer_hits(
        AtomicPhrase,
        query_embedding,
        phrase_max_distance,
        "phrase",
        ("start_char", "end_char", "content"),
    )
    for hit in phrase_hits:
        hit["highlight_start"] = hit.pop("start_char")
        hit["highlight_end"] = hit.pop("end_char")
        hit["matched_content"] = hit.pop("content")
    candidates.extend(phrase_hits)

    candidates.sort(key=lambda c: c["distance"])

    best_per_chunk = {}
    for candidate in candidates:
        chunk_id = candidate["chunk_id"]
        if chunk_id not in best_per_chunk:
            best_per_chunk[chunk_id] = candidate

    merged = sorted(best_per_chunk.values(), key=lambda c: c["distance"])[:top_k]
    return merged


def _valid_highlight(start, end, content_length):
    if start is None or end is None:
        return False
    if end <= start:
        return False
    if start == 0 and end == 0:
        return False
    if start < 0 or end > content_length:
        return False
    return True


def retrieve_similar_chunks(query_text, embed_fn):
    query_text = query_text.strip()
    if not query_text:
        return []

    if connection.vendor != "postgresql":
        raise ValueError("Retrieval requires PostgreSQL with pgvector.")

    threshold, phrase_threshold, top_k = get_retrieval_config()
    query_embedding = embed_fn([query_text])[0].tolist()

    logger.info(
        "retrieval search — threshold %.2f, phrase threshold %.2f, top_k %s, query length %s",
        threshold,
        phrase_threshold,
        top_k,
        len(query_text),
    )

    merged = search_layers(query_embedding, threshold, phrase_threshold, top_k)
    if not merged:
        logger.info("retrieval complete — 0/%s chunks above threshold", top_k)
        return []

    chunk_ids = [hit["chunk_id"] for hit in merged]
    chunks = (
        Chunk.objects.select_related("episode")
        .filter(id__in=chunk_ids)
        .defer("embedding")
        .only(
            "id",
            "chunk_index",
            "content",
            "episode_id",
            "episode__id",
            "episode__title",
            "episode__guest",
            "episode__date",
        )
    )
    chunks_by_id = {chunk.id: chunk for chunk in chunks}

    results = []
    for hit in merged:
        chunk = chunks_by_id.get(hit["chunk_id"])
        if chunk is None:
            continue

        similarity = round(1 - hit["distance"], 4)
        start = hit.get("highlight_start")
        end = hit.get("highlight_end")
        if not _valid_highlight(start, end, len(chunk.content)):
            start, end = None, None

        results.append({
            "chunk": chunk,
            "similarity": similarity,
            "matched_layer": hit["layer"],
            "highlight_start": start,
            "highlight_end": end,
            "matched_content": hit.get("matched_content"),
        })
        logger.info(
            "retrieval hit — score %.3f, layer %s, episode %s, chunk #%s",
            similarity,
            hit["layer"],
            chunk.episode.title,
            chunk.chunk_index,
        )

    # Same transcript uploaded twice → duplicate episodes with identical chunk text.
    # Keep the highest-scoring hit per unique chunk content.
    seen_content = set()
    deduped_results = []
    for result in results:
        content_key = result["chunk"].content
        if content_key in seen_content:
            continue
        seen_content.add(content_key)
        deduped_results.append(result)

    if len(deduped_results) < len(results):
        logger.info(
            "retrieval deduped %s duplicate chunk(s) with identical content",
            len(results) - len(deduped_results),
        )

    logger.info("retrieval complete — %s/%s chunks returned", len(deduped_results), top_k)
    return deduped_results
