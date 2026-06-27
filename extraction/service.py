import json
import logging
import re
import time
from difflib import SequenceMatcher

import numpy as np
from django.conf import settings
from django.db import connection, transaction
from django.db.utils import OperationalError
from django.utils import timezone
from openai import OpenAI

from apps.transcripts.models import AtomicPhrase, Chunk, Claim, Proposition
from extraction.lookback_pass import find_flagged_items, LookbackSummary, run_lookback_pass
from extraction.prompts import EXTRACTION_SYSTEM_PROMPT, EXTRACTION_USER_TEMPLATE

logger = logging.getLogger(__name__)

MAX_RETRIES = 2
MAX_DB_RETRIES = 5
FUZZY_MATCH_THRESHOLD = 0.82


def _openai_client():
    api_key = settings.OPENAI_API_KEY
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set.")
    return OpenAI(api_key=api_key, timeout=settings.OPENAI_TIMEOUT)


def _normalize_for_match(text: str) -> str:
    replacements = [
        ("\u2019", "'"),
        ("\u2018", "'"),
        ("\u201c", '"'),
        ("\u201d", '"'),
        ("\u2013", "-"),
        ("\u2014", "-"),
        ("…", "..."),
    ]
    for old, new in replacements:
        text = text.replace(old, new)
    return re.sub(r"\s+", " ", text).strip()


def _find_source_span(chunk_text: str, source_text: str) -> tuple[int, int] | None:
    """Locate source_text in chunk_text; tolerate quote/whitespace differences."""
    source_text = source_text.strip()
    if not source_text:
        return None

    start = chunk_text.find(source_text)
    if start >= 0:
        return start, start + len(source_text)

    stripped = source_text.strip()
    start = chunk_text.find(stripped)
    if start >= 0:
        return start, start + len(stripped)

    words = stripped.split()
    if len(words) >= 2:
        pattern = r"\s+".join(re.escape(word) for word in words)
        match = re.search(pattern, chunk_text)
        if match:
            return match.start(), match.end()

    norm_source = _normalize_for_match(stripped)
    if len(norm_source) < 8:
        return None

    target_len = len(stripped)
    best_ratio = 0.0
    best_span = None

    for start in range(len(chunk_text)):
        for length in range(max(8, target_len - 15), target_len + 25):
            end = start + length
            if end > len(chunk_text):
                break
            candidate = chunk_text[start:end]
            ratio = SequenceMatcher(
                None,
                _normalize_for_match(candidate),
                norm_source,
            ).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_span = (start, end)

    if best_span and best_ratio >= FUZZY_MATCH_THRESHOLD:
        return best_span

    return None


def extract_layers_for_chunk(chunk: Chunk) -> tuple[dict, int]:
    """
    Calls the extraction model, returns validated items with character spans.
    Retries on JSON parse failure only.
    """
    user_content = EXTRACTION_USER_TEMPLATE.format(chunk_text=chunk.content)
    extraction_model = settings.EXTRACTION_MODEL
    client = _openai_client()

    for attempt in range(MAX_RETRIES + 1):
        response = client.chat.completions.create(
            model=extraction_model,
            messages=[
                {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            temperature=0.2,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("Chunk %s: JSON parse failed on attempt %s", chunk.id, attempt)
            continue

        validated, dropped = _resolve_source_spans(parsed, chunk.content)

        if dropped:
            logger.warning(
                "Chunk %s: dropped %s item(s) with no locatable source span on attempt %s",
                chunk.id,
                dropped,
                attempt,
            )

        return validated, dropped

    raise RuntimeError(
        f"Extraction failed for chunk {chunk.id} after {MAX_RETRIES + 1} attempts"
    )


def _normalize_layer_items(raw_items) -> list[dict]:
    """Accept model output whether items are dicts or bare strings."""
    if not raw_items:
        return []
    if isinstance(raw_items, dict):
        raw_items = list(raw_items.values())
    if not isinstance(raw_items, list):
        return []

    normalized = []
    for item in raw_items:
        if isinstance(item, str):
            text = item.strip()
            if text:
                normalized.append({"content": text, "source_text": text})
        elif isinstance(item, dict):
            content = (item.get("content") or "").strip()
            source_text = (item.get("source_text") or content or "").strip()
            if content:
                normalized_item = {"content": content, "source_text": source_text}
                if item.get("needs_lookback"):
                    normalized_item["needs_lookback"] = True
                normalized.append(normalized_item)
        else:
            logger.warning("Skipping unexpected layer item type: %s", type(item).__name__)
    return normalized


def _resolve_source_spans(parsed: dict, chunk_text: str) -> tuple[dict, int]:
    """
    Resolve each item's source_text to start_char/end_char in chunk_text.
    source_text is overwritten with the exact chunk substring at that span.
  """
    dropped = 0
    cleaned = {}

    for layer in ("propositions", "claims", "phrases"):
        items = _normalize_layer_items(parsed.get(layer, []))
        kept = []
        for item in items:
            content = item["content"]
            source_text = item["source_text"]

            if not content or not source_text:
                dropped += 1
                continue

            span = _find_source_span(chunk_text, source_text)
            if span is None:
                dropped += 1
                continue

            start_char, end_char = span
            kept_item = {
                "content": content,
                "source_text": chunk_text[start_char:end_char],
                "start_char": start_char,
                "end_char": end_char,
            }
            if layer != "phrases" and item.get("needs_lookback"):
                kept_item["needs_lookback"] = True
            kept.append(kept_item)

        cleaned[layer] = kept

    return cleaned, dropped


def _cosine_similarity(vec_a, vec_b) -> float:
    a = np.asarray(vec_a, dtype=np.float32).flatten()
    b = np.asarray(vec_b, dtype=np.float32).flatten()
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def dedupe_claims_against_propositions(
    claims: list,
    propositions: list,
    claim_embeddings: list,
    proposition_embeddings: list,
    threshold: float | None = None,
) -> tuple[list, list, int]:
    """
    Drop claims whose embedding is too similar to any proposition from the same chunk.
    Returns (kept_claims, kept_claim_embeddings, dropped_count).
    """
    if not claims or not propositions:
        return claims, claim_embeddings, 0

    threshold = threshold or settings.CLAIM_PROPOSITION_DEDUP_THRESHOLD
    kept_claims = []
    kept_embeddings = []
    dropped = 0

    for claim, claim_embedding in zip(claims, claim_embeddings):
        max_sim = max(
            _cosine_similarity(claim_embedding, prop_embedding)
            for prop_embedding in proposition_embeddings
        )
        if max_sim < threshold:
            kept_claims.append(claim)
            kept_embeddings.append(claim_embedding)
        else:
            dropped += 1
            logger.info(
                "Dropped redundant claim (sim %.3f): %s",
                max_sim,
                claim["content"][:80],
            )

    return kept_claims, kept_embeddings, dropped


def embed_and_dedupe_extraction_layers(extracted: dict, embed_fn) -> tuple[dict, tuple[list, list, list], int]:
    """
    Embed all layers once, dedupe claims against propositions, return updated extracted
    dict, embeddings tuple, and redundant claim drop count.
    """
    proposition_texts = [item["content"] for item in extracted["propositions"]]
    claim_texts = [item["content"] for item in extracted["claims"]]
    phrase_texts = [item["content"] for item in extracted["phrases"]]

    prop_count = len(proposition_texts)
    claim_count = len(claim_texts)
    all_texts = proposition_texts + claim_texts + phrase_texts

    if not all_texts:
        return extracted, ([], [], []), 0

    all_embeddings = embed_fn(all_texts)
    proposition_embeddings = all_embeddings[:prop_count]
    claim_embeddings = all_embeddings[prop_count:prop_count + claim_count]
    phrase_embeddings = all_embeddings[prop_count + claim_count:]

    kept_claims, kept_claim_embeddings, redundant_dropped = dedupe_claims_against_propositions(
        extracted["claims"],
        extracted["propositions"],
        claim_embeddings,
        proposition_embeddings,
    )
    extracted = {
        **extracted,
        "claims": kept_claims,
    }

    return extracted, (
        proposition_embeddings,
        kept_claim_embeddings,
        phrase_embeddings,
    ), redundant_dropped


def embed_extraction_layers(extracted: dict, embed_fn) -> tuple[list, list, list]:
    proposition_texts = [item["content"] for item in extracted["propositions"]]
    claim_texts = [item["content"] for item in extracted["claims"]]
    phrase_texts = [item["content"] for item in extracted["phrases"]]

    prop_count = len(proposition_texts)
    claim_count = len(claim_texts)
    phrase_count = len(phrase_texts)
    all_texts = proposition_texts + claim_texts + phrase_texts

    if not all_texts:
        return [], [], []

    all_embeddings = embed_fn(all_texts)
    proposition_embeddings = all_embeddings[:prop_count]
    claim_embeddings = all_embeddings[prop_count:prop_count + claim_count]
    phrase_embeddings = all_embeddings[prop_count + claim_count:]

    return proposition_embeddings, claim_embeddings, phrase_embeddings


def persist_extraction(
    chunk: Chunk,
    extracted: dict,
    embeddings: tuple[list, list, list],
) -> None:
    """Write extraction rows to the database; retries on transient connection errors."""
    proposition_embeddings, claim_embeddings, phrase_embeddings = embeddings
    embedding_model = settings.EMBEDDING_MODEL
    extraction_model = settings.EXTRACTION_MODEL
    embedded_at = timezone.now()

    for attempt in range(MAX_DB_RETRIES):
        connection.close()
        try:
            with transaction.atomic():
                chunk = Chunk.objects.get(pk=chunk.pk)

                Proposition.objects.bulk_create([
                    Proposition(
                        chunk=chunk,
                        content=item["content"],
                        source_text=item["source_text"],
                        start_char=item["start_char"],
                        end_char=item["end_char"],
                        embedding=embedding,
                        embedding_model=embedding_model,
                        embedded_at=embedded_at,
                        extraction_model=extraction_model,
                    )
                    for item, embedding in zip(extracted["propositions"], proposition_embeddings)
                ])

                Claim.objects.bulk_create([
                    Claim(
                        chunk=chunk,
                        content=item["content"],
                        source_text=item["source_text"],
                        start_char=item["start_char"],
                        end_char=item["end_char"],
                        embedding=embedding,
                        embedding_model=embedding_model,
                        embedded_at=embedded_at,
                        extraction_model=extraction_model,
                    )
                    for item, embedding in zip(extracted["claims"], claim_embeddings)
                ])

                AtomicPhrase.objects.bulk_create([
                    AtomicPhrase(
                        chunk=chunk,
                        content=item["content"],
                        source_text=item["source_text"],
                        start_char=item["start_char"],
                        end_char=item["end_char"],
                        embedding=embedding,
                        embedding_model=embedding_model,
                        embedded_at=embedded_at,
                        extraction_model=extraction_model,
                    )
                    for item, embedding in zip(extracted["phrases"], phrase_embeddings)
                ])

                chunk.extraction_model = extraction_model
                chunk.extracted_at = timezone.now()
                chunk.save(update_fields=["extraction_model", "extracted_at"])
            return
        except OperationalError as exc:
            if attempt >= MAX_DB_RETRIES - 1:
                raise
            wait = min(2 ** attempt, 30)
            logger.warning(
                "Chunk %s: DB write failed (attempt %s/%s): %s — retrying in %ss",
                chunk.id,
                attempt + 1,
                MAX_DB_RETRIES,
                exc,
                wait,
            )
            time.sleep(wait)


def save_extraction(chunk: Chunk, extracted: dict, embed_fn) -> tuple[dict, int]:
    """
    embed_fn: callable taking a list[str] and returning list[list[float]],
    wrapping the existing text-embedding-3-small call used elsewhere in the project.
    Returns (updated extracted dict, redundant claim drop count).
    """
    extracted, embeddings, redundant_dropped = embed_and_dedupe_extraction_layers(
        extracted,
        embed_fn,
    )
    persist_extraction(chunk, extracted, embeddings)
    return extracted, redundant_dropped


def _call_lookback_llm(system_prompt: str, user_prompt: str) -> dict:
    client = _openai_client()
    response = client.chat.completions.create(
        model=settings.LOOKBACK_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
        response_format={"type": "json_object"},
    )
    raw = response.choices[0].message.content
    return json.loads(raw)


def build_prior_chunks_text(chunks_by_index: dict[int, Chunk], chunk_index: int, n: int) -> str:
    """Concatenate raw text from the n chunks immediately before chunk_index."""
    prior_indices = sorted(i for i in chunks_by_index if i < chunk_index)
    prior_indices = prior_indices[-n:]
    return "\n\n".join(chunks_by_index[i].content for i in prior_indices)


def run_episode_lookback(
    pending: list[dict],
    chunks_by_index: dict[int, Chunk],
    max_lookback_chunks: int | None = None,
    guest_name: str | None = None,
) -> LookbackSummary:
    """
    Run deferred lookback for one episode's in-memory extractions.

    pending: list of {"chunk": Chunk, "extracted": dict} entries from the main pass.
    chunks_by_index: all episode chunks keyed by chunk_index (for prior-chunk text).
    guest_name: optional episode guest name for resolving bare "you"/"your".
    Mutates extracted dicts in place.

    Returns LookbackSummary with initiated=True only when flagged items triggered
    lookback LLM calls; initiated=False when nothing needed lookback.
    """
    extracted_chunks = [
        {
            "chunk_index": entry["chunk"].chunk_index,
            "chunk_id": entry["chunk"].id,
            "propositions": entry["extracted"]["propositions"],
            "claims": entry["extracted"]["claims"],
        }
        for entry in pending
    ]

    flagged_count = len(find_flagged_items(extracted_chunks))
    if not flagged_count:
        logger.info("Lookback pass: not initiated (0 flagged items)")
        return LookbackSummary(initiated=False, flagged_count=0)

    max_chunks = max_lookback_chunks or settings.LOOKBACK_MAX_CHUNKS
    guest = (guest_name or "").strip() or None
    if guest:
        logger.info(
            "Lookback pass: initiated — resolving %s flagged item(s) (guest: %s)",
            flagged_count,
            guest,
        )
    else:
        logger.info("Lookback pass: initiated — resolving %s flagged item(s)", flagged_count)

    def get_prior_chunks_text(chunk_index, n):
        return build_prior_chunks_text(chunks_by_index, chunk_index, n)

    summary = run_lookback_pass(
        extracted_chunks,
        _call_lookback_llm,
        get_prior_chunks_text,
        guest_name=guest,
        max_lookback_chunks=max_chunks,
    )

    if summary.still_unresolved:
        logger.warning(
            "Lookback pass: %s item(s) still unresolved: %s",
            len(summary.still_unresolved),
            summary.still_unresolved,
        )
    else:
        logger.info(
            "Lookback pass: resolved %s/%s flagged item(s) in %s LLM call(s)",
            summary.resolved_count,
            summary.flagged_count,
            summary.llm_calls,
        )

    return summary


def format_lookback_summary(summary: LookbackSummary) -> str:
    """Human-readable one-line status for CLI logging."""
    if not summary.initiated:
        return "Lookback: not initiated (0 items flagged with needs_lookback)"
    parts = [
        "Lookback: initiated",
        f"{summary.flagged_count} flagged",
        f"{summary.llm_calls} LLM call(s)",
        f"{summary.resolved_count} resolved",
    ]
    if summary.still_unresolved:
        parts.append(f"{len(summary.still_unresolved)} still unresolved")
    return " — ".join(parts)
