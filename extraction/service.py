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
from extraction.prompts import EXTRACTION_PROMPT

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
    Calls the extraction model, returns validated items with character spans into chunk.content.
    Retries on JSON parse failure only.
    """
    prompt = EXTRACTION_PROMPT.replace("{chunk_text}", chunk.content)
    extraction_model = settings.EXTRACTION_MODEL
    client = _openai_client()

    for attempt in range(MAX_RETRIES + 1):
        response = client.chat.completions.create(
            model=extraction_model,
            messages=[{"role": "user", "content": prompt}],
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


def _resolve_source_spans(parsed: dict, chunk_text: str) -> tuple[dict, int]:
    """
    Resolve each item's source_text to start_char/end_char in chunk_text.
    source_text is overwritten with the exact chunk substring at that span.
  """
    dropped = 0
    cleaned = {}

    for layer in ("propositions", "claims", "phrases"):
        items = parsed.get(layer, [])
        kept = []
        for item in items:
            content = (item.get("content") or "").strip()
            source_text = (item.get("source_text") or "").strip()

            if not content or not source_text:
                dropped += 1
                continue

            span = _find_source_span(chunk_text, source_text)
            if span is None:
                dropped += 1
                continue

            start_char, end_char = span
            kept.append({
                "content": content,
                "source_text": chunk_text[start_char:end_char],
                "start_char": start_char,
                "end_char": end_char,
            })

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
