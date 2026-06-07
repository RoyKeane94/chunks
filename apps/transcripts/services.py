import logging
import os
import re
import time

import numpy as np
import pdfplumber
from django.conf import settings
from openai import OpenAI

from .models import Chunk, Episode

logger = logging.getLogger(__name__)


def estimate_tokens(text):
    return max(1, len(text.split()))


def parse_pdf(filepath):
    pages = []
    with pdfplumber.open(filepath) as pdf:
        page_count = len(pdf.pages)
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
    text = "\n".join(pages).strip()
    logger.info("extracted %s characters from %s/%s pdf pages", len(text), len(pages), page_count)
    return text


def parse_txt(filepath):
    with open(filepath, encoding="utf-8") as handle:
        text = handle.read().strip()
    logger.info("read %s characters from txt file", len(text))
    return text


def parse_transcript(filepath):
    if filepath.lower().endswith(".txt"):
        return parse_txt(filepath)
    return parse_pdf(filepath)


def clean_text(text):
    lines = text.split("\n")
    cleaned = []
    removed = 0
    for line in lines:
        stripped = line.strip()
        if len(stripped) < 4:
            removed += 1
            continue
        if re.search(r"©|Copyright|Page \d+ of \d+|Colossus|Invest Like the Best", stripped):
            removed += 1
            continue
        cleaned.append(line)
    result = "\n".join(cleaned)
    logger.info(
        "cleaned %s → %s characters (%s lines removed)",
        len(text),
        len(result),
        removed,
    )
    return result


def _split_sentences(text):
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [part.strip() for part in parts if part.strip()]


def chunk_text(text):
    sentences = _split_sentences(text)
    if not sentences:
        return []

    max_tokens = settings.CHUNK_MAX_TOKENS
    overlap = settings.CHUNK_OVERLAP_SENTENCES
    chunks = []
    start = 0

    while start < len(sentences):
        current = []
        token_count = 0
        idx = start

        while idx < len(sentences):
            sentence = sentences[idx]
            sentence_tokens = estimate_tokens(sentence)
            if current and token_count + sentence_tokens > max_tokens:
                break
            current.append(sentence)
            token_count += sentence_tokens
            idx += 1

        if not current:
            current.append(sentences[start])
            idx = start + 1

        chunks.append(" ".join(current))

        if idx >= len(sentences):
            break

        next_start = max(idx - overlap, start + 1)
        start = next_start

    logger.info(
        "chunked %s sentences → %s raw chunks (max %s tokens, overlap %s)",
        len(sentences),
        len(chunks),
        max_tokens,
        overlap,
    )
    return chunks


def _openai_client():
    api_key = os.environ.get("OPENAI_API_KEY") or settings.OPENAI_API_KEY
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY is not set. Add it to your .env file and restart the server."
        )
    return OpenAI(api_key=api_key, timeout=settings.OPENAI_TIMEOUT)


def _embed_texts(texts):
    logger.info("embedding %s chunks with %s", len(texts), settings.EMBEDDING_MODEL)
    client = _openai_client()
    response = client.embeddings.create(
        model=settings.EMBEDDING_MODEL,
        input=texts,
    )
    embeddings = [np.array(item.embedding, dtype=np.float32) for item in response.data]
    logger.info("received %s embeddings", len(embeddings))
    return embeddings


def _normalize(vector):
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


def semantic_merge(chunks):
    if not chunks:
        return [], {"merges": [], "groups": []}

    embeddings = _embed_texts(chunks)
    threshold = settings.SEMANTIC_CHUNK_THRESHOLD
    max_merged_tokens = settings.SEMANTIC_MAX_MERGED_TOKENS

    merged_chunks = []
    merged_embeddings = []
    merges = []
    groups = []

    current_text = chunks[0]
    current_embedding = embeddings[0]
    current_tokens = estimate_tokens(current_text)
    current_indices = [0]

    for i in range(1, len(chunks)):
        similarity = float(np.dot(current_embedding, embeddings[i]))
        next_tokens = estimate_tokens(chunks[i])
        combined_tokens = current_tokens + next_tokens

        if similarity > threshold and combined_tokens <= max_merged_tokens:
            merges.append({
                "raw_index_a": current_indices[-1],
                "raw_index_b": i,
                "similarity": round(similarity, 3),
                "combined_tokens": combined_tokens,
            })
            current_text = f"{current_text} {chunks[i]}"
            current_embedding = _normalize((current_embedding + embeddings[i]) / 2)
            current_tokens = combined_tokens
            current_indices.append(i)
        else:
            groups.append({
                "raw_indices": current_indices,
                "token_estimate": current_tokens,
                "merged": len(current_indices) > 1,
            })
            merged_chunks.append(current_text)
            merged_embeddings.append(current_embedding)
            current_text = chunks[i]
            current_embedding = embeddings[i]
            current_tokens = next_tokens
            current_indices = [i]

    groups.append({
        "raw_indices": current_indices,
        "token_estimate": current_tokens,
        "merged": len(current_indices) > 1,
    })
    merged_chunks.append(current_text)
    merged_embeddings.append(current_embedding)

    logger.info(
        "merged %s raw chunks → %s final chunks (%s pairwise merges)",
        len(chunks),
        len(merged_chunks),
        len(merges),
    )
    return list(zip(merged_chunks, merged_embeddings)), {"merges": merges, "groups": groups}


def ingest_episode(episode_id):
    episode = Episode.objects.get(pk=episode_id)
    started = time.monotonic()
    logger.info("——— ingest start — episode %s: %s ———", episode.id, episode.title)

    logger.info("[1/5] reading transcript file")
    step_start = time.monotonic()
    raw_text = parse_transcript(episode.pdf_file.path)
    logger.info("[1/5] done (%.1fs)", time.monotonic() - step_start)

    logger.info("[2/5] cleaning transcript")
    step_start = time.monotonic()
    text = clean_text(raw_text)
    logger.info("[2/5] done (%.1fs)", time.monotonic() - step_start)

    logger.info("[3/5] chunking text")
    step_start = time.monotonic()
    raw_chunks = chunk_text(text)
    logger.info("[3/5] done (%.1fs)", time.monotonic() - step_start)

    logger.info("[4/5] embedding and merging")
    step_start = time.monotonic()
    merged, merge_log = semantic_merge(raw_chunks)
    logger.info("[4/5] done (%.1fs)", time.monotonic() - step_start)

    logger.info("[5/5] saving %s chunks to database", len(merged))
    step_start = time.monotonic()
    Chunk.objects.filter(episode=episode).delete()

    for index, (content, embedding) in enumerate(merged):
        Chunk.objects.create(
            episode=episode,
            content=content,
            chunk_index=index,
            token_estimate=estimate_tokens(content),
            embedding=embedding.tolist(),
            embedding_model=settings.EMBEDDING_MODEL,
        )
        if (index + 1) % 25 == 0 or index + 1 == len(merged):
            logger.info("[5/5] saved %s/%s chunks", index + 1, len(merged))

    logger.info("[5/5] done (%.1fs)", time.monotonic() - step_start)
    logger.info(
        "——— ingest complete — episode %s: %s final chunks (%.1fs total) ———",
        episode.id,
        len(merged),
        time.monotonic() - started,
    )

    return {
        "raw_character_count": len(raw_text),
        "character_count": len(text),
        "raw_chunk_count": len(raw_chunks),
        "final_chunk_count": len(merged),
        "settings": {
            "chunk_max_tokens": settings.CHUNK_MAX_TOKENS,
            "chunk_overlap_sentences": settings.CHUNK_OVERLAP_SENTENCES,
            "semantic_threshold": settings.SEMANTIC_CHUNK_THRESHOLD,
            "semantic_max_merged_tokens": settings.SEMANTIC_MAX_MERGED_TOKENS,
            "embedding_model": settings.EMBEDDING_MODEL,
        },
        "raw_chunks": [
            {
                "index": index,
                "tokens": estimate_tokens(content),
                "content": content,
            }
            for index, content in enumerate(raw_chunks)
        ],
        "merges": merge_log["merges"],
        "final_chunks": [
            {
                "index": index,
                "tokens": group["token_estimate"],
                "content": content,
                "raw_indices": group["raw_indices"],
                "merged": group["merged"],
            }
            for index, ((content, _embedding), group) in enumerate(zip(merged, merge_log["groups"]))
        ],
    }
