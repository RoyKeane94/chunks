import logging
import os
import re
import time
from datetime import datetime

import numpy as np
import pdfplumber
from django.conf import settings
from django.db import connection
from openai import OpenAI
from pgvector.django import CosineDistance

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


def _parse_episode_date(value):
    if not value:
        return None
    for fmt in ("%d.%m.%Y", "%Y-%m-%d", "%m/%d/%Y"):
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            continue
    logger.warning("could not parse date: %s", value)
    return None


def parse_structured_transcript(text):
    result = {
        "title": "",
        "guest": "",
        "date": None,
        "body": text,
        "has_metadata": False,
    }
    lines = text.splitlines()
    if not lines or not lines[0].startswith("Title:"):
        return result

    result["has_metadata"] = True
    body_start = None

    for index, line in enumerate(lines):
        if line.startswith("Title:"):
            result["title"] = line.split(":", 1)[1].strip()
        elif line.startswith("Guest:"):
            result["guest"] = line.split(":", 1)[1].strip()
        elif line.startswith("Date:"):
            result["date"] = _parse_episode_date(line.split(":", 1)[1].strip())
        elif re.match(r"^─+$", line.strip()):
            cursor = index + 1
            while cursor < len(lines) and not lines[cursor].strip():
                cursor += 1
            body_start = cursor
            break

    if body_start is None:
        for index, line in enumerate(lines):
            if index > 0 and not line.strip() and lines[index - 1].startswith("Episode:"):
                cursor = index + 1
                while cursor < len(lines) and not lines[cursor].strip():
                    cursor += 1
                body_start = cursor
                break

    if body_start is not None:
        result["body"] = "\n".join(lines[body_start:]).strip()

    return result


def read_transcript_text(filepath):
    if filepath.lower().endswith(".txt"):
        content = parse_txt(filepath)
        parsed = parse_structured_transcript(content)
        if parsed["has_metadata"]:
            logger.info(
                "parsed transcript metadata — title: %s, guest: %s",
                parsed["title"],
                parsed["guest"],
            )
            return parsed["body"]
        return content
    return parse_pdf(filepath)


def parse_transcript(filepath):
    return read_transcript_text(filepath)


_ARTIFACT_PATTERN = re.compile(r"©|Copyright|Page \d+ of \d+|Colossus|Invest Like the Best")


def _is_artifact_line(line):
    stripped = line.strip()
    if len(stripped) < 4:
        return True
    return bool(_ARTIFACT_PATTERN.search(stripped))


_BLOCK_START_PATTERN = re.compile(
    r"^(?:#{1,6}\s+|\*\*[^*]+:\*\*|[A-Z][a-z]+:\s)"
)
_SECTION_TITLE_PATTERN = re.compile(r"^(?:[A-Z][a-z]+(?:\s+)){1,12}[A-Z][a-z]+\s*$")


def _is_block_start(line):
    stripped = line.strip()
    if _BLOCK_START_PATTERN.match(stripped):
        return True
    if _SECTION_TITLE_PATTERN.match(stripped) and not stripped.endswith((".", "!", "?", ":", ",")):
        return True
    return False


def _split_paragraphs(text):
    paragraphs = re.split(r"\n\s*\n", text.strip())
    return [paragraph.strip() for paragraph in paragraphs if paragraph.strip()]


def clean_text(text):
    cleaned_paragraphs = []
    current_lines = []
    removed = 0

    def flush():
        if current_lines:
            cleaned_paragraphs.append(" ".join(current_lines))
            current_lines.clear()

    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            flush()
            continue
        if _is_artifact_line(stripped):
            removed += 1
            continue
        if current_lines and _is_block_start(stripped):
            flush()
        current_lines.append(stripped)

    flush()

    result = "\n\n".join(cleaned_paragraphs)
    logger.info(
        "cleaned %s → %s characters (%s lines removed, %s paragraphs)",
        len(text),
        len(result),
        removed,
        len(cleaned_paragraphs),
    )
    return result


def _split_sentences(text):
    parts = re.split(r"(?<=[.!?])[ \t]+", text.strip())
    return [part.strip() for part in parts if part.strip()]


def _iter_sentences(text):
    for paragraph_index, paragraph in enumerate(_split_paragraphs(text)):
        for sentence in _split_sentences(paragraph):
            yield paragraph_index, sentence


def _join_sentences(sentence_items):
    parts = []
    previous_paragraph = None

    for paragraph_index, sentence in sentence_items:
        if previous_paragraph is not None and paragraph_index != previous_paragraph:
            parts.append("\n\n")
        elif parts:
            parts.append(" ")
        parts.append(sentence)
        previous_paragraph = paragraph_index

    return "".join(parts)


def _join_chunks(left, right):
    left = left.rstrip()
    right = right.lstrip()
    if not left:
        return right
    if not right:
        return left
    if left.endswith((".", "!", "?")) and (
        right.startswith("#")
        or right.startswith("**")
        or right[0].isupper()
    ):
        return f"{left}\n\n{right}"
    return f"{left} {right}"


def chunk_text(text):
    sentence_items = list(_iter_sentences(text))
    if not sentence_items:
        return []

    max_tokens = settings.CHUNK_MAX_TOKENS
    overlap = settings.CHUNK_OVERLAP_SENTENCES
    chunks = []
    start = 0

    while start < len(sentence_items):
        current = []
        token_count = 0
        idx = start

        while idx < len(sentence_items):
            paragraph_index, sentence = sentence_items[idx]
            sentence_tokens = estimate_tokens(sentence)
            if current and token_count + sentence_tokens > max_tokens:
                break
            current.append((paragraph_index, sentence))
            token_count += sentence_tokens
            idx += 1

        if not current:
            current.append(sentence_items[start])
            idx = start + 1

        chunks.append(_join_sentences(current))

        if idx >= len(sentence_items):
            break

        next_start = max(idx - overlap, start + 1)
        start = next_start

    logger.info(
        "chunked %s sentences → %s raw chunks (max %s tokens, overlap %s)",
        len(sentence_items),
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
            current_text = _join_chunks(current_text, chunks[i])
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


def ingest_episode(episode_id, raw_text=None):
    episode = Episode.objects.get(pk=episode_id)
    started = time.monotonic()
    logger.info("——— ingest start — episode %s: %s ———", episode.id, episode.title)

    logger.info("[1/5] reading transcript file")
    step_start = time.monotonic()
    if raw_text is None:
        raw_text = read_transcript_text(episode.pdf_file.path)
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


def get_retrieval_config():
    threshold = float(
        os.environ.get("RETRIEVAL_SIMILARITY_THRESHOLD")
        or settings.RETRIEVAL_SIMILARITY_THRESHOLD
    )
    top_k = int(
        os.environ.get("RETRIEVAL_TOP_K")
        or settings.RETRIEVAL_TOP_K
    )
    return threshold, top_k


def retrieve_similar_chunks(query_text):
    query_text = query_text.strip()
    if not query_text:
        return []

    if connection.vendor != "postgresql":
        raise ValueError("Retrieval requires PostgreSQL with pgvector.")

    threshold, top_k = get_retrieval_config()
    max_distance = 1 - threshold

    query_embedding = _embed_texts([query_text])[0].tolist()
    logger.info(
        "retrieval search — threshold %.2f, top_k %s, query length %s",
        threshold,
        top_k,
        len(query_text),
    )

    chunks = (
        Chunk.objects.select_related("episode")
        .annotate(distance=CosineDistance("embedding", query_embedding))
        .filter(distance__lt=max_distance)
        .order_by("distance")[:top_k]
    )

    results = []
    for chunk in chunks:
        similarity = 1 - chunk.distance
        results.append({
            "chunk": chunk,
            "similarity": round(similarity, 4),
        })
        logger.info(
            "retrieval hit — score %.3f, episode %s, chunk #%s",
            similarity,
            chunk.episode.title,
            chunk.chunk_index,
        )

    logger.info("retrieval complete — %s/%s chunks above threshold", len(results), top_k)
    return results


def ingest_bulk_files(files):
    max_files = settings.BULK_UPLOAD_MAX_FILES
    if len(files) > max_files:
        raise ValueError(f"Bulk upload limited to {max_files} files per batch.")

    results = {"succeeded": [], "failed": [], "total": len(files)}

    for index, uploaded_file in enumerate(files, start=1):
        filename = uploaded_file.name
        logger.info("bulk upload %s/%s — %s", index, len(files), filename)
        try:
            if not filename.lower().endswith(".txt"):
                raise ValueError("Only .txt files are supported for bulk upload.")

            content = uploaded_file.read().decode("utf-8").strip()
            parsed = parse_structured_transcript(content)

            if not parsed["has_metadata"]:
                raise ValueError("Missing Title:/Guest:/Date: header block.")
            if not parsed["title"]:
                raise ValueError("Title is missing from file header.")
            if not parsed["guest"]:
                raise ValueError("Guest is missing from file header.")

            uploaded_file.seek(0)
            episode = Episode.objects.create(
                title=parsed["title"][:255],
                guest=parsed["guest"][:255],
                date=parsed["date"],
                pdf_file=uploaded_file,
            )
            report = ingest_episode(episode.id, raw_text=parsed["body"])
            results["succeeded"].append({
                "filename": filename,
                "episode_id": episode.id,
                "title": episode.title,
                "guest": episode.guest,
                "date": episode.date.isoformat() if episode.date else None,
                "final_chunk_count": report["final_chunk_count"],
            })
        except Exception as exc:
            logger.exception("bulk upload failed for %s", filename)
            results["failed"].append({
                "filename": filename,
                "error": str(exc),
            })

    return results
