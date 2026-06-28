"""Layer extraction progress and unresolved lookback export."""

from collections import defaultdict
from datetime import datetime

from django.conf import settings
from django.db import connection
from django.db.models import Count, F, Prefetch, Q

from apps.transcripts.models import Chunk, Claim, Episode, Proposition

_lookback_column_available: bool | None = None


def lookback_column_available() -> bool:
    """True once migration 0009 has been applied."""
    global _lookback_column_available
    if _lookback_column_available is not None:
        return _lookback_column_available
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT 1 FROM information_schema.columns
                WHERE table_name = 'transcripts_proposition'
                  AND column_name = 'needs_lookback'
                LIMIT 1
                """
            )
            _lookback_column_available = cursor.fetchone() is not None
    except Exception:
        _lookback_column_available = False
    return _lookback_column_available


def _unresolved_counts_by_episode() -> dict[int, int]:
    if not lookback_column_available():
        return {}
    counts: dict[int, int] = defaultdict(int)
    for row in (
        Proposition.objects.filter(needs_lookback=True)
        .values("chunk__episode_id")
        .annotate(c=Count("id"))
    ):
        counts[row["chunk__episode_id"]] += row["c"]
    for row in (
        Claim.objects.filter(needs_lookback=True)
        .values("chunk__episode_id")
        .annotate(c=Count("id"))
    ):
        counts[row["chunk__episode_id"]] += row["c"]
    return counts


def _prior_indices(chunk_index: int, n: int, indices: list[int]) -> list[int]:
    return sorted(i for i in indices if i < chunk_index)[-n:]


def _chunk_export_block(
    chunk: Chunk,
    include_propositions: bool = True,
    include_claims: bool = False,
) -> dict:
    block = {
        "chunk_id": chunk.id,
        "chunk_index": chunk.chunk_index,
        "token_estimate": chunk.token_estimate,
        "transcript": chunk.content,
    }
    if include_propositions:
        block["propositions"] = [
            {
                "id": prop.id,
                "content": prop.content,
                "source_text": prop.source_text,
                "needs_lookback": prop.needs_lookback,
            }
            for prop in chunk.propositions.all()
        ]
    if include_claims:
        block["claims"] = [
            {
                "id": claim.id,
                "content": claim.content,
                "source_text": claim.source_text,
                "needs_lookback": claim.needs_lookback,
            }
            for claim in chunk.claims.all()
        ]
    return block


def resolve_unresolved_item(layer: str, item_id: int, content: str) -> tuple[bool, str]:
    """Update item text and clear needs_lookback (manual resolution)."""
    content = content.strip()
    if not content:
        return False, "Content cannot be empty."
    if not lookback_column_available():
        return False, "Unresolved tracking is not available until migrations are applied."

    if layer == "propositions":
        model = Proposition
    elif layer == "claims":
        model = Claim
    else:
        return False, "Invalid layer."

    try:
        row = model.objects.get(pk=item_id, needs_lookback=True)
    except model.DoesNotExist:
        return False, "Item not found or already resolved."

    row.content = content
    row.needs_lookback = False
    row.save(update_fields=["content", "needs_lookback"])
    return True, ""


def delete_unresolved_item(layer: str, item_id: int) -> tuple[bool, str]:
    """Remove an unresolved proposition or claim."""
    if not lookback_column_available():
        return False, "Unresolved tracking is not available until migrations are applied."

    if layer == "propositions":
        model = Proposition
    elif layer == "claims":
        model = Claim
    else:
        return False, "Invalid layer."

    try:
        row = model.objects.get(pk=item_id, needs_lookback=True)
    except model.DoesNotExist:
        return False, "Item not found or already resolved."

    row.delete()
    return True, ""


def get_extraction_summary() -> dict:
    episodes = Episode.objects.annotate(
        chunk_count=Count("chunks", distinct=True),
        layered_chunk_count=Count(
            "chunks",
            filter=Q(chunks__extracted_at__isnull=False),
            distinct=True,
        ),
    )
    with_chunks = episodes.filter(chunk_count__gt=0)
    fully_layered = with_chunks.filter(chunk_count=F("layered_chunk_count")).count()
    partially_layered = (
        with_chunks.filter(layered_chunk_count__gt=0)
        .exclude(chunk_count=F("layered_chunk_count"))
        .count()
    )
    not_started = with_chunks.filter(layered_chunk_count=0).count()
    empty_episodes = episodes.filter(chunk_count=0).count()

    total_chunks = Chunk.objects.count()
    layered_chunks = Chunk.objects.filter(extracted_at__isnull=False).count()

    if lookback_column_available():
        unresolved_propositions = Proposition.objects.filter(needs_lookback=True).count()
        unresolved_claims = Claim.objects.filter(needs_lookback=True).count()
        episodes_with_unresolved = (
            Episode.objects.filter(
                Q(chunks__propositions__needs_lookback=True)
                | Q(chunks__claims__needs_lookback=True)
            )
            .distinct()
            .count()
        )
    else:
        unresolved_propositions = 0
        unresolved_claims = 0
        episodes_with_unresolved = 0

    return {
        "episodes_total": episodes.count(),
        "episodes_with_chunks": with_chunks.count(),
        "episodes_fully_layered": fully_layered,
        "episodes_partially_layered": partially_layered,
        "episodes_not_started": not_started,
        "episodes_empty": empty_episodes,
        "chunks_total": total_chunks,
        "chunks_layered": layered_chunks,
        "chunks_pending": total_chunks - layered_chunks,
        "unresolved_propositions": unresolved_propositions,
        "unresolved_claims": unresolved_claims,
        "unresolved_total": unresolved_propositions + unresolved_claims,
        "episodes_with_unresolved": episodes_with_unresolved,
        "lookback_column_available": lookback_column_available(),
    }


def list_episode_layer_status(limit: int | None = None) -> list[dict]:
    unresolved_counts = _unresolved_counts_by_episode()
    episodes = (
        Episode.objects.annotate(
            chunk_count=Count("chunks", distinct=True),
            layered_chunk_count=Count(
                "chunks",
                filter=Q(chunks__extracted_at__isnull=False),
                distinct=True,
            ),
        )
        .filter(chunk_count__gt=0)
        .order_by("id")
    )
    if limit:
        episodes = episodes[:limit]

    rows = []
    for ep in episodes:
        if ep.chunk_count == ep.layered_chunk_count:
            status = "fully_layered"
        elif ep.layered_chunk_count > 0:
            status = "partial"
        else:
            status = "not_started"
        rows.append({
            "episode_id": ep.id,
            "title": ep.title,
            "guest": ep.guest,
            "chunk_count": ep.chunk_count,
            "layered_chunk_count": ep.layered_chunk_count,
            "status": status,
            "unresolved_count": unresolved_counts.get(ep.id, 0),
        })
    return rows


def build_unresolved_lookback_export(episode_id: int | None = None) -> dict:
    """
    JSON export of items still flagged needs_lookback after extraction,
    with the two prior chunks' transcript and propositions attached.
    """
    if not lookback_column_available():
        return {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "lookback_max_chunks": settings.LOOKBACK_MAX_CHUNKS,
            "episode_count": 0,
            "unresolved_total": 0,
            "episodes": [],
            "lookback_column_available": False,
        }

    n = settings.LOOKBACK_MAX_CHUNKS
    prop_qs = Proposition.objects.filter(needs_lookback=True).select_related("chunk")
    claim_qs = Claim.objects.filter(needs_lookback=True).select_related("chunk")

    if episode_id is not None:
        prop_qs = prop_qs.filter(chunk__episode_id=episode_id)
        claim_qs = claim_qs.filter(chunk__episode_id=episode_id)

    episode_ids = set(
        prop_qs.values_list("chunk__episode_id", flat=True)
    ) | set(claim_qs.values_list("chunk__episode_id", flat=True))

    episodes_payload = []

    for ep_id in sorted(episode_ids):
        episode = Episode.objects.only("id", "title", "guest", "date").get(pk=ep_id)
        chunks = (
            Chunk.objects.filter(episode_id=ep_id)
            .prefetch_related(
                Prefetch(
                    "propositions",
                    queryset=Proposition.objects.only(
                        "id", "content", "source_text", "needs_lookback", "chunk_id"
                    ),
                ),
                Prefetch(
                    "claims",
                    queryset=Claim.objects.only(
                        "id", "content", "source_text", "needs_lookback", "chunk_id"
                    ),
                ),
            )
            .only("id", "chunk_index", "content", "token_estimate", "episode_id")
            .order_by("chunk_index")
        )
        chunks_by_index = {c.chunk_index: c for c in chunks}
        indices = sorted(chunks_by_index.keys())

        items = []
        for layer, qs in (("propositions", prop_qs.filter(chunk__episode_id=ep_id)), ("claims", claim_qs.filter(chunk__episode_id=ep_id))):
            for row in qs.order_by("chunk__chunk_index", "id"):
                chunk = chunks_by_index[row.chunk.chunk_index]
                prior = [
                    _chunk_export_block(chunks_by_index[idx])
                    for idx in _prior_indices(row.chunk.chunk_index, n, indices)
                ]
                items.append({
                    "layer": layer,
                    "item_id": row.id,
                    "content": row.content,
                    "source_text": row.source_text,
                    "start_char": row.start_char,
                    "end_char": row.end_char,
                    "chunk": _chunk_export_block(
                        chunk,
                        include_propositions=True,
                        include_claims=True,
                    ),
                    "prior_chunks": prior,
                })

        episodes_payload.append({
            "episode_id": episode.id,
            "episode_title": episode.title,
            "guest": episode.guest,
            "date": episode.date.isoformat() if episode.date else None,
            "unresolved_count": len(items),
            "items": items,
        })

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "lookback_max_chunks": n,
        "episode_count": len(episodes_payload),
        "unresolved_total": sum(ep["unresolved_count"] for ep in episodes_payload),
        "episodes": episodes_payload,
        "lookback_column_available": True,
    }
