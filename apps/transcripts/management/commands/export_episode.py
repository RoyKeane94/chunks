import json
from datetime import datetime
from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError
from django.db.models import Count, Prefetch

from apps.transcripts.models import AtomicPhrase, Chunk, Claim, Episode, Proposition

LAYER_CHOICES = ("chunks", "propositions", "claims", "phrases", "all")


def _iso(dt):
    return dt.isoformat() if dt else None


def _chunk_meta(chunk, include_counts=False):
    data = {
        "chunk_id": chunk.id,
        "chunk_index": chunk.chunk_index,
        "token_estimate": chunk.token_estimate,
    }
    if include_counts:
        data["proposition_count"] = chunk.proposition_count
        data["claim_count"] = chunk.claim_count
        data["phrase_count"] = chunk.phrase_count
    return data


def _layer_row(row, chunk, include_embeddings=False):
    data = {
        **_chunk_meta(chunk),
        "id": row.id,
        "content": row.content,
        "source_text": row.source_text,
        "start_char": row.start_char,
        "end_char": row.end_char,
        "extraction_model": row.extraction_model,
        "embedding_model": row.embedding_model,
        "embedded_at": _iso(row.embedded_at),
        "created_at": _iso(row.created_at),
    }
    if include_embeddings:
        data["embedding"] = row.embedding
    return data


def _chunk_row(chunk, include_embeddings=False):
    if include_embeddings:
        return {
            "chunk_id": chunk.id,
            "chunk_index": chunk.chunk_index,
            "content": chunk.content,
            "embedding": chunk.embedding,
        }
    return {
        "chunk_id": chunk.id,
        "chunk_index": chunk.chunk_index,
        "content": chunk.content,
    }


class Command(BaseCommand):
    help = "Export a full episode as JSON at chunks, propositions, claims, or phrases."

    def add_arguments(self, parser):
        parser.add_argument(
            "--episode-id",
            type=int,
            required=True,
            help="Episode to export.",
        )
        parser.add_argument(
            "--layer",
            type=str,
            required=True,
            choices=LAYER_CHOICES,
            help=(
                "What to export: chunks, propositions, claims, phrases, "
                "or all (nested by chunk)."
            ),
        )
        parser.add_argument(
            "--output",
            type=str,
            default=None,
            help="Output JSON file path (default: export_ep{id}_{layer}_{timestamp}.json).",
        )
        parser.add_argument(
            "--output-dir",
            type=str,
            default=None,
            help="Directory for the JSON file (default: project root).",
        )
        parser.add_argument(
            "--include-embeddings",
            action="store_true",
            help="Include 1536-dim embedding vectors (large files).",
        )

    def handle(self, *args, **options):
        episode_id = options["episode_id"]
        layer = options["layer"]
        include_embeddings = options["include_embeddings"]

        try:
            episode = Episode.objects.get(pk=episode_id)
        except Episode.DoesNotExist:
            raise CommandError(f"Episode {episode_id} does not exist.")

        chunks = (
            Chunk.objects.filter(episode_id=episode_id)
            .annotate(
                proposition_count=Count("propositions", distinct=True),
                claim_count=Count("claims", distinct=True),
                phrase_count=Count("atomic_phrases", distinct=True),
            )
            .order_by("chunk_index")
        )
        if not include_embeddings:
            chunks = chunks.defer("embedding")

        layer_qs = {
            "propositions": Proposition.objects.order_by("start_char", "id"),
            "claims": Claim.objects.order_by("start_char", "id"),
            "phrases": AtomicPhrase.objects.order_by("start_char", "id"),
        }
        if not include_embeddings:
            layer_qs = {
                key: qs.defer("embedding") for key, qs in layer_qs.items()
            }

        if layer == "all":
            chunks = chunks.prefetch_related(
                Prefetch("propositions", queryset=layer_qs["propositions"]),
                Prefetch("claims", queryset=layer_qs["claims"]),
                Prefetch("atomic_phrases", queryset=layer_qs["phrases"]),
            )
        elif layer == "chunks":
            pass
        elif layer == "propositions":
            chunks = chunks.prefetch_related(
                Prefetch("propositions", queryset=layer_qs["propositions"]),
            )
        elif layer == "claims":
            chunks = chunks.prefetch_related(
                Prefetch("claims", queryset=layer_qs["claims"]),
            )
        elif layer == "phrases":
            chunks = chunks.prefetch_related(
                Prefetch("atomic_phrases", queryset=layer_qs["phrases"]),
            )

        chunk_list = list(chunks)
        if not chunk_list:
            raise CommandError(f"No chunks found for episode {episode_id}.")

        payload = {
            "episode_id": episode.id,
            "episode_title": episode.title,
            "episode_guest": episode.guest,
            "episode_date": episode.date.isoformat() if episode.date else None,
            "layer": layer,
            "exported_at": datetime.now().isoformat(timespec="seconds"),
            "chunk_count": len(chunk_list),
        }

        if layer == "chunks":
            payload["items"] = [
                _chunk_row(chunk, include_embeddings=include_embeddings)
                for chunk in chunk_list
            ]
        elif layer == "propositions":
            payload["item_count"] = sum(c.proposition_count for c in chunk_list)
            payload["items"] = [
                _layer_row(row, chunk, include_embeddings=include_embeddings)
                for chunk in chunk_list
                for row in chunk.propositions.all()
            ]
        elif layer == "claims":
            payload["item_count"] = sum(c.claim_count for c in chunk_list)
            payload["items"] = [
                _layer_row(row, chunk, include_embeddings=include_embeddings)
                for chunk in chunk_list
                for row in chunk.claims.all()
            ]
        elif layer == "phrases":
            payload["item_count"] = sum(c.phrase_count for c in chunk_list)
            payload["items"] = [
                _layer_row(row, chunk, include_embeddings=include_embeddings)
                for chunk in chunk_list
                for row in chunk.atomic_phrases.all()
            ]
        elif layer == "all":
            payload["chunks"] = []
            for chunk in chunk_list:
                entry = {
                    **_chunk_row(chunk, include_embeddings=include_embeddings),
                    "propositions": [
                        _layer_row(row, chunk, include_embeddings=include_embeddings)
                        for row in chunk.propositions.all()
                    ],
                    "claims": [
                        _layer_row(row, chunk, include_embeddings=include_embeddings)
                        for row in chunk.claims.all()
                    ],
                    "phrases": [
                        _layer_row(row, chunk, include_embeddings=include_embeddings)
                        for row in chunk.atomic_phrases.all()
                    ],
                }
                payload["chunks"].append(entry)

        if options["output"]:
            output_path = Path(options["output"])
        else:
            output_dir = Path(options["output_dir"] or settings.BASE_DIR)
            output_dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = output_dir / f"export_ep{episode_id}_{layer}_{stamp}.json"

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        if layer == "all":
            summary = (
                f"{len(payload['chunks'])} chunk(s) with nested layers"
            )
        else:
            count = len(payload.get("items", []))
            summary = f"{count} {layer} item(s)"

        self.stdout.write(
            self.style.SUCCESS(
                f"Exported episode «{episode.title}» — {summary}\n  {output_path}"
            )
        )
