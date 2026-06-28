import json
import time

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from apps.transcripts.models import Chunk, Episode
from apps.transcripts.services import embed_texts_as_lists
from extraction.prompts import EXTRACTION_USER_TEMPLATE
from extraction.lookback_pass import LookbackSummary
from extraction.service import (
    extract_layers_for_chunk,
    format_lookback_summary,
    mark_chunk_lookback_completed,
    run_incremental_lookback,
    save_extraction,
)


class Command(BaseCommand):
    help = (
        "Run layer extraction sequentially with lookback (debug). "
        "Optionally limit to chunk_index <= N."
    )

    def add_arguments(self, parser):
        parser.add_argument(
            "--episode-id",
            type=int,
            required=True,
            help="Episode to process.",
        )
        parser.add_argument(
            "up_to_index",
            type=int,
            nargs="?",
            default=None,
            help=(
                "Max chunk_index to run through, inclusive and 0-based. "
                "Example: 8 runs chunks 0–8. Omit to run the full episode."
            ),
        )
        parser.add_argument(
            "--show-prompt",
            action="store_true",
            help="Also print the user-message prompt sent to the model.",
        )
        parser.add_argument(
            "--apply",
            action="store_true",
            help="Save extracted layers to the database (default: dry run).",
        )
        parser.add_argument(
            "--output",
            type=str,
            default=None,
            help="Optional path to write the extraction trace as JSON.",
        )

    def _line(self, message=""):
        self.stdout.write(message)
        self.stdout.flush()

    def handle(self, *args, **options):
        episode_id = options["episode_id"]
        up_to_index = options["up_to_index"]

        try:
            episode = Episode.objects.get(pk=episode_id)
        except Episode.DoesNotExist:
            raise CommandError(f"Episode {episode_id} does not exist.")

        chunks = list(
            Chunk.objects.filter(episode_id=episode_id).order_by("chunk_index")
        )
        if not chunks:
            raise CommandError(f"No chunks found for episode {episode_id}.")

        if up_to_index is not None:
            chunks = [c for c in chunks if c.chunk_index <= up_to_index]
            if not chunks:
                raise CommandError(
                    f"No chunks with chunk_index <= {up_to_index} on episode {episode_id}."
                )

        self._line(
            f"Episode {episode_id} «{episode.title}» — "
            f"model {settings.EXTRACTION_MODEL}"
        )
        if up_to_index is not None:
            self._line(
                f"Running chunks 0 through {up_to_index} "
                f"({len(chunks)} chunk(s))"
            )
        else:
            self._line(f"Running all {len(chunks)} chunk(s)")
        self._line(f"Mode: {'SAVE to DB' if options['apply'] else 'dry run (no save)'}")
        self._line("")

        trace = []
        pending = []
        started = time.monotonic()
        chunks_by_index = {c.chunk_index: c for c in chunks}

        for index, chunk in enumerate(chunks, start=1):
            self._line(
                f"{'=' * 72}\n"
                f"CHUNK {chunk.chunk_index} (id {chunk.id}) — step {index}/{len(chunks)}"
            )

            if options["show_prompt"]:
                user_message = EXTRACTION_USER_TEMPLATE.format(
                    chunk_text=chunk.content,
                )
                self._line("--- User prompt ---")
                self._line(user_message)
                self._line("--- end prompt ---\n")

            chunk_started = time.monotonic()
            extracted, dropped = extract_layers_for_chunk(chunk)
            elapsed = time.monotonic() - chunk_started

            flagged_items = [
                {
                    "layer": layer,
                    "content": item["content"],
                }
                for layer in ("propositions", "claims")
                for item in extracted[layer]
                if item.get("needs_lookback")
            ]

            prop_n = len(extracted["propositions"])
            claim_n = len(extracted["claims"])
            phrase_n = len(extracted["phrases"])
            self._line(
                f"Extracted: {prop_n} proposition(s), {claim_n} claim(s), "
                f"{phrase_n} phrase(s) in {elapsed:.1f}s"
            )
            if dropped:
                self._line(f"Dropped {dropped} item(s) with no locatable source span")
            if flagged_items:
                self._line(f"Flagged for lookback ({len(flagged_items)}):")
                for item in flagged_items:
                    self._line(f"  [{item['layer']}] {item['content']}")
            self._line("")

            pending.append({
                "chunk": chunk,
                "extracted": extracted,
                "dropped": dropped,
            })

            trace.append({
                "chunk_id": chunk.id,
                "chunk_index": chunk.chunk_index,
                "flagged_for_lookback": flagged_items,
                "counts": {
                    "propositions": prop_n,
                    "claims": claim_n,
                    "phrases": phrase_n,
                },
                "dropped_span_items": dropped,
                "elapsed_seconds": round(elapsed, 2),
            })

        lookback = LookbackSummary(initiated=False, flagged_count=0)
        finalized_by_index: dict[int, dict] = {}

        for entry in sorted(pending, key=lambda e: e["chunk"].chunk_index):
            chunk_summary = run_incremental_lookback(
                entry,
                finalized_by_index,
                chunks_by_index,
                episode_id,
            )
            lookback = lookback.merge(chunk_summary)

            if options["apply"]:
                save_extraction(entry["chunk"], entry["extracted"], embed_texts_as_lists)
                mark_chunk_lookback_completed(entry["chunk"])
                self._line(f"Saved chunk {entry['chunk'].id} to database.")

        self._line(format_lookback_summary(lookback))
        if lookback.still_unresolved:
            self._line(f"Still unresolved: {lookback.still_unresolved}")
        self._line("")

        total_elapsed = time.monotonic() - started
        self._line(
            self.style.SUCCESS(
                f"Done — {len(chunks)} chunk(s) in {total_elapsed / 60:.1f} min"
            )
        )

        if options["output"]:
            output_path = options["output"]
            payload = {
                "episode_id": episode_id,
                "episode_title": episode.title,
                "extraction_model": settings.EXTRACTION_MODEL,
                "up_to_index": up_to_index,
                "chunk_count": len(chunks),
                "lookback": lookback.as_dict(),
                "steps": trace,
            }
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
            self._line(f"Trace written to {output_path}")
