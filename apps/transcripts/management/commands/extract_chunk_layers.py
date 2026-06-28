import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from django.conf import settings
from django.core.management.base import BaseCommand
from django.db import close_old_connections, connection
from django.db.models import Count, Q
from django.db.utils import OperationalError

from apps.transcripts.layer_status import get_extraction_summary
from apps.transcripts.models import Chunk, Episode
from apps.transcripts.services import embed_texts_as_lists
from extraction.service import (
    clear_chunk_lookback_completed,
    extract_layers_for_chunk,
    format_lookback_summary,
    mark_chunk_lookback_completed,
    register_finalised_predecessor,
    run_incremental_lookback,
    save_extraction,
)

MAX_CHUNK_RETRIES = 3


class Command(BaseCommand):
    help = "Extract proposition, claim, and atomic phrase layers for chunks."

    def _bad_span_q(self):
        """Layer rows with unset spans from legacy runs (both start and end are zero)."""
        return (
            Q(propositions__start_char=0, propositions__end_char=0)
            | Q(claims__start_char=0, claims__end_char=0)
            | Q(atomic_phrases__start_char=0, atomic_phrases__end_char=0)
        )

    def _write_line(self, message="", style=None):
        """Write a line and flush so piped output (tee) matches interactive output."""
        if style:
            self.stdout.write(style(message))
        else:
            self.stdout.write(message)
        self.stdout.flush()

    def add_arguments(self, parser):
        parser.add_argument(
            "--episode-id",
            type=int,
            default=None,
            help="Process only chunks belonging to this episode.",
        )
        parser.add_argument(
            "--status",
            action="store_true",
            help="Show extraction progress and exit (no processing).",
        )
        parser.add_argument(
            "--force",
            action="store_true",
            help="Re-extract chunks that already have layers (deletes existing rows first).",
        )
        parser.add_argument(
            "--reextract-zero-spans",
            action="store_true",
            help=(
                "Re-extract chunks that have layer rows with unset char spans "
                "(start_char=0 and end_char=0 from an older run)."
            ),
        )
        parser.add_argument(
            "--workers",
            type=int,
            default=None,
            help=(
                "Parallel episode workers (default: EXTRACT_CHUNK_WORKERS env or 4). "
                "Each episode runs extract → lookback → save independently."
            ),
        )
        parser.add_argument(
            "--chunk-workers",
            type=int,
            default=None,
            help=(
                "Parallel extraction workers within each episode (default: "
                "EXTRACT_CHUNK_PARALLEL env or 4). Lookback still runs sequentially "
                "in chunk_index order after all extractions finish."
            ),
        )

    def handle(self, *args, **options):
        if options["status"]:
            self._print_status(options["episode_id"])
            return

        workers = options["workers"] or settings.EXTRACT_CHUNK_WORKERS
        workers = max(1, workers)
        chunk_workers = options["chunk_workers"] or settings.EXTRACT_CHUNK_PARALLEL
        chunk_workers = max(1, chunk_workers)
        options["chunk_workers"] = chunk_workers

        chunks = self._chunks_queryset(
            episode_id=options["episode_id"],
            force=options["force"],
            reextract_zero_spans=options["reextract_zero_spans"],
        )

        chunk_ids = list(
            chunks.order_by("episode_id", "chunk_index").values_list("id", flat=True)
        )
        total = len(chunk_ids)
        if total == 0:
            self.stdout.write("No chunks to process.")
            self._print_status(options["episode_id"])
            return

        episode_chunk_ids = defaultdict(list)
        for row in (
            Chunk.objects.filter(id__in=chunk_ids)
            .order_by("episode_id", "chunk_index")
            .values_list("id", "episode_id")
        ):
            chunk_id, episode_id = row
            episode_chunk_ids[episode_id].append(chunk_id)

        self._write_line(
            f"Processing {total} chunk(s) across {len(episode_chunk_ids)} episode(s) "
            f"with {workers} episode worker(s), {chunk_workers} extract worker(s) per episode..."
        )

        processed = 0
        skipped = 0

        if workers == 1:
            for episode_id, ids in sorted(episode_chunk_ids.items()):
                ep_processed, ep_skipped = self._process_episode(
                    episode_id, ids, options, total
                )
                processed += ep_processed
                skipped += ep_skipped
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(
                        self._process_episode,
                        episode_id,
                        ids,
                        options,
                        total,
                    ): episode_id
                    for episode_id, ids in episode_chunk_ids.items()
                }
                for future in as_completed(futures):
                    ep_processed, ep_skipped = future.result()
                    processed += ep_processed
                    skipped += ep_skipped

        self._write_line(
            f"Done — processed {processed} chunk(s), skipped {skipped} already extracted.",
            style=self.style.SUCCESS,
        )

    def _process_episode(self, episode_id, chunk_ids, options, total_chunks):
        close_old_connections()
        episode = Episode.objects.get(pk=episode_id)
        chunk_workers = options["chunk_workers"]
        self._write_line(
            f"Episode {episode_id} «{episode.title}»"
        )

        chunks_by_index = {
            c.chunk_index: c
            for c in Chunk.objects.filter(episode_id=episode_id).only(
                "id", "chunk_index", "content"
            )
        }

        finalized_by_index: dict[int, dict] = {}
        extract_queue: list[tuple[int, int]] = []
        skipped = 0

        for index, chunk_id in enumerate(chunk_ids, start=1):
            chunk = Chunk.objects.get(pk=chunk_id)
            if self._should_skip_chunk(chunk, options):
                self._write_line(
                    f"[{index}/{len(chunk_ids)}] chunk {chunk.id} "
                    f"(index {chunk.chunk_index}) — skipped"
                )
                register_finalised_predecessor(chunk, finalized_by_index)
                skipped += 1
            else:
                extract_queue.append((index, chunk_id))

        pending: list[dict] = []
        if extract_queue:
            self._write_line(
                f"  extracting {len(extract_queue)} chunk(s) "
                f"with {min(chunk_workers, len(extract_queue))} worker(s)..."
            )
            pending = self._parallel_extract(extract_queue, options, len(chunk_ids))

        lookback_summary = None
        if pending:
            pending.sort(key=lambda e: e["chunk"].chunk_index)
            self._write_line(
                f"  lookback + save for {len(pending)} chunk(s) (sequential)..."
            )
            for entry in pending:
                chunk_summary = run_incremental_lookback(
                    entry,
                    finalized_by_index,
                    chunks_by_index,
                    episode_id,
                )
                if lookback_summary is None:
                    lookback_summary = chunk_summary
                else:
                    lookback_summary = lookback_summary.merge(chunk_summary)

                self._save_and_log_chunk(entry)
                mark_chunk_lookback_completed(entry["chunk"])

        if lookback_summary and lookback_summary.initiated:
            self._write_line(
                f"Episode {episode_id}: {format_lookback_summary(lookback_summary)}",
                style=self.style.WARNING if lookback_summary.still_unresolved else None,
            )

        return len(pending), skipped

    def _parallel_extract(
        self,
        extract_queue: list[tuple[int, int]],
        options: dict,
        total_in_episode: int,
    ) -> list[dict]:
        chunk_workers = min(options["chunk_workers"], len(extract_queue))
        pending: list[dict] = []

        if chunk_workers == 1:
            for index, chunk_id in extract_queue:
                entry = self._extract_with_retries(chunk_id, index, total_in_episode, options)
                if entry:
                    pending.append(entry)
            return pending

        with ThreadPoolExecutor(max_workers=chunk_workers) as executor:
            futures = {
                executor.submit(
                    self._extract_with_retries,
                    chunk_id,
                    index,
                    total_in_episode,
                    options,
                ): chunk_id
                for index, chunk_id in extract_queue
            }
            for future in as_completed(futures):
                chunk_id = futures[future]
                try:
                    entry = future.result()
                    if entry:
                        pending.append(entry)
                except Exception:
                    self._write_line(
                        f"  chunk {chunk_id}: extraction failed",
                        style=self.style.ERROR,
                    )
                    raise

        return pending

    def _should_skip_chunk(self, chunk, options):
        if options["reextract_zero_spans"]:
            return not self._needs_span_reextract(chunk)
        if not options["force"]:
            return self._is_already_extracted(chunk)
        return False

    def _extract_with_retries(self, chunk_id, index, total, options):
        chunk = Chunk.objects.select_related("episode").get(pk=chunk_id)
        self._write_line(
            f"[{index}/{total}] chunk {chunk.id} "
            f"(episode {chunk.episode_id}, index {chunk.chunk_index}) — extracting"
        )

        for chunk_attempt in range(MAX_CHUNK_RETRIES):
            close_old_connections()
            connection.close()
            try:
                entry = self._extract_chunk(chunk, options)
                if entry:
                    extracted = entry["extracted"]
                    flagged = sum(
                        1
                        for layer in ("propositions", "claims")
                        for item in extracted[layer]
                        if item.get("needs_lookback")
                    )
                    flag_note = f", lookback {flagged}" if flagged else ""
                    self._write_line(
                        f"  ✓ chunk {chunk.id}: "
                        f"{len(extracted['propositions'])} prop / "
                        f"{len(extracted['claims'])} claim / "
                        f"{len(extracted['phrases'])} phrase{flag_note}"
                    )
                return entry
            except OperationalError as exc:
                if chunk_attempt >= MAX_CHUNK_RETRIES - 1:
                    raise
                wait = min(2 ** chunk_attempt, 30)
                self._write_line(
                    f"  chunk {chunk.id}: DB connection lost ({exc}) — "
                    f"retrying in {wait}s "
                    f"(attempt {chunk_attempt + 2}/{MAX_CHUNK_RETRIES})",
                    style=self.style.WARNING,
                )
                time.sleep(wait)

        return None

    def _is_already_extracted(self, chunk):
        if chunk.extracted_at is not None:
            return True
        return (
            chunk.propositions.exists()
            or chunk.claims.exists()
            or chunk.atomic_phrases.exists()
        )

    def _needs_span_reextract(self, chunk):
        return (
            chunk.propositions.filter(start_char=0, end_char=0).exists()
            or chunk.claims.filter(start_char=0, end_char=0).exists()
            or chunk.atomic_phrases.filter(start_char=0, end_char=0).exists()
        )

    def _extract_chunk(self, chunk, options):
        chunk = Chunk.objects.select_related("episode").get(pk=chunk.pk)

        if options["force"] or options["reextract_zero_spans"]:
            deleted = (
                chunk.propositions.count()
                + chunk.claims.count()
                + chunk.atomic_phrases.count()
            )
            if deleted:
                chunk.propositions.all().delete()
                chunk.claims.all().delete()
                chunk.atomic_phrases.all().delete()
                clear_chunk_lookback_completed(chunk)
                self._write_line(f"  chunk {chunk.id}: cleared {deleted} existing layer row(s)")

        extracted, dropped = extract_layers_for_chunk(chunk)

        flagged = sum(
            1
            for layer in ("propositions", "claims")
            for item in extracted[layer]
            if item.get("needs_lookback")
        )
        if flagged:
            self._write_line(
                f"  chunk {chunk.id}: {flagged} item(s) flagged for lookback"
            )

        return {
            "chunk": chunk,
            "extracted": extracted,
            "dropped": dropped,
        }

    def _save_and_log_chunk(self, entry):
        chunk = entry["chunk"]
        extracted = entry["extracted"]
        dropped = entry["dropped"]

        extracted, redundant_dropped = save_extraction(
            chunk, extracted, embed_texts_as_lists
        )

        prop_count = len(extracted["propositions"])
        claim_count = len(extracted["claims"])
        phrase_count = len(extracted["phrases"])

        self._write_line(
            f"  chunk {chunk.id}: saved {prop_count} proposition(s), "
            f"{claim_count} claim(s), {phrase_count} phrase(s)"
        )
        if dropped:
            self._write_line(
                f"  chunk {chunk.id}: dropped {dropped} item(s) with no locatable source span"
            )
        if redundant_dropped:
            self._write_line(
                f"  chunk {chunk.id}: dropped {redundant_dropped} redundant claim(s)"
            )

    def _base_chunks(self, episode_id):
        chunks = Chunk.objects.select_related("episode")
        if episode_id:
            chunks = chunks.filter(episode_id=episode_id)
        return chunks

    def _with_layer_counts(self, chunks):
        return chunks.annotate(
            proposition_count=Count("propositions", distinct=True),
            claim_count=Count("claims", distinct=True),
            phrase_count=Count("atomic_phrases", distinct=True),
        )

    def _chunks_queryset(self, episode_id, force, reextract_zero_spans):
        chunks = self._base_chunks(episode_id)

        if force:
            return chunks

        if reextract_zero_spans:
            return chunks.filter(self._bad_span_q()).distinct()

        return self._with_layer_counts(chunks).filter(
            extracted_at__isnull=True,
            proposition_count=0,
            claim_count=0,
            phrase_count=0,
        )

    def _print_status(self, episode_id):
        summary = get_extraction_summary()
        chunks = self._with_layer_counts(self._base_chunks(episode_id))
        total = chunks.count()

        bad_spans = chunks.filter(self._bad_span_q()).distinct()
        bad_count = bad_spans.count()

        if episode_id:
            ep = Episode.objects.filter(pk=episode_id).first()
            ep_label = f"episode {episode_id} «{ep.title}»" if ep else f"episode {episode_id}"
            self.stdout.write(f"Scope: {ep_label}")
            done_count = chunks.filter(
                Q(extracted_at__isnull=False)
                | Q(proposition_count__gt=0)
                | Q(claim_count__gt=0)
                | Q(phrase_count__gt=0)
            ).count()
            pending_count = total - done_count
            self.stdout.write(f"Chunks in scope:  {total}")
            self.stdout.write(f"  layered:        {done_count}")
            self.stdout.write(f"  pending:        {pending_count}")
        else:
            self.stdout.write("Library-wide extraction status")
            self.stdout.write(f"Episodes fully layered:   {summary['episodes_fully_layered']}")
            self.stdout.write(
                f"Episodes partially layered: {summary['episodes_partially_layered']}"
            )
            self.stdout.write(f"Episodes not started:     {summary['episodes_not_started']}")
            self.stdout.write(f"Chunks layered:         {summary['chunks_layered']}")
            self.stdout.write(f"Chunks pending:         {summary['chunks_pending']}")
            self.stdout.write(
                f"Unresolved lookback:      {summary['unresolved_total']} "
                f"({summary['episodes_with_unresolved']} episodes)"
            )

        self.stdout.write(
            f"Need re-extract (bad spans): {bad_count} "
            "(layer rows with start_char=0 and end_char=0)"
        )
        self.stdout.write(f"Episode workers:        {settings.EXTRACT_CHUNK_WORKERS}")
        self.stdout.write(f"Extract parallel:       {settings.EXTRACT_CHUNK_PARALLEL}")

        if bad_count:
            sample = bad_spans.order_by("id")[:10]
            ids = ", ".join(str(c.id) for c in sample)
            self.stdout.write(f"Bad-span sample:  chunk ids {ids}" + (" …" if bad_count > 10 else ""))

        self.stdout.write("")
        self.stdout.write("Continue pending:  python manage.py extract_chunk_layers")
        self.stdout.write("Full status:       /library/status/")
        if bad_count:
            self.stdout.write(
                "Fix bad spans:     python manage.py extract_chunk_layers --reextract-zero-spans"
            )
