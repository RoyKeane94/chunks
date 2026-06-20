import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from django.conf import settings
from django.core.management.base import BaseCommand
from django.db import close_old_connections, connection
from django.db.models import Count, Q
from django.db.utils import OperationalError

from apps.transcripts.models import Chunk
from apps.transcripts.services import embed_texts_as_lists
from extraction.service import extract_layers_for_chunk, save_extraction

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
                "Parallel chunk workers (default: EXTRACT_CHUNK_WORKERS env or 4). "
                "Use 1 for sequential processing."
            ),
        )

    def handle(self, *args, **options):
        if options["status"]:
            self._print_status(options["episode_id"])
            return

        workers = options["workers"] or settings.EXTRACT_CHUNK_WORKERS
        workers = max(1, workers)

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

        self._write_line(
            f"Processing {total} chunk(s) with {workers} worker(s)..."
        )

        processed = 0
        skipped = 0

        if workers == 1:
            for index, chunk_id in enumerate(chunk_ids, start=1):
                result = self._run_chunk_with_retries(chunk_id, index, total, options)
                if result == "processed":
                    processed += 1
                elif result == "skipped":
                    skipped += 1
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(
                        self._run_chunk_with_retries,
                        chunk_id,
                        index,
                        total,
                        options,
                    ): chunk_id
                    for index, chunk_id in enumerate(chunk_ids, start=1)
                }
                for future in as_completed(futures):
                    result = future.result()
                    if result == "processed":
                        processed += 1
                    elif result == "skipped":
                        skipped += 1

        self._write_line(
            f"Done — processed {processed} chunk(s), skipped {skipped} already extracted.",
            style=self.style.SUCCESS,
        )

    def _run_chunk_with_retries(self, chunk_id, index, total, options):
        chunk = Chunk.objects.select_related("episode").get(pk=chunk_id)
        self._write_line(
            f"[{index}/{total}] chunk {chunk.id} "
            f"(episode {chunk.episode_id}, index {chunk.chunk_index})"
        )

        for chunk_attempt in range(MAX_CHUNK_RETRIES):
            close_old_connections()
            connection.close()
            try:
                if self._process_chunk(chunk, options):
                    return "processed"
                return "skipped"
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

        return "skipped"

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

    def _process_chunk(self, chunk, options):
        chunk = Chunk.objects.select_related("episode").get(pk=chunk.pk)

        if options["reextract_zero_spans"]:
            if not self._needs_span_reextract(chunk):
                self._write_line(f"  chunk {chunk.id}: skipped (char spans already set)")
                return False
        elif not options["force"]:
            if self._is_already_extracted(chunk):
                self._write_line(f"  chunk {chunk.id}: skipped (already extracted)")
                return False

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
                self._write_line(f"  chunk {chunk.id}: cleared {deleted} existing layer row(s)")

        extracted, dropped = extract_layers_for_chunk(chunk)
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

        return True

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
        chunks = self._with_layer_counts(self._base_chunks(episode_id))
        total = chunks.count()

        done = chunks.filter(
            Q(extracted_at__isnull=False)
            | Q(proposition_count__gt=0)
            | Q(claim_count__gt=0)
            | Q(phrase_count__gt=0)
        )
        done_count = done.count()

        pending = chunks.filter(
            extracted_at__isnull=True,
            proposition_count=0,
            claim_count=0,
            phrase_count=0,
        )
        pending_count = pending.count()

        bad_spans = chunks.filter(self._bad_span_q()).distinct()
        bad_count = bad_spans.count()

        self.stdout.write(f"Total chunks:     {total}")
        self.stdout.write(f"Extracted:        {done_count}")
        self.stdout.write(f"Pending:          {pending_count}")
        self.stdout.write(
            f"Need re-extract:  {bad_count} (layer rows with start_char=0 and end_char=0)"
        )
        self.stdout.write(f"Default workers:  {settings.EXTRACT_CHUNK_WORKERS}")

        if done_count:
            last = done.order_by("-extracted_at", "-id").first()
            if last:
                self.stdout.write(
                    f"Last extracted:   chunk {last.id} "
                    f"(episode {last.episode_id} «{last.episode.title}», "
                    f"index {last.chunk_index})"
                )

        if bad_count:
            sample = bad_spans.order_by("id")[:10]
            ids = ", ".join(str(c.id) for c in sample)
            self.stdout.write(f"Bad-span sample:  chunk ids {ids}" + (" …" if bad_count > 10 else ""))

        self.stdout.write("")
        self.stdout.write("Continue pending:  python manage.py extract_chunk_layers")
        if bad_count:
            self.stdout.write(
                "Fix bad spans:     python manage.py extract_chunk_layers --reextract-zero-spans"
            )
