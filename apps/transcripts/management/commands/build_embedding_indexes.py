import time

from django.core.management.base import BaseCommand
from django.db import connection


# Cosine distance — matches CosineDistance in retrieval.py.
EMBEDDING_INDEX_PLANS = [
    {
        "label": "chunks",
        "table": "transcripts_chunk",
        "hnsw_name": "chunk_embedding_hnsw_idx",
        "ivfflat_name": "chunk_embedding_ivfflat_idx",
        "ivfflat_lists": 50,
    },
    {
        "label": "propositions",
        "table": "transcripts_proposition",
        "hnsw_name": "proposition_embedding_hnsw_idx",
        "ivfflat_name": "proposition_embedding_ivfflat_idx",
        "ivfflat_lists": 100,
    },
    {
        "label": "claims",
        "table": "transcripts_claim",
        "hnsw_name": "claim_embedding_hnsw_idx",
        "ivfflat_name": "claim_embedding_ivfflat_idx",
        "ivfflat_lists": 50,
    },
    {
        "label": "phrases",
        "table": "transcripts_atomicphrase",
        "hnsw_name": "phrase_embedding_hnsw_idx",
        "ivfflat_name": "phrase_embedding_ivfflat_idx",
        "ivfflat_lists": 100,
    },
]

HNSW_M = 8
HNSW_EF_CONSTRUCTION = 32
DEFAULT_MAINTENANCE_WORK_MEM = "128MB"


def _write_line(stdout, text=""):
    stdout.write(text)
    stdout.flush()


class Command(BaseCommand):
    help = (
        "Build pgvector indexes on embedding columns for fast retrieval. "
        "Tries HNSW first (low-memory settings); falls back to IVFFlat on Railway-sized Postgres."
    )

    def add_arguments(self, parser):
        parser.add_argument(
            "--status",
            action="store_true",
            help="List embedding indexes and exit.",
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Show what would be built without running CREATE INDEX.",
        )
        parser.add_argument(
            "--maintenance-work-mem",
            default=DEFAULT_MAINTENANCE_WORK_MEM,
            help=f"SET maintenance_work_mem before each build (default: {DEFAULT_MAINTENANCE_WORK_MEM}).",
        )
        parser.add_argument(
            "--ivfflat-only",
            action="store_true",
            help="Skip HNSW and build IVFFlat indexes only.",
        )

    def handle(self, *args, **options):
        if connection.vendor != "postgresql":
            self.stderr.write(self.style.ERROR("PostgreSQL with pgvector is required."))
            return

        with connection.cursor() as cursor:
            if options["status"]:
                self._print_status(cursor)
                return

            built = 0
            skipped = 0
            failed = 0

            for plan in EMBEDDING_INDEX_PLANS:
                existing = self._embedding_indexes_on_table(cursor, plan["table"])
                if existing:
                    _write_line(
                        self.stdout,
                        f"skip {plan['label']} — already indexed: {', '.join(existing)}",
                    )
                    skipped += 1
                    continue

                if options["dry_run"]:
                    strategy = "ivfflat" if options["ivfflat_only"] else "hnsw (then ivfflat fallback)"
                    _write_line(
                        self.stdout,
                        f"would build {plan['label']} on {plan['table']} via {strategy}",
                    )
                    continue

                _write_line(self.stdout, f"building index for {plan['label']} ({plan['table']})...")
                ok, index_name, strategy = self._build_index(
                    cursor,
                    plan,
                    options["maintenance_work_mem"],
                    ivfflat_only=options["ivfflat_only"],
                )
                if ok:
                    built += 1
                    _write_line(
                        self.stdout,
                        self.style.SUCCESS(f"  done — {index_name} ({strategy})"),
                    )
                else:
                    failed += 1
                    _write_line(
                        self.stdout,
                        self.style.ERROR(f"  failed — could not build index for {plan['label']}"),
                    )

                time.sleep(2)

            if not options["dry_run"]:
                _write_line(
                    self.stdout,
                    self.style.SUCCESS(
                        f"Finished — built {built}, skipped {skipped}, failed {failed}."
                    ),
                )

    def _print_status(self, cursor):
        _write_line(self.stdout, "Embedding indexes:")
        for plan in EMBEDDING_INDEX_PLANS:
            names = self._embedding_indexes_on_table(cursor, plan["table"])
            if names:
                _write_line(self.stdout, f"  {plan['label']}: {', '.join(names)}")
            else:
                _write_line(self.stdout, f"  {plan['label']}: none")

    def _embedding_indexes_on_table(self, cursor, table):
        cursor.execute(
            """
            SELECT indexname
            FROM pg_indexes
            WHERE schemaname = 'public'
              AND tablename = %s
              AND indexdef LIKE %s
            ORDER BY indexname
            """,
            [table, "%embedding%"],
        )
        return [row[0] for row in cursor.fetchall()]

    def _index_exists(self, cursor, index_name):
        cursor.execute(
            "SELECT 1 FROM pg_indexes WHERE schemaname = 'public' AND indexname = %s",
            [index_name],
        )
        return cursor.fetchone() is not None

    def _set_work_mem(self, cursor, value):
        cursor.execute("SET maintenance_work_mem = %s", [value])

    def _build_index(self, cursor, plan, maintenance_work_mem, ivfflat_only):
        self._set_work_mem(cursor, maintenance_work_mem)

        if not ivfflat_only:
            ok = self._try_hnsw(cursor, plan)
            if ok:
                return True, plan["hnsw_name"], "hnsw"

        ok = self._try_ivfflat(cursor, plan)
        if ok:
            return True, plan["ivfflat_name"], "ivfflat"

        return False, None, None

    def _try_hnsw(self, cursor, plan):
        if self._index_exists(cursor, plan["hnsw_name"]):
            return True

        sql = (
            f"CREATE INDEX CONCURRENTLY {plan['hnsw_name']} "
            f"ON {plan['table']} USING hnsw (embedding vector_cosine_ops) "
            f"WITH (m = {HNSW_M}, ef_construction = {HNSW_EF_CONSTRUCTION})"
        )
        try:
            cursor.execute(sql)
            return True
        except Exception as exc:
            _write_line(self.stdout, f"  hnsw failed: {exc}")
            return False

    def _try_ivfflat(self, cursor, plan):
        if self._index_exists(cursor, plan["ivfflat_name"]):
            return True

        sql = (
            f"CREATE INDEX CONCURRENTLY {plan['ivfflat_name']} "
            f"ON {plan['table']} USING ivfflat (embedding vector_cosine_ops) "
            f"WITH (lists = {plan['ivfflat_lists']})"
        )
        try:
            cursor.execute(sql)
            return True
        except Exception as exc:
            _write_line(self.stdout, f"  ivfflat failed: {exc}")
            return False
