from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from apps.transcripts.models import AtomicPhrase, Chunk, Claim, Episode, Proposition
from apps.transcripts.services import embed_texts_as_lists
from extraction.service import backfill_embeddings, count_missing_embeddings


class Command(BaseCommand):
    help = (
        "Find proposition, claim, phrase, and chunk rows missing embedded_at "
        "and embed them with the current EMBEDDING_MODEL."
    )

    def add_arguments(self, parser):
        parser.add_argument(
            "--episode-id",
            type=int,
            default=None,
            help="Only embed rows for one episode.",
        )
        parser.add_argument(
            "--status",
            action="store_true",
            help="Show missing-embedding counts and exit.",
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Show what would be embedded without calling the API.",
        )
        parser.add_argument(
            "--layers-only",
            action="store_true",
            help="Skip chunk transcript embeddings; only embed layer rows.",
        )
        parser.add_argument(
            "--batch-size",
            type=int,
            default=None,
            help=f"Texts per embedding API call (default: EMBED_BATCH_SIZE / {settings.EMBED_BATCH_SIZE}).",
        )

    def handle(self, *args, **options):
        episode_id = options["episode_id"]
        batch_size = options["batch_size"]

        if episode_id is not None:
            try:
                episode = Episode.objects.get(pk=episode_id)
            except Episode.DoesNotExist:
                raise CommandError(f"Episode {episode_id} does not exist.")
            chunk_qs = Chunk.objects.filter(episode_id=episode_id)
            scope = f"episode {episode_id} «{episode.title}»"
        else:
            chunk_qs = Chunk.objects.all()
            scope = "all episodes"

        counts = count_missing_embeddings(chunk_qs)
        total_missing = sum(counts.values())

        self.stdout.write(f"Scope: {scope}")
        self.stdout.write(f"Model: {settings.EMBEDDING_MODEL}")
        self.stdout.write(f"Missing embeddings:")
        self.stdout.write(f"  Chunks:         {counts['chunks']}")
        self.stdout.write(f"  Propositions:   {counts['propositions']}")
        self.stdout.write(f"  Claims:         {counts['claims']}")
        self.stdout.write(f"  Phrases:        {counts['phrases']}")
        self.stdout.write(f"  Total:          {total_missing}")

        if options["status"]:
            return

        if total_missing == 0:
            self.stdout.write(self.style.SUCCESS("Nothing to embed."))
            return

        if options["dry_run"]:
            self.stdout.write(self.style.WARNING("Dry run — no embeddings created."))
            return

        updated = {}

        if not options["layers_only"] and counts["chunks"]:
            self.stdout.write(f"Embedding {counts['chunks']} chunk(s)...")
            updated["chunks"] = backfill_embeddings(
                chunk_qs, embed_texts_as_lists, batch_size
            )

        for label, model, count in (
            ("propositions", Proposition, counts["propositions"]),
            ("claims", Claim, counts["claims"]),
            ("phrases", AtomicPhrase, counts["phrases"]),
        ):
            if not count:
                continue
            self.stdout.write(f"Embedding {count} {label}...")
            qs = model.objects.filter(chunk__in=chunk_qs)
            updated[label] = backfill_embeddings(qs, embed_texts_as_lists, batch_size)

        parts = [f"{n} {label}" for label, n in updated.items()]
        self.stdout.write(
            self.style.SUCCESS(f"Done — embedded {', '.join(parts)}.")
        )
