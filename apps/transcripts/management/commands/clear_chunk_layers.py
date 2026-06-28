from django.core.management.base import BaseCommand, CommandError
from django.db.models import Count, Q

from apps.transcripts.models import AtomicPhrase, Chunk, Claim, Episode, Proposition


class Command(BaseCommand):
    help = (
        "Delete all proposition, claim, and atomic phrase rows. "
        "Chunks and their transcript embeddings are kept."
    )

    def add_arguments(self, parser):
        parser.add_argument(
            "--episode-id",
            type=int,
            default=None,
            help="Clear layers for one episode only.",
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Show counts without deleting.",
        )

    def handle(self, *args, **options):
        episode_id = options["episode_id"]
        dry_run = options["dry_run"]

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

        prop_qs = Proposition.objects.filter(chunk__in=chunk_qs)
        claim_qs = Claim.objects.filter(chunk__in=chunk_qs)
        phrase_qs = AtomicPhrase.objects.filter(chunk__in=chunk_qs)

        prop_count = prop_qs.count()
        claim_count = claim_qs.count()
        phrase_count = phrase_qs.count()
        chunk_count = chunk_qs.count()
        chunks_with_layers = (
            chunk_qs.annotate(
                proposition_count=Count("propositions", distinct=True),
                claim_count=Count("claims", distinct=True),
                phrase_count=Count("atomic_phrases", distinct=True),
            )
            .filter(
                Q(proposition_count__gt=0)
                | Q(claim_count__gt=0)
                | Q(phrase_count__gt=0)
            )
            .count()
        )

        self.stdout.write(f"Scope: {scope}")
        self.stdout.write(f"Chunks:              {chunk_count}")
        self.stdout.write(f"Chunks with layers:  {chunks_with_layers}")
        self.stdout.write(f"Propositions:        {prop_count}")
        self.stdout.write(f"Claims:              {claim_count}")
        self.stdout.write(f"Atomic phrases:      {phrase_count}")

        total_layers = prop_count + claim_count + phrase_count
        if total_layers == 0:
            self.stdout.write(self.style.WARNING("Nothing to delete."))
            return

        if dry_run:
            self.stdout.write(self.style.WARNING("Dry run — no rows deleted."))
            return

        deleted_props, _ = prop_qs.delete()
        deleted_claims, _ = claim_qs.delete()
        deleted_phrases, _ = phrase_qs.delete()

        cleared_chunks = chunk_qs.update(
            extracted_at=None,
            extraction_model="",
            lookback_completed_at=None,
        )

        self.stdout.write(
            self.style.SUCCESS(
                f"Deleted {deleted_props} proposition(s), "
                f"{deleted_claims} claim(s), "
                f"{deleted_phrases} phrase(s). "
                f"Cleared extraction metadata on {cleared_chunks} chunk(s)."
            )
        )
