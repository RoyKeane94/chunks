from django.core.management.base import BaseCommand

from apps.transcripts.services import backfill_episode_content_hashes


class Command(BaseCommand):
    help = "Compute and store content_hash for episodes uploaded before duplicate detection."

    def handle(self, *args, **options):
        updated, skipped = backfill_episode_content_hashes()
        self.stdout.write(
            self.style.SUCCESS(f"Updated {updated} episode(s), skipped {skipped}.")
        )
