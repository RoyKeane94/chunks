from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("transcripts", "0008_chunk_lookback_completed_at"),
    ]

    operations = [
        migrations.SeparateDatabaseAndState(
            state_operations=[
                migrations.AddField(
                    model_name="proposition",
                    name="needs_lookback",
                    field=models.BooleanField(default=False),
                ),
                migrations.AddField(
                    model_name="claim",
                    name="needs_lookback",
                    field=models.BooleanField(default=False),
                ),
            ],
            database_operations=[
                migrations.RunSQL(
                    sql="""
                        SET lock_timeout = '10s';
                        ALTER TABLE transcripts_proposition
                            ADD COLUMN IF NOT EXISTS needs_lookback boolean NOT NULL DEFAULT false;
                        ALTER TABLE transcripts_claim
                            ADD COLUMN IF NOT EXISTS needs_lookback boolean NOT NULL DEFAULT false;
                    """,
                    reverse_sql="""
                        ALTER TABLE transcripts_proposition DROP COLUMN IF EXISTS needs_lookback;
                        ALTER TABLE transcripts_claim DROP COLUMN IF EXISTS needs_lookback;
                    """,
                ),
            ],
        ),
    ]
