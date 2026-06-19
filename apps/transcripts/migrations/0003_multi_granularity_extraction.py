# Generated manually for multi-granularity extraction layer

import django.db.models.deletion
import pgvector.django.vector
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("transcripts", "0002_atomic_claim_proposition"),
    ]

    operations = [
        migrations.DeleteModel(
            name="AtomicClaim",
        ),
        migrations.DeleteModel(
            name="Proposition",
        ),
        migrations.CreateModel(
            name="Proposition",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("content", models.TextField()),
                ("source_text", models.TextField()),
                ("embedding", pgvector.django.vector.VectorField(dimensions=1536)),
                ("embedding_model", models.CharField(default="text-embedding-3-small", max_length=64)),
                ("extraction_model", models.CharField(default="gpt-4o-mini", max_length=64)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                (
                    "chunk",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="propositions",
                        to="transcripts.chunk",
                    ),
                ),
            ],
        ),
        migrations.CreateModel(
            name="Claim",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("content", models.TextField()),
                ("source_text", models.TextField()),
                ("embedding", pgvector.django.vector.VectorField(dimensions=1536)),
                ("embedding_model", models.CharField(default="text-embedding-3-small", max_length=64)),
                ("extraction_model", models.CharField(default="gpt-4o-mini", max_length=64)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                (
                    "chunk",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="claims",
                        to="transcripts.chunk",
                    ),
                ),
            ],
        ),
        migrations.CreateModel(
            name="AtomicPhrase",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("content", models.TextField()),
                ("source_text", models.TextField()),
                ("embedding", pgvector.django.vector.VectorField(dimensions=1536)),
                ("embedding_model", models.CharField(default="text-embedding-3-small", max_length=64)),
                ("extraction_model", models.CharField(default="gpt-4o-mini", max_length=64)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                (
                    "chunk",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="atomic_phrases",
                        to="transcripts.chunk",
                    ),
                ),
            ],
        ),
    ]
