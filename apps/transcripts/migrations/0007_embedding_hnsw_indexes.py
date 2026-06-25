# Index DDL is applied via: python manage.py build_embedding_indexes
# (Railway Postgres often cannot build all HNSW indexes inside migrate.)

import pgvector.django.indexes
from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ("transcripts", "0006_episode_content_hash"),
    ]

    operations = [
        migrations.SeparateDatabaseAndState(
            state_operations=[
                migrations.AddIndex(
                    model_name="chunk",
                    index=pgvector.django.indexes.HnswIndex(
                        fields=["embedding"],
                        name="chunk_embedding_hnsw_idx",
                        opclasses=["vector_cosine_ops"],
                        m=8,
                        ef_construction=32,
                    ),
                ),
                migrations.AddIndex(
                    model_name="proposition",
                    index=pgvector.django.indexes.HnswIndex(
                        fields=["embedding"],
                        name="proposition_embedding_hnsw_idx",
                        opclasses=["vector_cosine_ops"],
                        m=8,
                        ef_construction=32,
                    ),
                ),
                migrations.AddIndex(
                    model_name="claim",
                    index=pgvector.django.indexes.HnswIndex(
                        fields=["embedding"],
                        name="claim_embedding_hnsw_idx",
                        opclasses=["vector_cosine_ops"],
                        m=8,
                        ef_construction=32,
                    ),
                ),
                migrations.AddIndex(
                    model_name="atomicphrase",
                    index=pgvector.django.indexes.HnswIndex(
                        fields=["embedding"],
                        name="phrase_embedding_hnsw_idx",
                        opclasses=["vector_cosine_ops"],
                        m=8,
                        ef_construction=32,
                    ),
                ),
            ],
            database_operations=[],
        ),
    ]
