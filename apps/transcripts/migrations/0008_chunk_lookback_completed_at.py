from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("transcripts", "0007_embedding_hnsw_indexes"),
    ]

    operations = [
        migrations.AddField(
            model_name="chunk",
            name="lookback_completed_at",
            field=models.DateTimeField(blank=True, null=True),
        ),
    ]
