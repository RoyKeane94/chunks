from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("transcripts", "0005_embedded_at"),
    ]

    operations = [
        migrations.AddField(
            model_name="episode",
            name="content_hash",
            field=models.CharField(blank=True, max_length=64, null=True, unique=True),
        ),
    ]
