from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("transcripts", "0004_source_char_spans"),
    ]

    operations = [
        migrations.AddField(
            model_name="chunk",
            name="embedded_at",
            field=models.DateTimeField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name="proposition",
            name="embedded_at",
            field=models.DateTimeField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name="claim",
            name="embedded_at",
            field=models.DateTimeField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name="atomicphrase",
            name="embedded_at",
            field=models.DateTimeField(blank=True, null=True),
        ),
    ]
