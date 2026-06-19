from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("transcripts", "0003_multi_granularity_extraction"),
    ]

    operations = [
        migrations.AddField(
            model_name="proposition",
            name="start_char",
            field=models.IntegerField(default=0),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name="proposition",
            name="end_char",
            field=models.IntegerField(default=0),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name="claim",
            name="start_char",
            field=models.IntegerField(default=0),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name="claim",
            name="end_char",
            field=models.IntegerField(default=0),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name="atomicphrase",
            name="start_char",
            field=models.IntegerField(default=0),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name="atomicphrase",
            name="end_char",
            field=models.IntegerField(default=0),
            preserve_default=False,
        ),
    ]
