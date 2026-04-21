from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("core", "0005_pg_trgm"),
    ]

    operations = [
        migrations.AddField(
            model_name="page",
            name="embedding_units",
            field=models.JSONField(blank=True, null=True),
        ),
    ]
