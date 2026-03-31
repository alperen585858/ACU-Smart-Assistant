import django.db.models.deletion
import pgvector.django
from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("core", "0001_initial"),
    ]

    operations = [
        migrations.RunSQL(
            sql="CREATE EXTENSION IF NOT EXISTS vector",
            reverse_sql="DROP EXTENSION IF EXISTS vector",
        ),
        migrations.CreateModel(
            name="DocumentChunk",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True, primary_key=True, serialize=False, verbose_name="ID"
                    ),
                ),
                ("chunk_index", models.PositiveIntegerField()),
                ("content", models.TextField()),
                ("embedding", pgvector.django.VectorField(dimensions=384)),
                ("source_url", models.URLField(max_length=2000)),
                ("page_title", models.CharField(blank=True, default="", max_length=500)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("updated_at", models.DateTimeField(auto_now=True)),
                (
                    "page",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="chunks",
                        to="core.page",
                    ),
                ),
            ],
            options={
                "unique_together": {("page", "chunk_index")},
            },
        ),
        migrations.AddIndex(
            model_name="documentchunk",
            index=models.Index(fields=["page", "chunk_index"], name="core_docume_page_id_d27b35_idx"),
        ),
    ]
