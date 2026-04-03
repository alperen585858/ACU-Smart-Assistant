from django.db import migrations
from pgvector.django import HnswIndex


class Migration(migrations.Migration):

    dependencies = [
        ("core", "0003_rename_core_docume_page_id_d27b35_idx_core_docume_page_id_2746ba_idx"),
    ]

    operations = [
        migrations.AddIndex(
            model_name="documentchunk",
            index=HnswIndex(
                name="chunk_embedding_hnsw_idx",
                fields=["embedding"],
                m=16,
                ef_construction=64,
                opclasses=["vector_cosine_ops"],
            ),
        ),
    ]
