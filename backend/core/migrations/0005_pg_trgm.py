from django.db import migrations


def enable_pg_trgm(apps, schema_editor):
    if schema_editor.connection.vendor != "postgresql":
        return
    schema_editor.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")


def drop_pg_trgm(apps, schema_editor):
    if schema_editor.connection.vendor != "postgresql":
        return
    schema_editor.execute("DROP EXTENSION IF EXISTS pg_trgm")


def add_trgm_index(apps, schema_editor):
    if schema_editor.connection.vendor != "postgresql":
        return
    schema_editor.execute(
        "CREATE INDEX IF NOT EXISTS core_docchunk_content_trgm_idx "
        "ON core_documentchunk USING gin (content gin_trgm_ops)"
    )


def drop_trgm_index(apps, schema_editor):
    if schema_editor.connection.vendor != "postgresql":
        return
    schema_editor.execute("DROP INDEX IF EXISTS core_docchunk_content_trgm_idx")


class Migration(migrations.Migration):
    dependencies = [
        ("core", "0004_documentchunk_embedding_vector_index"),
    ]

    operations = [
        migrations.RunPython(enable_pg_trgm, drop_pg_trgm),
        migrations.RunPython(add_trgm_index, drop_trgm_index),
    ]
