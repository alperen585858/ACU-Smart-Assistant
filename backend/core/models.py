from django.db import models
from pgvector.django import HnswIndex, VectorField


class Page(models.Model):
    url = models.URLField(unique=True, max_length=2000)
    title = models.CharField(max_length=500, blank=True, default="")
    content = models.TextField()
    embedding_units = models.JSONField(
        null=True,
        blank=True,
        help_text="Optional list of DOM-derived text units (table row, list item, etc.) for entity-aware chunking.",
    )
    source = models.CharField(max_length=100, db_index=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.title or self.url


class DocumentChunk(models.Model):
    page = models.ForeignKey(Page, on_delete=models.CASCADE, related_name="chunks")
    chunk_index = models.PositiveIntegerField()
    content = models.TextField()
    embedding = VectorField(dimensions=384)
    source_url = models.URLField(max_length=2000)
    page_title = models.CharField(max_length=500, blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = ("page", "chunk_index")
        indexes = [
            models.Index(fields=["page", "chunk_index"]),
            HnswIndex(
                name="chunk_embedding_hnsw_idx",
                fields=["embedding"],
                m=16,
                ef_construction=64,
                opclasses=["vector_cosine_ops"],
            ),
        ]

    def __str__(self):
        return f"{self.page_id}:{self.chunk_index}"
