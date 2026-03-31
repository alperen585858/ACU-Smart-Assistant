from django.contrib import admin

from .models import DocumentChunk, Page


@admin.register(Page)
class PageAdmin(admin.ModelAdmin):
    list_display = ("title", "source", "updated_at")


@admin.register(DocumentChunk)
class DocumentChunkAdmin(admin.ModelAdmin):
    list_display = ("page_id", "chunk_index", "updated_at")
    search_fields = ("page_title", "source_url", "content")
