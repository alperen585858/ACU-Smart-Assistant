from django.contrib import admin

from .models import Page


@admin.register(Page)
class PageAdmin(admin.ModelAdmin):
    list_display = ("title", "url", "source", "updated_at", "created_at")
    list_filter = ("source",)
    search_fields = ("title", "url", "content")
    readonly_fields = ("created_at", "updated_at")
