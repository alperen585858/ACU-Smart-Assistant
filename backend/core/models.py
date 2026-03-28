from django.db import models


class Page(models.Model):
    url = models.URLField(unique=True, max_length=2000)
    title = models.CharField(max_length=500, blank=True, default="")
    content = models.TextField()
    source = models.CharField(max_length=100, db_index=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.title or self.url
