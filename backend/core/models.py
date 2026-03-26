from django.db import models


class Page(models.Model):
    url = models.URLField(unique=True, max_length=2000)
    title = models.CharField(max_length=500)
    content = models.TextField()
    source = models.CharField(
        max_length=100,
        blank=True,
        default="",
        db_index=True,
        help_text="Origin label, e.g. acibadem.edu.tr",
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-updated_at"]

    def __str__(self):
        return self.title
