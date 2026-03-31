from django.core.management.base import BaseCommand

from core.models import DocumentChunk, Page


class Command(BaseCommand):
    help = "Print simple RAG corpus stats (pages/chunks/coverage)."

    def handle(self, *args, **options):
        page_count = Page.objects.count()
        chunk_count = DocumentChunk.objects.count()
        covered_pages = (
            DocumentChunk.objects.values_list("page_id", flat=True).distinct().count()
        )
        avg_chunks = (chunk_count / covered_pages) if covered_pages else 0.0

        self.stdout.write(self.style.NOTICE(f"Pages total: {page_count}"))
        self.stdout.write(self.style.NOTICE(f"Pages embedded: {covered_pages}"))
        self.stdout.write(self.style.NOTICE(f"Chunks total: {chunk_count}"))
        self.stdout.write(self.style.NOTICE(f"Avg chunks/page: {avg_chunks:.2f}"))
