"""İlk açılış: indeks boşken refresh_rag (sınırsız HTTP crawl + embed)."""
from django.core.management import call_command
from django.core.management.base import BaseCommand

from core.models import DocumentChunk


class Command(BaseCommand):
    help = (
        "DocumentChunk yoksa refresh_rag calistirir: --max-pages 0, --depth -1, "
        "headless (OBS/JS yok) — Docker imajinda Chrome yok. Veri zaten pg volume'de."
    )

    def handle(self, *args, **options):
        if DocumentChunk.objects.exists():
            self.stdout.write(
                self.style.NOTICE("init_rag_if_empty: RAG zaten veri iceriyor, atlanıyor.")
            )
            return
        self.stdout.write(
            self.style.NOTICE(
                "init_rag_if_empty: Bos indeks; refresh_rag (sınırsız sayfa, depth sınırsız, "
                "HTTP-only; OBS/JS atlandı)..."
            )
        )
        call_command(
            "refresh_rag",
            max_pages=0,
            depth=-1,
            with_obs=False,
            with_acibadem_js=False,
        )
        self.stdout.write(self.style.SUCCESS("init_rag_if_empty: tamamlandı."))
