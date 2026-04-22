"""
Diagnose "who is X" against the DB used by *this* process (Docker vs local may differ).

  python manage.py rag_whois_probe "who is Mahsa Ziraksima"
  docker compose exec backend python manage.py rag_whois_probe "who is Mahsa Ziraksima"
"""

from django.core.management.base import BaseCommand

from core.models import DocumentChunk
from core.rag_query_expand import whois_name_from_queries, whois_name_in_content
from core.rag_retrieval import _load_whois_name_chunks


class Command(BaseCommand):
    help = "Print who-is extraction and DocumentChunk hit counts (full string vs AND of name parts)."

    def add_arguments(self, parser):
        parser.add_argument("text", type=str, help="User line, e.g. who is Mahsa Ziraksima")

    def handle(self, *args, **options):
        raw = (options["text"] or "").strip()
        w = whois_name_from_queries("", raw)
        self.stdout.write(f"whois extract (primary=raw only): {w!r}")

        if not w:
            self.stdout.write(self.style.WARNING("Extraction failed — RAG who-is path will not run."))
            return

        n_full = DocumentChunk.objects.filter(content__icontains=w).count()
        self.stdout.write(
            f"count SQL icontains full string {w!r}: {n_full} (may not match TR spelling)"
        )
        load = _load_whois_name_chunks(w, 10)
        self.stdout.write(f"whois_name_in_content (TR/Latin fold, same as RAG): {len(load)}")
        for ch in load[:5]:
            ok = whois_name_in_content(str(ch.content or ""), w)
            self.stdout.write(
                f"  pk={ch.pk} whois_in_content={ok} url={ch.source_url[:90]}"
            )
            head = (ch.content or "")[:140].replace("\n", " ")
            self.stdout.write(f"     head: {head!r}…")
