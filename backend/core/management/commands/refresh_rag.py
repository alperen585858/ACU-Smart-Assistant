from django.core.management import call_command
from django.core.management.base import BaseCommand


class Command(BaseCommand):
    help = "Run end-to-end RAG refresh: scrape pages then rebuild embeddings."

    def add_arguments(self, parser):
        parser.add_argument("--max-pages", type=int, default=60)
        parser.add_argument("--depth", type=int, default=2)
        parser.add_argument("--delay", type=float, default=1.5)
        parser.add_argument("--ignore-robots", action="store_true")
        parser.add_argument("--batch-size", type=int, default=16)
        parser.add_argument("--chunk-size", type=int, default=700)
        parser.add_argument("--chunk-overlap", type=int, default=120)

    def handle(self, *args, **options):
        self.stdout.write(self.style.NOTICE("Step 1/2: Scraping ACU pages..."))
        call_command(
            "scrape_acibadem",
            crawl=True,
            max_pages=options["max_pages"],
            depth=options["depth"],
            delay=options["delay"],
            ignore_robots=options["ignore_robots"],
        )

        self.stdout.write(self.style.NOTICE("Step 2/2: Building embeddings..."))
        call_command(
            "build_page_embeddings",
            batch_size=options["batch_size"],
            chunk_size=options["chunk_size"],
            chunk_overlap=options["chunk_overlap"],
        )

        self.stdout.write(self.style.SUCCESS("RAG refresh completed."))
