from django.core.management import call_command
from django.core.management.base import BaseCommand
from django.db import transaction

from core.models import DocumentChunk, Page


class Command(BaseCommand):
    help = "Run end-to-end RAG refresh: scrape pages (ACU + OBS) then rebuild embeddings."

    def add_arguments(self, parser):
        parser.add_argument("--max-pages", type=int, default=60)
        parser.add_argument("--depth", type=int, default=2)
        parser.add_argument("--delay", type=float, default=1.5)
        parser.add_argument("--ignore-robots", action="store_true")
        parser.add_argument(
            "--with-obs",
            action="store_true",
            default=True,
            help="Run OBS Bologna scrape step (enabled by default).",
        )
        parser.add_argument(
            "--without-obs",
            action="store_false",
            dest="with_obs",
            help="Skip OBS Bologna scrape step.",
        )
        parser.add_argument(
            "--obs-delay",
            type=float,
            default=1.5,
            help="Delay for OBS Bologna Selenium scraping.",
        )
        parser.add_argument(
            "--obs-max-programs",
            type=int,
            default=None,
            help="Optional cap for scraped OBS Bologna pages.",
        )
        parser.add_argument(
            "--obs-lang",
            type=str,
            default="en",
            help="Target OBS language (default: en).",
        )
        parser.add_argument("--batch-size", type=int, default=16)
        parser.add_argument("--chunk-size", type=int, default=700)
        parser.add_argument("--chunk-overlap", type=int, default=120)
        parser.add_argument(
            "--keep-existing",
            action="store_true",
            help="Do not clear existing Page/DocumentChunk rows before refresh.",
        )

    def handle(self, *args, **options):
        if not options["keep_existing"]:
            self.stdout.write(
                self.style.NOTICE("Step 0/3: Clearing existing RAG rows...")
            )
            with transaction.atomic():
                deleted_chunks, _ = DocumentChunk.objects.all().delete()
                deleted_pages, _ = Page.objects.all().delete()
            self.stdout.write(
                self.style.WARNING(
                    f"Cleared rows: DocumentChunk={deleted_chunks}, Page={deleted_pages}"
                )
            )

        self.stdout.write(self.style.NOTICE("Step 1/3: Scraping ACU pages..."))
        call_command(
            "scrape_acibadem",
            crawl=True,
            max_pages=options["max_pages"],
            depth=options["depth"],
            delay=options["delay"],
            ignore_robots=options["ignore_robots"],
        )

        if options["with_obs"]:
            self.stdout.write(self.style.NOTICE("Step 2/3: Scraping OBS Bologna pages..."))
            obs_args = {
                "delay": options["obs_delay"],
                "lang": options["obs_lang"],
            }
            if options["obs_max_programs"] is not None:
                obs_args["max_programs"] = options["obs_max_programs"]
            call_command("scrape_obs_bologna", **obs_args)
        else:
            self.stdout.write(self.style.WARNING("Step 2/3: Skipping OBS Bologna scrape (--without-obs)."))

        self.stdout.write(self.style.NOTICE("Step 3/3: Building embeddings..."))
        call_command(
            "build_page_embeddings",
            batch_size=options["batch_size"],
            chunk_size=options["chunk_size"],
            chunk_overlap=options["chunk_overlap"],
        )

        self.stdout.write(self.style.SUCCESS("RAG refresh completed."))
