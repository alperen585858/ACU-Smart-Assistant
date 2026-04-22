# pyright: reportMissingModuleSource=false
"""Re-fetch Drupal AJAX-filled ACU pages (e.g. academic staff) with headless Chrome."""
import time
from typing import Any

from django.core.management.base import BaseCommand

from core.acibadem_js_scraper import ACIBADEM_JS_URLS, fetch_rendered_page, open_driver
from core.management.commands.scrape_acibadem import SOURCE_LABEL, normalize_url
from core.models import Page


class Command(BaseCommand):
    help = (
        "Fetch selected acibadem.edu.tr pages with Selenium so JS-injected content "
        "(e.g. department academic staff lists) is stored in core.Page."
    )

    def add_arguments(self, parser):
        parser.add_argument(
            "--delay",
            type=float,
            default=1.5,
            help="Seconds to sleep between page loads (default: 1.5).",
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Do not write to the database.",
        )

    def handle(self, *args, **options):
        style: Any = self.style
        page_objects: Any = getattr(Page, "objects")
        delay: float = max(0.0, float(options["delay"]))
        dry_run: bool = options["dry_run"]

        seeds = [normalize_url(u) for u in ACIBADEM_JS_URLS]
        seeds = [u for u in seeds if u]
        if not seeds:
            self.stderr.write(style.ERROR("No valid URLs in ACIBADEM_JS_URLS."))
            return

        self.stdout.write("Starting headless Chrome for ACU JS pages…")
        driver = None
        ok = 0
        fail = 0
        try:
            driver = open_driver()
            for url in seeds:
                try:
                    time.sleep(delay)
                    triple = fetch_rendered_page(driver, url)
                    if not triple:
                        self.stdout.write(
                            style.WARNING(f"Skip (no rendered staff content): {url}")
                        )
                        fail += 1
                        continue
                    title, text, embedding_units = triple
                    if not title:
                        title = url[:500]
                    if dry_run:
                        self.stdout.write(
                            f"[dry-run] {url} | {title[:80]!r} | {len(text)} chars"
                        )
                    else:
                        page_objects.update_or_create(
                            url=url,
                            defaults={
                                "title": title,
                                "content": text,
                                "embedding_units": embedding_units or None,
                                "source": SOURCE_LABEL,
                            },
                        )
                        self.stdout.write(style.SUCCESS(f"Saved (JS): {url}"))
                    ok += 1
                except Exception as exc:
                    self.stdout.write(
                        style.WARNING(f"Failed {url}: {exc}")
                    )
                    fail += 1
        finally:
            if driver is not None:
                driver.quit()

        self.stdout.write(
            style.NOTICE(
                f"scrape_acibadem_js: ok={ok}, skipped_or_err={fail}, dry_run={dry_run}."
            )
        )
