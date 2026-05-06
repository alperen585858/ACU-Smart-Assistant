"""Fill missing OBS programme prog*.aspx pages using existing showPac rows (HTTP only)."""
from __future__ import annotations

from django.core.management.base import BaseCommand

from core.obs_bologna_scraper import run_backfill_obs_prog_tabs_http


class Command(BaseCommand):
    help = (
        "Without re-running Selenium discovery: scan DB for OBS Pages whose URL is a "
        "showPac shell, synthesize standard prog*.aspx URLs (curSunit), fetch each via HTTP, "
        "and upsert Page rows. Skips URLs that already have non-empty content unless --force. "
        "This same HTTP pass also runs automatically at the end of scrape_obs_bologna unless "
        "you pass --skip-prog-backfill there."
    )

    def add_arguments(self, parser):
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="List how many URLs would be tried; no HTTP.",
        )
        parser.add_argument(
            "--limit",
            type=int,
            default=0,
            help="Maximum showPac Page rows to read from DB (0 = all).",
        )
        parser.add_argument(
            "--force",
            action="store_true",
            help="Re-fetch prog pages even when a row exists with body text.",
        )

    def handle(self, *args, **options):
        dry: bool = options["dry_run"]
        force: bool = options["force"]
        limit = max(0, int(options.get("limit") or 0))
        sty = self.style

        _, _, _, _, _ = run_backfill_obs_prog_tabs_http(
            limit=limit,
            force=force,
            dry_run=dry,
            delay=0.08,
            target_lang="en",
            on_notice=lambda m: self.stdout.write(sty.NOTICE(m)),
            on_saved=lambda m: self.stdout.write(sty.SUCCESS(m)),
            on_warn=lambda m: self.stderr.write(sty.WARNING(m)),
        )
