# pyright: reportMissingModuleSource=false
"""Scrape OBS Bologna (JavaScript-driven) into core.Page via Selenium + headless Chrome."""
from __future__ import annotations

import logging
import queue
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from django.core.management.base import BaseCommand
from django.db import transaction

from core.models import Page
from core.obs_bologna_scraper import (
    SOURCE_LABEL,
    build_driver,
    collect_bologna_urls,
    fetch_page_extract,
)

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = (
        "Scrape obs.acibadem.edu.tr Bologna pages with Selenium (headless Chrome) "
        "and store them in core.Page."
    )

    def add_arguments(self, parser):
        parser.add_argument(
            "--delay",
            type=float,
            default=0.5,
            help="Seconds to sleep after each navigation or JS-triggering click (default: 0.5).",
        )
        parser.add_argument(
            "--max-programs",
            type=int,
            default=None,
            help="Optional cap on total pages to fetch and save (including index).",
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Do not write to the database; print URL, title preview, and content length.",
        )
        parser.add_argument(
            "--skip-section",
            type=str,
            default="",
            help="Comma-separated substrings; skip expand clicks whose link text or href matches (case-insensitive).",
        )
        parser.add_argument(
            "--lang",
            type=str,
            default="en",
            help="Prefer this site language: drop index/section URLs with lang=tr when set to en (default: en).",
        )
        parser.add_argument(
            "--fetch-workers",
            type=int,
            default=3,
            help=(
                "Parallel Chrome instances for URL fetch after discovery (default: 3). "
                "Use 1 for sequential (low RAM). Each worker needs one headless browser."
            ),
        )
        parser.add_argument(
            "--verbose",
            action="store_true",
            help=(
                "Print discovery progress, Done summary, failures, and scraper WARNING logs. "
                "Default is quiet: only Saved lines (and [dry-run] lines when using --dry-run). "
                "(Short -v is reserved by Django for --verbosity.)"
            ),
        )
        parser.add_argument(
            "--clear-existing",
            action="store_true",
            help=(
                "Delete existing OBS Bologna Page rows (source=obs.acibadem.edu.tr) and their "
                "chunks before scraping. ACU and other sources are unchanged."
            ),
        )

    def handle(self, *args, **options):
        style: Any = self.style
        page_objects: Any = getattr(Page, "objects")

        delay: float = max(0.0, float(options["delay"]))
        dry_run: bool = options["dry_run"]
        clear_existing: bool = bool(options.get("clear_existing"))
        max_programs: int | None = options["max_programs"]
        skip_raw: str = options["skip_section"] or ""
        skip_parts = [s.strip() for s in skip_raw.split(",") if s.strip()]
        fetch_workers: int = max(1, min(8, int(options["fetch_workers"] or 1)))
        target_lang: str = options.get("lang") or "en"
        verbose: bool = bool(options.get("verbose"))

        log_levels_restored: list[tuple[logging.Logger, int]] = []
        if not verbose:
            for name in (
                "core.obs_bologna_scraper",
                "urllib3",
                "urllib3.connectionpool",
            ):
                lg = logging.getLogger(name)
                log_levels_restored.append((lg, lg.level))
                lg.setLevel(logging.CRITICAL)

        if clear_existing:
            if dry_run:
                self.stdout.write(
                    style.WARNING(
                        "[dry-run] Skipping --clear-existing (no database changes)."
                    )
                )
            else:
                with transaction.atomic():
                    qs = Page.objects.filter(source=SOURCE_LABEL)
                    n = qs.count()
                    deleted, detail = qs.delete()
                self.stdout.write(
                    style.WARNING(
                        f"Cleared OBS pages: {n} row(s) removed "
                        f"(DocumentChunk cascade: {detail})."
                    )
                )

        drivers_to_quit: list[Any] = []
        saved = 0
        failed = 0
        urls: list[str] = []

        def _quit_all_drivers() -> None:
            for d in drivers_to_quit:
                try:
                    d.quit()
                except Exception as e:
                    if verbose:
                        logger.warning("driver.quit() failed: %s", e)
                    else:
                        logger.debug("driver.quit() failed: %s", e)

        try:
            if verbose:
                self.stdout.write("Starting headless Chrome…")
            collect_driver = build_driver()
            drivers_to_quit.append(collect_driver)
            urls = collect_bologna_urls(
                collect_driver,
                delay,
                skip_section_parts=skip_parts,
                target_lang=target_lang,
            )
            if verbose:
                self.stdout.write(
                    style.NOTICE(
                        f"Collected {len(urls)} unique URLs. "
                        f"Fetching with {fetch_workers} worker(s) — "
                        "showPac pages are slowest (sidebar + dynCon HTTP)."
                    )
                )

            if max_programs is not None:
                cap = max(1, int(max_programs))
                urls = urls[:cap]
                if verbose:
                    self.stdout.write(style.WARNING(f"Capped to {cap} URLs (--max-programs)."))

            def _save_one(url: str, title: str, text: str, embedding_units: list[str]) -> None:
                nonlocal saved
                if not text:
                    if verbose:
                        self.stdout.write(style.WARNING(f"Empty body text: {url}"))
                    else:
                        logger.debug("Empty body text: %s", url)
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
                    saved += 1
                    self.stdout.write(style.SUCCESS(f"Saved: {url}"))

            if fetch_workers == 1:
                for url in urls:
                    try:
                        title, text, embedding_units = fetch_page_extract(
                            collect_driver,
                            url,
                            delay,
                            target_lang=target_lang,
                        )
                    except Exception as exc:
                        failed += 1
                        if verbose:
                            logger.warning("Failed to fetch %s: %s", url, exc)
                            self.stdout.write(style.WARNING(f"Failed: {url} ({exc})"))
                        else:
                            logger.debug("Failed to fetch %s: %s", url, exc)
                        continue
                    _save_one(url, title, text, embedding_units)
            else:
                pool: queue.Queue[Any] = queue.Queue()
                pool.put(collect_driver)
                extra = fetch_workers - 1
                if verbose:
                    self.stdout.write(
                        style.NOTICE(
                            f"Parallel fetch: {fetch_workers} Chrome instance(s) "
                            f"({extra} extra after discovery driver)."
                        )
                    )
                for _ in range(extra):
                    d = build_driver()
                    drivers_to_quit.append(d)
                    pool.put(d)

                def _fetch_with_pooled_driver(
                    page_url: str,
                ) -> tuple[str, str, str, list[str]]:
                    """Return (url, title, text, embedding_units) — no DB in this thread."""
                    d = pool.get()
                    try:
                        title, text, units = fetch_page_extract(
                            d,
                            page_url,
                            delay,
                            target_lang=target_lang,
                        )
                        return page_url, title, text, units
                    finally:
                        pool.put(d)

                fut_to_url: dict[Any, str] = {}
                with ThreadPoolExecutor(max_workers=fetch_workers) as ex:
                    for u in urls:
                        fut = ex.submit(_fetch_with_pooled_driver, u)
                        fut_to_url[fut] = u
                    for fut in as_completed(fut_to_url):
                        url = fut_to_url[fut]
                        try:
                            _, title, text, embedding_units = fut.result()
                        except Exception as exc:
                            failed += 1
                            if verbose:
                                logger.warning("Failed to fetch %s: %s", url, exc)
                                self.stdout.write(style.WARNING(f"Failed: {url} ({exc})"))
                            else:
                                logger.debug("Failed to fetch %s: %s", url, exc)
                            continue
                        _save_one(url, title, text, embedding_units)

        finally:
            for lg, prev in log_levels_restored:
                lg.setLevel(prev)
            _quit_all_drivers()

        if verbose:
            self.stdout.write(
                style.NOTICE(
                    f"Done. URLs processed={len(urls)}, "
                    f"saved={saved if not dry_run else 0}, failed={failed}, dry_run={dry_run}."
                )
            )
