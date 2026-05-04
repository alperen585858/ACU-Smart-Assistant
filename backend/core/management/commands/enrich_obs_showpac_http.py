# pyright: reportMissingModuleSource=false
"""Re-fetch OBS showPac programme shells over HTTP and merge dynConPage detail text into Page rows."""
from __future__ import annotations

import logging
from typing import Any
from urllib.parse import urldefrag

from django.core.management.base import BaseCommand
from django.db.models import Q

from core.models import Page
from core.obs_bologna_scraper import SOURCE_LABEL, fetch_page_extract_http

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = (
        "For Pages whose URL is an OBS showPac programme shell, fetch HTML via HTTP "
        "and append dynConPage detail bodies (no Selenium). Then run build_page_embeddings."
    )

    def add_arguments(self, parser):
        parser.add_argument(
            "--delay",
            type=float,
            default=0.12,
            help="Seconds to sleep between dynCon HTTP requests (default: 0.12).",
        )
        parser.add_argument(
            "--limit",
            type=int,
            default=None,
            help="Optional max number of showPac pages to process.",
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Print URL and new content length only; do not write to the database.",
        )
        parser.add_argument(
            "--lang",
            type=str,
            default="en",
            help="Preferred language for dynCon URLs (default: en).",
        )
        parser.add_argument(
            "--url",
            type=str,
            default="",
            help="Process only this Page.url (OBS showPac). URL fragment (#...) is stripped. Ignores --limit.",
        )

    def handle(self, *args, **options):
        style: Any = self.style
        delay = max(0.0, float(options["delay"]))
        limit = options["limit"]
        dry_run: bool = options["dry_run"]
        lang = (options.get("lang") or "en").strip()
        one_url = (options.get("url") or "").strip()
        if one_url:
            one_url, _frag = urldefrag(one_url)
            one_url = one_url.strip()

        if one_url:
            if "showpac" not in one_url.lower():
                self.stderr.write(
                    style.ERROR("--url must be an OBS showPac programme URL.")
                )
                return
            qs = Page.objects.filter(url=one_url)
            if not qs.exists():
                self.stderr.write(
                    style.ERROR(
                        f"No Page row with exact url={one_url!r}. "
                        "Add it via scrape first, or fix the URL string."
                    )
                )
                return
        else:
            qs = Page.objects.filter(
                Q(url__icontains="obs.acibadem.edu.tr")
                & Q(url__icontains="showPac")
            ).order_by("id")
            if limit is not None:
                qs = qs[: max(1, int(limit))]

        n = 0
        for p in qs.iterator(chunk_size=50):
            url = str(p.url or "").strip()
            if not url:
                continue
            try:
                title, text, embedding_units = fetch_page_extract_http(
                    url, delay=delay, target_lang=lang
                )
            except Exception as exc:
                logger.warning("enrich failed %s: %s", url, exc)
                self.stdout.write(style.WARNING(f"Failed: {url} ({exc})"))
                continue
            if not (text or "").strip():
                self.stdout.write(style.WARNING(f"Empty after enrich: {url}"))
                continue
            if dry_run:
                self.stdout.write(
                    f"[dry-run] page_id={p.pk} {url} | title={title[:60]!r} | len={len(text)}"
                )
                n += 1
                continue
            Page.objects.filter(pk=p.pk).update(
                title=title,
                content=text,
                embedding_units=embedding_units or None,
                source=SOURCE_LABEL,
            )
            n += 1
            self.stdout.write(
                style.SUCCESS(
                    f"Updated page_id={p.pk} ({len(text)} chars): {url}\n"
                    f"Next: python manage.py build_page_embeddings --page-id {p.pk}"
                )
            )

        self.stdout.write(style.NOTICE(f"Done. Pages updated={n}, dry_run={dry_run}."))
