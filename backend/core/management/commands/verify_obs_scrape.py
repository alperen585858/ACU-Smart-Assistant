"""Print OBS (Bologna) scrape coverage stats for quick validation."""
from __future__ import annotations

from django.core.management.base import BaseCommand

from core.models import Page
from core.obs_bologna_scraper import SOURCE_LABEL


class Command(BaseCommand):
    help = (
        "Show how many OBS Bologna pages are stored and sample URL patterns "
        "(dynConPage, prog*.aspx programme pages, showPac, unitSelection)."
    )

    def handle(self, *args, **options):
        qs = Page.objects.filter(source=SOURCE_LABEL)
        n = qs.count()
        self.stdout.write(self.style.NOTICE(f"OBS source ({SOURCE_LABEL}) Page rows: {n}"))
        for label, needle in [
            ("dynConPage", "dynConPage.aspx"),
            ("progCourses", "progCourses.aspx"),
            ("progAbout", "progAbout.aspx"),
            ("progGoalsObjectives", "progGoalsObjectives.aspx"),
            ("showPac (case-insensitive)", "showpac"),
            ("unitSelection", "unitSelection.aspx"),
        ]:
            c = qs.filter(url__icontains=needle).count()
            self.stdout.write(f"  URL contains {label!r}: {c}")
        self.stdout.write(self.style.NOTICE("First 5 URLs (by id):"))
        for p in qs.order_by("id")[:5]:
            self.stdout.write(f"  {p.id} {p.url[:120]}")
