import re
import time
<<<<<<< HEAD
from typing import Any
=======
>>>>>>> 8b8976e (Add Acibadem web scraper and Page storage)
from collections import deque
from urllib.parse import urldefrag, urljoin, urlparse
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup
from django.core.management.base import BaseCommand

from core.models import Page

SOURCE_LABEL = "acibadem.edu.tr"
DEFAULT_SEEDS = (
    "https://www.acibadem.edu.tr/",
    "https://www.acibadem.edu.tr/en",
)
ALLOWED_NETLOCS = frozenset(
    {
<<<<<<< HEAD
=======
        "www.acibadem.edu.tr",
>>>>>>> 8b8976e (Add Acibadem web scraper and Page storage)
        "acibadem.edu.tr",
    }
)
USER_AGENT = (
    "ACU-Smart-Assistant/0.1 (+university project; respectful crawl; contact: student)"
)
REQUEST_TIMEOUT = 25
<<<<<<< HEAD
MAX_CONTENT_CHARS = 5000
=======
>>>>>>> 8b8976e (Add Acibadem web scraper and Page storage)


def normalize_url(url: str) -> str:
    url, _frag = urldefrag(url.strip())
    parsed = urlparse(url)
    if not parsed.scheme:
        url = "https://" + url.lstrip("/")
        parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return ""
    host = parsed.netloc.lower()
<<<<<<< HEAD
    # Normalize host so "www." and non-"www" URLs collapse to the same record.
    if host.startswith("www."):
        host = host[4:]
=======
>>>>>>> 8b8976e (Add Acibadem web scraper and Page storage)
    if host not in ALLOWED_NETLOCS:
        return ""
    path = parsed.path or "/"
    return f"{parsed.scheme}://{host}{path}" + (
        f"?{parsed.query}" if parsed.query else ""
    )


def same_site(url: str) -> bool:
    try:
<<<<<<< HEAD
        host = urlparse(url).netloc.lower()
        if host.startswith("www."):
            host = host[4:]
        return host in ALLOWED_NETLOCS
=======
        return urlparse(url).netloc.lower() in ALLOWED_NETLOCS
>>>>>>> 8b8976e (Add Acibadem web scraper and Page storage)
    except Exception:
        return False


def extract_title_and_text(html: str) -> tuple[str, str]:
    soup = BeautifulSoup(html, "lxml")
    raw_title = ""
    if soup.title and soup.title.string:
        raw_title = soup.title.string.strip()
    for tag in soup(["script", "style", "noscript", "svg"]):
        tag.decompose()
    root = soup.find("main") or soup.find("article") or soup.body
    if root is None:
        text = ""
    else:
        text = root.get_text(separator="\n", strip=True)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
<<<<<<< HEAD
    # Keep DB rows (and later LLM context) bounded.
    text = text[:MAX_CONTENT_CHARS]
=======
>>>>>>> 8b8976e (Add Acibadem web scraper and Page storage)
    title = (raw_title or "")[:500]
    return title, text


def load_robot_parser(base: str) -> RobotFileParser | None:
    robots_url = urljoin(base, "/robots.txt")
    rp = RobotFileParser()
    rp.set_url(robots_url)
    try:
        rp.read()
        return rp
    except Exception:
        return None


class Command(BaseCommand):
    help = (
        "Fetch public pages from www.acibadem.edu.tr via requests + BeautifulSoup "
        "and store them in core.Page (polite delays; same host only)."
    )

    def add_arguments(self, parser):
        parser.add_argument(
            "--crawl",
            action="store_true",
            help="Follow same-site links starting from seed URLs (bounded).",
        )
        parser.add_argument(
            "--max-pages",
            type=int,
            default=40,
            help="Maximum pages to fetch when --crawl is set (default: 40).",
        )
        parser.add_argument(
            "--depth",
            type=int,
            default=2,
            help="Maximum link hops from seeds when crawling (default: 2).",
        )
        parser.add_argument(
            "--delay",
            type=float,
            default=1.5,
            help="Seconds to sleep between HTTP requests (default: 1.5).",
        )
        parser.add_argument(
            "--ignore-robots",
            action="store_true",
            help="Do not check robots.txt (not recommended).",
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Fetch and parse but do not write to the database.",
        )

    def handle(self, *args, **options):
<<<<<<< HEAD
        # `BaseCommand.style` is provided dynamically by Django and basedpyright
        # sometimes fails to understand its full attribute surface.
        style: Any = self.style
=======
>>>>>>> 8b8976e (Add Acibadem web scraper and Page storage)
        delay: float = max(0.0, options["delay"])
        dry_run: bool = options["dry_run"]
        crawl: bool = options["crawl"]
        max_pages: int = max(1, options["max_pages"])
        max_depth: int = max(0, options["depth"])
        ignore_robots: bool = options["ignore_robots"]

        session = requests.Session()
        session.headers.update(
            {
                "User-Agent": USER_AGENT,
                "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "tr-TR,tr;q=0.9,en;q=0.8",
            }
        )

<<<<<<< HEAD
        rp = (
            None
            if ignore_robots
            else load_robot_parser("https://acibadem.edu.tr/")
        )
        if not ignore_robots and rp is None:
            self.stdout.write(
                style.WARNING(
=======
        rp = None if ignore_robots else load_robot_parser("https://www.acibadem.edu.tr/")
        if not ignore_robots and rp is None:
            self.stdout.write(
                self.style.WARNING(
>>>>>>> 8b8976e (Add Acibadem web scraper and Page storage)
                    "Could not load robots.txt; continuing without robots checks."
                )
            )

        seeds = [normalize_url(u) for u in DEFAULT_SEEDS]
        seeds = [u for u in seeds if u]
        if not seeds:
<<<<<<< HEAD
            self.stderr.write(style.ERROR("No valid seed URLs."))
=======
            self.stderr.write(self.style.ERROR("No valid seed URLs."))
>>>>>>> 8b8976e (Add Acibadem web scraper and Page storage)
            return

        if crawl:
            queue: deque[tuple[str, int]] = deque((u, 0) for u in seeds)
        else:
            queue = deque((u, 0) for u in seeds)

        visited: set[str] = set()
        saved = 0
        fetched = 0

        while queue and fetched < max_pages:
            url, depth = queue.popleft()
            if url in visited:
                continue
            visited.add(url)

            if rp is not None and not rp.can_fetch(USER_AGENT, url):
<<<<<<< HEAD
                self.stdout.write(style.WARNING(f"robots.txt disallows: {url}"))
=======
                self.stdout.write(self.style.WARNING(f"robots.txt disallows: {url}"))
>>>>>>> 8b8976e (Add Acibadem web scraper and Page storage)
                continue

            try:
                time.sleep(delay)
                resp = session.get(url, timeout=REQUEST_TIMEOUT)
                resp.raise_for_status()
            except requests.RequestException as exc:
<<<<<<< HEAD
                self.stdout.write(style.WARNING(f"GET failed {url}: {exc}"))
=======
                self.stdout.write(self.style.WARNING(f"GET failed {url}: {exc}"))
>>>>>>> 8b8976e (Add Acibadem web scraper and Page storage)
                continue

            ctype = (resp.headers.get("Content-Type") or "").lower()
            if "text/html" not in ctype and "application/xhtml" not in ctype:
<<<<<<< HEAD
                self.stdout.write(style.WARNING(f"Skip non-HTML: {url}"))
=======
                self.stdout.write(self.style.WARNING(f"Skip non-HTML: {url}"))
>>>>>>> 8b8976e (Add Acibadem web scraper and Page storage)
                continue

            fetched += 1
            html = resp.text
            title, text = extract_title_and_text(html)
            if not title:
                title = url[:500]
            if not text:
<<<<<<< HEAD
                self.stdout.write(style.WARNING(f"Empty body text: {url}"))
=======
                self.stdout.write(self.style.WARNING(f"Empty body text: {url}"))
>>>>>>> 8b8976e (Add Acibadem web scraper and Page storage)
                text = ""

            if dry_run:
                self.stdout.write(f"[dry-run] {url} | {title[:80]!r} | {len(text)} chars")
            else:
<<<<<<< HEAD
                Page.objects.update_or_create(  # type: ignore[attr-defined]
=======
                Page.objects.update_or_create(
>>>>>>> 8b8976e (Add Acibadem web scraper and Page storage)
                    url=url,
                    defaults={
                        "title": title,
                        "content": text,
                        "source": SOURCE_LABEL,
                    },
                )
                saved += 1
<<<<<<< HEAD
                self.stdout.write(style.SUCCESS(f"Saved: {url}"))
=======
                self.stdout.write(self.style.SUCCESS(f"Saved: {url}"))
>>>>>>> 8b8976e (Add Acibadem web scraper and Page storage)

            if not crawl or depth >= max_depth:
                continue

            soup = BeautifulSoup(html, "lxml")
            for a in soup.find_all("a", href=True):
                href = a["href"].strip()
                if href.startswith(("mailto:", "tel:", "javascript:")):
                    continue
                next_url = normalize_url(urljoin(url, href))
                if not next_url or next_url in visited:
                    continue
                if not same_site(next_url):
                    continue
                queue.append((next_url, depth + 1))

        self.stdout.write(
<<<<<<< HEAD
            style.NOTICE(
=======
            self.style.NOTICE(
>>>>>>> 8b8976e (Add Acibadem web scraper and Page storage)
                f"Done. Fetched={fetched}, DB rows touched={saved if not dry_run else 0}, "
                f"crawl={'on' if crawl else 'off'}."
            )
        )
