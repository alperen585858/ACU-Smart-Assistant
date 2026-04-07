# pyright: reportMissingModuleSource=false
"""Fetch pages from acibadem.edu.tr into core.Page."""
import re
import time
from collections import deque
from urllib.parse import urldefrag, urljoin, urlparse
from urllib.robotparser import RobotFileParser
from typing import Any

import requests
from bs4 import BeautifulSoup
from django.core.management.base import BaseCommand

from core.models import Page

# Selenium is optional at import time; fallback is skipped if not installed.
try:
    from selenium import webdriver
    from selenium.common.exceptions import TimeoutException
    from selenium.webdriver.chrome.options import Options as ChromeOptions
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.support.ui import WebDriverWait

    _SELENIUM_IMPORTS_OK = True
except ImportError:
    _SELENIUM_IMPORTS_OK = False
    webdriver = None  # type: ignore[assignment,misc]
    TimeoutException = Exception  # type: ignore[misc,assignment]
    ChromeOptions = None  # type: ignore[misc,assignment]
    By = None  # type: ignore[misc,assignment]
    EC = None  # type: ignore[misc,assignment]
    WebDriverWait = None  # type: ignore[misc,assignment]

SOURCE_LABEL = "acibadem.edu.tr"
DEFAULT_SEEDS = (
    "https://www.acibadem.edu.tr/en",
    # Registration contact/transport — often not reached within default depth/page limits from home alone.
    "https://www.acibadem.edu.tr/en/kayit/iletisim/ulasim",
    "https://www.acibadem.edu.tr/en/academic/undergraduate-programs/faculty-of-engineering-and-natural-sciences/departments/computer-engineering/message-from-head-of-department",
    "https://www.acibadem.edu.tr/en/academic/undergraduate-programs/faculty-of-engineering-and-natural-sciences/departments/computer-engineering/about",
    "https://www.acibadem.edu.tr/en/academic/undergraduate-programs/faculty-of-engineering-and-natural-sciences/departments/computer-engineering/academic-staff",
    "https://www.acibadem.edu.tr/en/academic/undergraduate-programs/faculty-of-engineering-and-natural-sciences/departments/computer-engineering/commissions",
)
ALLOWED_NETLOCS = frozenset(
    {
        "acibadem.edu.tr",
    }
)
USER_AGENT = (
    "ACU-Smart-Assistant/0.1 (+university project; respectful crawl; contact: student)"
)
REQUEST_TIMEOUT = 25
MAX_CONTENT_CHARS = 5000
# If requests+BeautifulSoup yield less text than this, the page may be a JS shell (SPA);
# we then retry once with Selenium so client-rendered content (e.g. Bologna) can appear.
MIN_MEANINGFUL_TEXT_CHARS = 120
SELENIUM_PAGE_LOAD_TIMEOUT = 30
SELENIUM_MAIN_WAIT = 15

# One headless Chrome instance per crawler run (see fetch_with_selenium); torn down in handle().
_selenium_driver: Any = None


def is_english_path(path: str) -> bool:
    normalized_path = (path or "/").rstrip("/").lower()
    return normalized_path == "/en" or normalized_path.startswith("/en/")


def normalize_url(url: str, english_only: bool = True) -> str:
    url, _frag = urldefrag(url.strip())
    parsed = urlparse(url)
    if not parsed.scheme:
        url = "https://" + url.lstrip("/")
        parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return ""
    host = parsed.netloc.lower()
    # Normalize host so "www." and non-"www" URLs collapse to the same record.
    if host.startswith("www."):
        host = host[4:]
    if host not in ALLOWED_NETLOCS:
        return ""
    path = parsed.path or "/"
    if english_only and not is_english_path(path):
        return ""
    return f"{parsed.scheme}://{host}{path}" + (
        f"?{parsed.query}" if parsed.query else ""
    )


def same_site(url: str) -> bool:
    try:
        host = urlparse(url).netloc.lower()
        if host.startswith("www."):
            host = host[4:]
        return host in ALLOWED_NETLOCS
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
    # Keep DB rows (and later LLM context) bounded.
    text = text[:MAX_CONTENT_CHARS]
    title = (raw_title or "")[:500]
    return title, text


def _text_insufficient(title: str, text: str) -> bool:
    """True when static HTML likely missed client-rendered body content."""
    return len((text or "").strip()) < MIN_MEANINGFUL_TEXT_CHARS


def _get_or_create_selenium_driver() -> Any:
    """Lazily start headless Chrome once; reused for every Selenium fallback in this process."""
    global _selenium_driver
    if not _SELENIUM_IMPORTS_OK or webdriver is None or ChromeOptions is None:
        raise RuntimeError("selenium is not installed")
    if _selenium_driver is not None:
        return _selenium_driver
    opts = ChromeOptions()
    opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--disable-gpu")
    opts.add_argument(f"--user-agent={USER_AGENT}")
    driver = webdriver.Chrome(options=opts)
    driver.set_page_load_timeout(SELENIUM_PAGE_LOAD_TIMEOUT)
    _selenium_driver = driver
    return _selenium_driver


def _shutdown_selenium_driver() -> None:
    global _selenium_driver
    if _selenium_driver is not None:
        try:
            _selenium_driver.quit()
        except Exception:
            pass
        _selenium_driver = None


def fetch_with_selenium(url: str) -> str:
    """Load URL in headless Chrome, wait for DOM content, return rendered HTML.

    Used only when the fast path (requests + BeautifulSoup) returns too little text,
    which often means the server sent an empty shell and real markup arrived via JavaScript
    (course catalogs, Bologna-style pages). Not called for every URL—only on demand.
    """
    if WebDriverWait is None or By is None or EC is None:
        raise RuntimeError("selenium is not installed")
    driver = _get_or_create_selenium_driver()
    driver.get(url)
    # Wait for document shell, then prefer semantic main content (many academic sites use it).
    wait = WebDriverWait(driver, SELENIUM_PAGE_LOAD_TIMEOUT)
    wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))
    try:
        WebDriverWait(driver, SELENIUM_MAIN_WAIT).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "main, article"))
        )
    except TimeoutException:
        # Some pages only populate generic divs; body is enough to call page_source.
        pass
    return driver.page_source


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
        "(Selenium headless Chrome as fallback when static HTML has little text) "
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
        parser.add_argument(
            "--allow-non-english",
            action="store_true",
            help="Allow crawling non-English paths (disabled by default).",
        )

    def handle(self, *args, **options):
        # Django provides these dynamically at runtime, but type checkers often
        # don't have enough metadata for `self.style` and `Model.objects`.
        style: Any = self.style
        page_objects: Any = getattr(Page, "objects")

        delay: float = max(0.0, options["delay"])
        dry_run: bool = options["dry_run"]
        crawl: bool = options["crawl"]
        max_pages: int = max(1, options["max_pages"])
        max_depth: int = max(0, options["depth"])
        ignore_robots: bool = options["ignore_robots"]
        english_only: bool = not options["allow_non_english"]

        session = requests.Session()
        session.headers.update(
            {
                "User-Agent": USER_AGENT,
                "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9,tr;q=0.5",
            }
        )

        rp = (
            None
            if ignore_robots
            else load_robot_parser("https://acibadem.edu.tr/")
        )
        if not ignore_robots and rp is None:
            self.stdout.write(
                style.WARNING(
                    "Could not load robots.txt; continuing without robots checks."
                )
            )

        seeds = [normalize_url(u, english_only=english_only) for u in DEFAULT_SEEDS]
        seeds = [u for u in seeds if u]
        if not seeds:
            self.stderr.write(style.ERROR("No valid seed URLs."))
            return

        if crawl:
            queue: deque[tuple[str, int]] = deque((u, 0) for u in seeds)
        else:
            queue = deque((u, 0) for u in seeds)

        visited: set[str] = set()
        saved = 0
        fetched = 0
        selenium_skip_logged = False

        try:
            while queue and fetched < max_pages:
                url, depth = queue.popleft()
                if url in visited:
                    continue
                visited.add(url)

                if rp is not None and not rp.can_fetch(USER_AGENT, url):
                    self.stdout.write(style.WARNING(f"robots.txt disallows: {url}"))
                    continue

                try:
                    time.sleep(delay)
                    resp = session.get(url, timeout=REQUEST_TIMEOUT)
                    resp.raise_for_status()
                except requests.RequestException as exc:
                    self.stdout.write(style.WARNING(f"GET failed {url}: {exc}"))
                    continue

                ctype = (resp.headers.get("Content-Type") or "").lower()
                if "text/html" not in ctype and "application/xhtml" not in ctype:
                    self.stdout.write(style.WARNING(f"Skip non-HTML: {url}"))
                    continue

                fetched += 1
                html = resp.text
                title, text = extract_title_and_text(html)

                # Slow path: static HTML often lacks body text when content is injected via JS
                # (e.g. Bologna / course tools). Same delay as between normal GETs to stay polite.
                if _text_insufficient(title, text):
                    if _SELENIUM_IMPORTS_OK:
                        try:
                            time.sleep(delay)
                            html = fetch_with_selenium(url)
                            title, text = extract_title_and_text(html)
                            self.stdout.write(
                                style.NOTICE(f"Selenium fallback used (short static text): {url}")
                            )
                        except Exception as exc:
                            self.stdout.write(
                                style.WARNING(f"Selenium fallback failed {url}: {exc}")
                            )
                    elif not selenium_skip_logged:
                        self.stdout.write(
                            style.WARNING(
                                "Short text from static HTML and selenium is not installed; "
                                "pip install selenium (+ Chrome) for JS-heavy pages."
                            )
                        )
                        selenium_skip_logged = True

                if not title:
                    title = url[:500]
                if not text:
                    self.stdout.write(style.WARNING(f"Empty body text: {url}"))
                    text = ""

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
                            "source": SOURCE_LABEL,
                        },
                    )
                    saved += 1
                    self.stdout.write(style.SUCCESS(f"Saved: {url}"))

                if not crawl or depth >= max_depth:
                    continue

                soup = BeautifulSoup(html, "lxml")
                for a in soup.find_all("a", href=True):
                    href = a["href"].strip()
                    if href.startswith(("mailto:", "tel:", "javascript:")):
                        continue
                    next_url = normalize_url(
                        urljoin(url, href), english_only=english_only
                    )
                    if not next_url or next_url in visited:
                        continue
                    if not same_site(next_url):
                        continue
                    queue.append((next_url, depth + 1))
        finally:
            _shutdown_selenium_driver()

        self.stdout.write(
            style.NOTICE(
                f"Done. Fetched={fetched}, DB rows touched={saved if not dry_run else 0}, "
                f"crawl={'on' if crawl else 'off'}."
            )
        )
