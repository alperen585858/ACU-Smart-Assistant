"""Selenium helpers for scraping OBS Bologna (obs.acibadem.edu.tr) pages."""
from __future__ import annotations

import logging
import os
import re
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import parse_qs, urldefrag, urljoin, urlparse

import requests
from requests import exceptions as req_exc
from selenium.common.exceptions import (
    ElementClickInterceptedException,
    ElementNotInteractableException,
    NoSuchDriverException,
    StaleElementReferenceException,
    TimeoutException,
    WebDriverException,
)
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

from core.html_extract import extract_title_text_and_embedding_units

logger = logging.getLogger(__name__)

BOLOGNA_INDEX_URL = "https://obs.acibadem.edu.tr/oibs/bologna/index.aspx?lang=en"
SOURCE_LABEL = "obs.acibadem.edu.tr"
OBS_HOST = "obs.acibadem.edu.tr"
BOLOGNA_PATH_MARKER = "/oibs/bologna/"
BOLOGNA_SEED_URLS = (
    "https://obs.acibadem.edu.tr/oibs/bologna/index.aspx?lang=en&curOp=showPac&curUnit=16&curSunit=6288",
    "https://obs.acibadem.edu.tr/oibs/bologna/index.aspx?lang=en&curOp=showPac&curUnit=16&curSunit=6289",
    "https://obs.acibadem.edu.tr/oibs/bologna/index.aspx?lang=en&curOp=showPac&curUnit=14&curSunit=6247",
    "https://obs.acibadem.edu.tr/oibs/bologna/index.aspx?lang=en&curOp=showPac&curUnit=14&curSunit=6246",
    "https://obs.acibadem.edu.tr/oibs/bologna/index.aspx?lang=en&curOp=showPac&curUnit=14&curSunit=6248",
    "https://obs.acibadem.edu.tr/oibs/bologna/index.aspx?lang=en&curOp=showPac&curUnit=05&curSunit=5",
    "https://obs.acibadem.edu.tr/oibs/bologna/index.aspx?lang=en&curOp=showPac&curUnit=05&curSunit=3",
    "https://obs.acibadem.edu.tr/oibs/bologna/index.aspx?lang=en&curOp=showPac&curUnit=05&curSunit=6108",
    "https://obs.acibadem.edu.tr/oibs/bologna/index.aspx?lang=en&curOp=showPac&curUnit=05&curSunit=4",
    "https://obs.acibadem.edu.tr/oibs/bologna/index.aspx?lang=en&curOp=showPac&curUnit=05&curSunit=6107",
    "https://obs.acibadem.edu.tr/oibs/bologna/index.aspx?lang=en&curOp=showPac&curUnit=05&curSunit=2",
    "https://obs.acibadem.edu.tr/oibs/bologna/index.aspx?lang=en&curOp=showPac&curUnit=16&curSunit=6287",
    "https://obs.acibadem.edu.tr/oibs/bologna/index.aspx?lang=en&curOp=showPac&curUnit=12&curSunit=17",
    "https://obs.acibadem.edu.tr/oibs/bologna/index.aspx?lang=en&curOp=showPac&curUnit=06&curSunit=1",
)

USER_AGENT = (
    "ACU-Smart-Assistant/0.1 (+university project; respectful crawl; contact: student)"
)

MAX_EXPAND_PASSES = 8
MAX_POSTBACK_CLICKS_PER_PASS = 50
MAX_MENU_CLOSE_CLICKS = 36
# showPac pages: only click a small set of left-rail "Information Package" links (not full-site __doPostBack).
_MAX_SHOWPAC_SIDEBAR_CLICKS = 12
_SHOWPAC_NAV_READY_TIMEOUT = 4.0
_SHOWPAC_NAV_CLICK_SLEEP = 0.22
PAGE_READY_TIMEOUT = 10
# facAbout / heavy ASP.NET pages: shorter post-load wait (content is usually in DOM early).
PAGE_READY_TIMEOUT_FACABOUT = 6.0
SHORT_WAIT = 12
# Index/menu expand clicks: full SHORT_WAIT per click made discovery take hours.
EXPAND_CLICK_READY_TIMEOUT = 6.0
HTTP_TIMEOUT = 28


def _env_bounded_int(key: str, default: int, lo: int, hi: int) -> int:
    raw = (os.environ.get(key) or "").strip()
    if not raw:
        return default
    try:
        return max(lo, min(hi, int(raw)))
    except ValueError:
        return default


def _env_bounded_float(key: str, default: float, lo: float, hi: float) -> float:
    raw = (os.environ.get(key) or "").strip()
    if not raw:
        return default
    try:
        return max(lo, min(hi, float(raw)))
    except ValueError:
        return default


def _showpac_dyncon_settings() -> tuple[int, int, float]:
    """max dynCon URLs per showPac, parallel HTTP workers, per-request read timeout (s)."""
    md = _env_bounded_int("OBS_SHOWPAC_MAX_DYNCON", 12, 4, 40)
    workers = _env_bounded_int("OBS_DYNCON_HTTP_WORKERS", 4, 1, 12)
    # OBS often exceeds 14s under load; connect stays short, read uses this cap (max 55s).
    tout = _env_bounded_float("OBS_DYNCON_HTTP_TIMEOUT", 24.0, 10.0, 55.0)
    return md, workers, tout

# English labels commonly used on OBS showPac left navigation (substring match).
_SHOWPAC_SECTION_KEYWORDS: tuple[str, ...] = (
    "course structure",
    "program learning outcomes",
    "course & programme outcomes matrix",
    "nqf - fields",
    "about programme",
    "qualification awarded",
    "graduation requirements",
    "specific admission requirements",
    "programme director",
    "occupational profiles",
    "ects catalog",
    "academic staff",
    "contact information",
    "access to further studies",
    "recognition of prior learning",
    "type of education",
    "profile of the programme",
)

# Embedded content uses dynConPage.aspx; program details use showPac in the URL.
_DYNCON_RE = re.compile(r"dynConPage\.aspx\?[^\"'\\s<>]+", re.IGNORECASE)
_SHOWPAC_ABS_RE = re.compile(
    r"https?://[^\s\"'<>]+showpac[^\s\"'<>]*", re.IGNORECASE
)
# Relative URLs and quoted fragments in onclick/HTML attributes.
_LOOSE_SHOWPAC_RE = re.compile(r"[^\s\"'<>]*showPac[^\s\"'<>]*", re.IGNORECASE)
_UNITSEL_RE = re.compile(
    r"[^\s\"'<>]*unitSelection\.aspx[^\s\"'<>]*", re.IGNORECASE
)


def _http_get_text(
    url: str,
    timeout: float | tuple[float, float] = HTTP_TIMEOUT,
) -> str:
    """Plain HTTP GET for OBS HTML (works in Docker without Chrome)."""
    r = requests.get(
        url,
        timeout=timeout,
        headers={"User-Agent": USER_AGENT},
    )
    r.raise_for_status()
    return r.text or ""


def _fetch_dyncon_text_block(du: str, req_timeout: float) -> str:
    read_t = max(8.0, min(55.0, float(req_timeout)))
    connect_t = min(10.0, max(4.0, read_t * 0.35))
    timeout_pair: tuple[float, float] = (connect_t, read_t)
    retriable = (
        req_exc.ReadTimeout,
        req_exc.ConnectTimeout,
        req_exc.ConnectionError,
        req_exc.ChunkedEncodingError,
    )
    for attempt in range(2):
        try:
            d_html = _http_get_text(du, timeout=timeout_pair)
            d_title, d_text, _u = extract_title_text_and_embedding_units(d_html)
            body = (d_text or "").strip()
            if not body:
                return ""
            d_head = (d_title or du)[:160]
            return f"[{d_head}]\n{body[:8000]}"
        except retriable as e:
            if attempt == 0:
                time.sleep(0.45)
                continue
            logger.debug("dynCon HTTP fetch failed %s: %s", du, e)
            return ""
        except Exception as e:
            logger.debug("dynCon HTTP fetch failed %s: %s", du, e)
            return ""
    return ""


def append_showpac_dyncon_via_http(
    shell_html: str,
    base_url: str,
    *,
    target_lang: str = "en",
    max_detail: int | None = None,
    between_sleep: float = 0.12,
) -> str:
    """
    showPac shell pages mostly list nav labels; real programme text lives on dynConPage.
    Fetch those detail pages over HTTP and return appended plain text blocks.
    Parallelism: OBS_DYNCON_HTTP_WORKERS (default 4). Caps: OBS_SHOWPAC_MAX_DYNCON,
    OBS_DYNCON_HTTP_TIMEOUT (per request, seconds).
    """
    if "showpac" not in (base_url or "").lower():
        return ""
    env_md, workers, req_timeout = _showpac_dyncon_settings()
    eff_max = env_md if max_detail is None else max(1, min(40, max_detail))
    _sp, other = _urls_from_html_regex(shell_html, base_url)
    detail_urls = [
        u
        for u in sorted(set(other))
        if "dynconpage" in u.lower() and _preferred_lang(u, target_lang)
    ][:eff_max]
    if not detail_urls:
        return ""

    if workers <= 1:
        blocks: list[str] = []
        for du in detail_urls:
            if between_sleep > 0:
                time.sleep(between_sleep)
            b = _fetch_dyncon_text_block(du, req_timeout)
            if b:
                blocks.append(b)
        return "\n\n".join(blocks)

    n_workers = min(workers, len(detail_urls))
    ordered: list[str | None] = [None] * len(detail_urls)

    def _work(idx_du: tuple[int, str]) -> tuple[int, str]:
        i, du = idx_du
        return i, _fetch_dyncon_text_block(du, req_timeout)

    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futures = {ex.submit(_work, (i, du)): i for i, du in enumerate(detail_urls)}
        for fut in as_completed(futures):
            i, block = fut.result()
            if block:
                ordered[i] = block
    return "\n\n".join(b for b in ordered if b)


def fetch_page_extract_http(
    url: str, delay: float = 0.12, target_lang: str = "en"
) -> tuple[str, str, list[str]]:
    """
    Fetch a single OBS/Bologna URL with requests (no browser).
    For showPac programme shells, merges linked dynConPage bodies.
    """
    try:
        html = _http_get_text(url)
    except Exception as e:
        logger.warning("HTTP fetch failed %s: %s", url, e)
        return url[:500], "", []
    title, text, units = extract_title_text_and_embedding_units(html)
    if "showpac" in (url or "").lower():
        extra = append_showpac_dyncon_via_http(
            html,
            url,
            target_lang=target_lang,
            between_sleep=max(0.0, delay),
        )
        if extra:
            text = f"{(text or '').strip()}\n\n{extra}".strip()
    if not title:
        title = url[:500]
    return title, text, units


def build_driver() -> WebDriver:
    """Headless Chrome/Chromium. Docker slim has no browser; use host or an image with chromium.

    Env: CHROME_BINARY / GOOGLE_CHROME_BIN / CHROMIUM_BIN — browser executable.
    Env: CHROMEDRIVER_PATH — chromedriver executable (skips Selenium Manager when set).
    Falls back to PATH ``chromedriver``, then Selenium Manager, then ``webdriver-manager``.
    """
    from selenium import webdriver

    opts = Options()
    opts.add_argument("--headless=new")
    opts.add_argument("--window-size=1920,1080")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument(f"--user-agent={USER_AGENT}")
    opts.add_experimental_option("excludeSwitches", ["enable-automation"])

    chrome_bin = (
        os.environ.get("CHROME_BINARY")
        or os.environ.get("GOOGLE_CHROME_BIN")
        or os.environ.get("CHROMIUM_BIN")
    )
    if not chrome_bin and sys.platform == "darwin":
        mac_chrome = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
        if os.path.isfile(mac_chrome):
            chrome_bin = mac_chrome
    if chrome_bin:
        opts.binary_location = chrome_bin

    def _launch(service: Service) -> WebDriver:
        d = webdriver.Chrome(service=service, options=opts)
        # Without this, driver.get() can block many minutes on slow/hanging OBS pages (e.g. facAbout).
        load_s = _env_bounded_float("OBS_PAGE_LOAD_TIMEOUT", 25.0, 12.0, 180.0)
        d.set_page_load_timeout(load_s)
        return d

    drv = os.environ.get("CHROMEDRIVER_PATH", "").strip()
    if drv and os.path.isfile(drv):
        return _launch(Service(executable_path=drv))

    which_drv = shutil.which("chromedriver")
    if which_drv:
        return _launch(Service(executable_path=which_drv))

    try:
        return _launch(Service())
    except (WebDriverException, NoSuchDriverException, OSError):
        pass

    try:
        from webdriver_manager.chrome import ChromeDriverManager

        return _launch(Service(ChromeDriverManager().install()))
    except ImportError as exc:
        raise RuntimeError(
            "Chrome veya chromedriver bulunamadı. macOS’ta Google Chrome kurun; "
            "veya CHROMEDRIVER_PATH / PATH üzerinde chromedriver tanımlayın. "
            "Docker’daki python:slim imajında tarayıcı yok — bu scraper’ları host’ta "
            "çalıştırın (venv) ve POSTGRES_* ile Docker DB’ye bağlanın; veya "
            "imaja Chromium ekleyin."
        ) from exc


def normalize_obs_url(href: str, base: str) -> str:
    if not href:
        return ""
    h = href.strip()
    if h.startswith("#") or h.lower().startswith("javascript:"):
        return ""
    joined = urljoin(base, h)
    url, _frag = urldefrag(joined)
    url = url.replace("&amp;", "&")
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return ""
    host = parsed.netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    if host != OBS_HOST:
        return ""
    path = (parsed.path or "").lower()
    if BOLOGNA_PATH_MARKER not in path and "showpac" not in url.lower():
        return ""
    return url


def _preferred_lang(url: str, target_lang: str) -> bool:
    """Drop obvious language-switch duplicates (e.g. index?lang=tr when scraping en)."""
    if target_lang.lower() != "en":
        return True
    q = parse_qs(urlparse(url).query.lower())
    if "lang" in q and q["lang"] == ["tr"]:
        return False
    return True


def _urls_from_html_regex(html: str, base: str) -> tuple[set[str], set[str]]:
    """Extract showPac and dynCon URLs embedded in onclick/JS (not always in <a href>)."""
    showpac: set[str] = set()
    other: set[str] = set()
    if not html:
        return showpac, other
    clean = html.replace("&amp;", "&")
    for m in _SHOWPAC_ABS_RE.finditer(clean):
        nu = normalize_obs_url(m.group(0).strip(), base)
        if nu:
            showpac.add(nu)
    bologna_base = f"https://{OBS_HOST}{BOLOGNA_PATH_MARKER}"
    for m in _DYNCON_RE.finditer(clean):
        path = m.group(0)
        nu = normalize_obs_url(urljoin(bologna_base, path), base)
        if nu:
            other.add(nu)
    for m in _UNITSEL_RE.finditer(clean):
        raw = m.group(0).strip('\'"')
        if raw.startswith("//"):
            raw = "https:" + raw
        nu = normalize_obs_url(urljoin(bologna_base, raw), base)
        if nu:
            other.add(nu)
    seen_loose: set[str] = set()
    for m in _LOOSE_SHOWPAC_RE.finditer(clean):
        raw = m.group(0).strip('\'"')
        if len(raw) < 8 or raw.lower().startswith("javascript"):
            continue
        if raw in seen_loose:
            continue
        seen_loose.add(raw)
        if raw.startswith("//"):
            raw = "https:" + raw
        if not raw.lower().startswith("http"):
            raw = urljoin(bologna_base, raw.lstrip("/"))
        nu = normalize_obs_url(raw, base)
        if nu:
            showpac.add(nu)
    return showpac, other


def _page_ready_timeout_for_url(url: str) -> float:
    u = (url or "").lower()
    if "facabout" in u:
        return min(float(PAGE_READY_TIMEOUT), PAGE_READY_TIMEOUT_FACABOUT)
    return float(PAGE_READY_TIMEOUT)


def wait_for_page_ready(driver: WebDriver, timeout: float = PAGE_READY_TIMEOUT) -> None:
    try:
        WebDriverWait(driver, timeout).until(
            lambda d: d.execute_script("return document.readyState") == "complete"
        )
        WebDriverWait(driver, min(timeout, SHORT_WAIT)).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
    except TimeoutException:
        logger.warning("Timeout waiting for document.readyState complete")


def driver_get_obs(driver: WebDriver, url: str) -> None:
    """Like driver.get but bounded by OBS_PAGE_LOAD_TIMEOUT; keeps partial DOM on slow pages."""
    try:
        driver.get(url)
    except TimeoutException:
        logger.warning(
            "OBS page load hit OBS_PAGE_LOAD_TIMEOUT; stopping load, using partial DOM: %s",
            url,
        )
        try:
            driver.execute_script("window.stop();")
        except WebDriverException:
            pass


def _should_skip_element(href: str, text: str, skip_parts: list[str]) -> bool:
    if not skip_parts:
        return False
    blob = f"{href or ''} {text or ''}".lower()
    return any(p.strip() and p.strip().lower() in blob for p in skip_parts)


def _gather_anchor_hrefs(driver: WebDriver, base: str) -> tuple[set[str], set[str]]:
    """Returns (showpac_urls, other_bologna_urls)."""
    showpac: set[str] = set()
    other: set[str] = set()
    try:
        anchors = driver.find_elements(By.CSS_SELECTOR, "a[href]")
    except WebDriverException as e:
        logger.warning("Could not list anchors: %s", e)
        return showpac, other

    for a in anchors:
        try:
            raw = a.get_attribute("href") or ""
        except StaleElementReferenceException:
            continue
        if "showpac" not in raw.lower():
            nu = normalize_obs_url(raw, base)
            if nu:
                other.add(nu)
            continue
        nu = normalize_obs_url(raw, base)
        if nu:
            showpac.add(nu)
    return showpac, other


def _gather_from_all_frames(driver: WebDriver, base: str) -> tuple[set[str], set[str]]:
    """Walk default document and nested iframes; collect anchor hrefs + regex URLs from each frame's HTML."""
    showpac: set[str] = set()
    other: set[str] = set()

    def merge_html(html: str) -> None:
        sp, ot = _urls_from_html_regex(html, base)
        showpac.update(sp)
        other.update(ot)

    def walk_frames(depth: int = 0) -> None:
        if depth > 8:
            return
        try:
            html = driver.page_source
            merge_html(html)
            sp, ot = _gather_anchor_hrefs(driver, driver.current_url or base)
            showpac.update(sp)
            other.update(ot)
        except WebDriverException as e:
            logger.warning("gather frame: %s", e)

        try:
            frames = driver.find_elements(By.TAG_NAME, "iframe")
        except WebDriverException:
            return

        for f in frames:
            try:
                driver.switch_to.frame(f)
                time.sleep(0.05)
                walk_frames(depth + 1)
                driver.switch_to.parent_frame()
            except WebDriverException as e:
                logger.warning("iframe switch: %s", e)
                try:
                    driver.switch_to.parent_frame()
                except WebDriverException:
                    pass

    try:
        driver.switch_to.default_content()
        walk_frames(0)
    finally:
        try:
            driver.switch_to.default_content()
        except WebDriverException:
            pass

    return showpac, other


def _snapshot_html_from_all_frames(driver: WebDriver) -> list[str]:
    """
    Return HTML snapshots from default document + nested iframes.
    Useful when a showPac page updates content via JS/postback without URL change.
    """
    out: list[str] = []

    def walk(depth: int = 0) -> None:
        if depth > 8:
            return
        try:
            html = driver.page_source or ""
            if html and html not in out:
                out.append(html)
        except WebDriverException:
            return
        try:
            frames = driver.find_elements(By.TAG_NAME, "iframe")
        except WebDriverException:
            return
        for f in frames:
            try:
                driver.switch_to.frame(f)
                time.sleep(0.04)
                walk(depth + 1)
                driver.switch_to.parent_frame()
            except WebDriverException:
                try:
                    driver.switch_to.parent_frame()
                except WebDriverException:
                    pass

    try:
        driver.switch_to.default_content()
        walk(0)
    finally:
        try:
            driver.switch_to.default_content()
        except WebDriverException:
            pass
    return out


def _link_blob_for_showpac_nav(el) -> str:
    try:
        parts = [
            el.text or "",
            el.get_attribute("title") or "",
            el.get_attribute("aria-label") or "",
            el.get_attribute("onclick") or "",
            el.get_attribute("href") or "",
        ]
        return " ".join(parts).casefold()
    except StaleElementReferenceException:
        return ""


def _showpac_nav_rank(blob: str) -> tuple[int, int]:
    """(tier, order): tier 0 = keyword match (lower order = higher priority)."""
    if "dynconpage" not in blob:
        return (9, 9999)
    for i, kw in enumerate(_SHOWPAC_SECTION_KEYWORDS):
        if kw.casefold() in blob:
            return (0, i)
    return (1, 9999)


def _gather_showpac_menu_close_anchors(driver: WebDriver) -> list:
    """Collect menu_close→dynCon anchors from default document + shallow iframes."""
    found: list = []
    seen: set[tuple[str, str]] = set()

    def push_elements(els: list) -> None:
        for el in els:
            try:
                onclick = (el.get_attribute("onclick") or "").strip()
                text = (el.text or "").strip()[:240]
                key = (onclick[:520], text)
                if key in seen:
                    continue
                if "menu_close" not in onclick.lower() or "dynconpage" not in onclick.lower():
                    continue
                seen.add(key)
                found.append(el)
            except StaleElementReferenceException:
                continue

    try:
        driver.switch_to.default_content()
        push_elements(
            driver.find_elements(
                By.CSS_SELECTOR, "a[onclick*='menu_close'][onclick*='dynConPage']"
            )
        )
        frames = driver.find_elements(By.TAG_NAME, "iframe")
        for f in frames[:6]:
            try:
                driver.switch_to.frame(f)
                push_elements(
                    driver.find_elements(
                        By.CSS_SELECTOR,
                        "a[onclick*='menu_close'][onclick*='dynConPage']",
                    )
                )
                driver.switch_to.parent_frame()
            except WebDriverException:
                try:
                    driver.switch_to.parent_frame()
                except WebDriverException:
                    pass
        driver.switch_to.default_content()
    except WebDriverException as e:
        logger.warning("showPac nav gather failed: %s", e)
        try:
            driver.switch_to.default_content()
        except WebDriverException:
            pass
    return found


def _expand_showpac_sections_and_capture_html(driver: WebDriver, delay: float) -> str:
    """
    On a showPac page, open a *small* set of Information Package sidebar links so
    embedded ``dynConPage`` URLs appear in HTML for HTTP follow-up.

    Intentionally does **not** run the index-scale ``__doPostBack`` expander (that
    made each programme page take many minutes).
    """
    snapshots: list[str] = []

    def capture() -> None:
        for html in _snapshot_html_from_all_frames(driver):
            if html and html not in snapshots:
                snapshots.append(html)

    capture()
    anchors = _gather_showpac_menu_close_anchors(driver)
    scored: list[tuple[tuple[int, int], object]] = []
    for el in anchors:
        rk = _showpac_nav_rank(_link_blob_for_showpac_nav(el))
        scored.append((rk, el))
    scored.sort(key=lambda it: it[0])

    priority = [el for rk, el in scored if rk[0] == 0]
    fallback = [el for rk, el in scored if rk[0] != 0][: max(0, _MAX_SHOWPAC_SIDEBAR_CLICKS - len(priority))]
    candidates = (priority + fallback)[:_MAX_SHOWPAC_SIDEBAR_CLICKS]

    nav_sleep = min(_SHOWPAC_NAV_CLICK_SLEEP, max(0.08, float(delay) * 0.12))
    clicked_fp: set[tuple[str, str]] = set()
    clicks = 0

    for el in candidates:
        if clicks >= _MAX_SHOWPAC_SIDEBAR_CLICKS:
            break
        try:
            onclick = (el.get_attribute("onclick") or "").strip()
            text = (el.text or "").strip()[:200]
            fp = (onclick[:520], text)
            if fp in clicked_fp:
                continue
        except StaleElementReferenceException:
            continue

        try:
            driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", el)
            time.sleep(nav_sleep)
            try:
                el.click()
            except ElementNotInteractableException:
                driver.execute_script("arguments[0].click();", el)
            clicked_fp.add(fp)
            clicks += 1
            time.sleep(nav_sleep)
            wait_for_page_ready(driver, timeout=_SHOWPAC_NAV_READY_TIMEOUT)
            capture()
        except (
            StaleElementReferenceException,
            ElementClickInterceptedException,
            ElementNotInteractableException,
            TimeoutException,
            WebDriverException,
        ) as e:
            logger.debug("showPac nav click skipped: %s", e)
            continue
    try:
        driver.switch_to.default_content()
    except WebDriverException:
        pass
    return "\n\n".join(snapshots)


def _expand_postback_candidates(
    driver: WebDriver,
    delay: float,
    skip_parts: list[str],
    clicked: set[tuple[str, str, str]],
) -> int:
    """Click JavaScript / __doPostBack links to expand menus. Returns click count."""
    elements: list = []
    seen_fp: set[tuple[str, str, str]] = set()

    def _push_batch(found: list) -> None:
        for el in found:
            try:
                href = el.get_attribute("href") or ""
                onclick = el.get_attribute("onclick") or ""
                text = (el.text or "").strip()
                fp = (href[:600], onclick[:600], text[:200])
                if fp in seen_fp:
                    continue
                seen_fp.add(fp)
                elements.append(el)
            except StaleElementReferenceException:
                continue

    for sel in (
        "a[href*='__doPostBack']",
        "a[href*='javascript:']",
    ):
        try:
            _push_batch(driver.find_elements(By.CSS_SELECTOR, sel))
        except WebDriverException:
            continue

    try:
        _push_batch(
            driver.find_elements(By.XPATH, "//a[contains(@onclick,'__doPostBack')]")
        )
    except WebDriverException:
        pass

    clicks = 0
    for el in elements[:MAX_POSTBACK_CLICKS_PER_PASS]:
        if clicks >= MAX_POSTBACK_CLICKS_PER_PASS:
            break
        try:
            href = el.get_attribute("href") or ""
            onclick = el.get_attribute("onclick") or ""
            text = (el.text or "").strip()
        except StaleElementReferenceException:
            continue

        if _should_skip_element(href, f"{text} {onclick}", skip_parts):
            continue

        if "showpac" in href.lower() and "__dopostback" not in href.lower():
            continue

        fp = (href[:500], onclick[:500], text[:200])
        if fp in clicked:
            continue

        try:
            driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", el)
            time.sleep(min(0.2, delay / 3))
            try:
                el.click()
            except ElementNotInteractableException:
                driver.execute_script("arguments[0].click();", el)
            clicked.add(fp)
            clicks += 1
            time.sleep(delay)
            wait_for_page_ready(driver, timeout=EXPAND_CLICK_READY_TIMEOUT)
        except (
            StaleElementReferenceException,
            ElementClickInterceptedException,
            ElementNotInteractableException,
            TimeoutException,
            WebDriverException,
        ) as e:
            logger.warning("Expand click skipped: %s", e)
            continue
    return clicks


def _expand_menu_close_links(
    driver: WebDriver,
    delay: float,
    skip_parts: list[str],
    clicked: set[tuple[str, str, str]],
) -> int:
    """Click nav items that use menu_close(..., 'dynConPage.aspx?...') to load iframe content."""
    try:
        driver.switch_to.default_content()
        candidates = driver.find_elements(
            By.CSS_SELECTOR, "a[onclick*='menu_close'], a[onclick*='dynConPage']"
        )
    except WebDriverException as e:
        logger.warning("menu_close find failed: %s", e)
        return 0

    clicks = 0
    for el in candidates[:MAX_MENU_CLOSE_CLICKS]:
        try:
            onclick = el.get_attribute("onclick") or ""
            href = el.get_attribute("href") or ""
            text = (el.text or "").strip()
        except StaleElementReferenceException:
            continue

        if "dynconpage" not in onclick.lower():
            continue
        if _should_skip_element(href, f"{text} {onclick}", skip_parts):
            continue

        fp = (href[:200], onclick[:700], text[:200])
        if fp in clicked:
            continue

        try:
            driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", el)
            time.sleep(min(0.2, delay / 3))
            try:
                el.click()
            except ElementNotInteractableException:
                driver.execute_script("arguments[0].click();", el)
            clicked.add(fp)
            clicks += 1
            time.sleep(delay)
            wait_for_page_ready(driver, timeout=EXPAND_CLICK_READY_TIMEOUT)
        except (
            StaleElementReferenceException,
            ElementClickInterceptedException,
            ElementNotInteractableException,
            WebDriverException,
        ) as e:
            logger.debug("menu_close click skipped: %s", e)
            continue
    return clicks


def collect_bologna_urls(
    driver: WebDriver,
    delay: float,
    skip_section_parts: list[str] | None = None,
    target_lang: str = "en",
) -> list[str]:
    """
    Load the Bologna index, walk iframes, click expanders, collect showPac program URLs
    and other /oibs/bologna/ HTTP URLs. Returns sorted unique URLs (index first).
    """
    skip_parts = list(skip_section_parts or [])
    base = BOLOGNA_INDEX_URL
    all_showpac: set[str] = {
        u for u in BOLOGNA_SEED_URLS if _preferred_lang(u, target_lang)
    }
    all_other: set[str] = set()
    all_other.add(BOLOGNA_INDEX_URL)

    try:
        driver_get_obs(driver, BOLOGNA_INDEX_URL)
        time.sleep(delay)
        wait_for_page_ready(driver)
    except WebDriverException as e:
        logger.warning("Initial load failed: %s", e)
        fallback = [u for u in BOLOGNA_SEED_URLS if _preferred_lang(u, target_lang)]
        return [BOLOGNA_INDEX_URL, *fallback]

    clicked_fingerprints: set[tuple[str, str, str]] = set()

    # Discover section URLs from onclick / HTML (dynConPage) before interaction.
    sp0, ot0 = _urls_from_html_regex(driver.page_source, base)
    all_showpac |= sp0
    all_other |= ot0

    section_queue: list[str] = sorted(ot0)
    seen_section: set[str] = {BOLOGNA_INDEX_URL}
    # Each section visit is a full Selenium navigation; the queue can grow to 100+ URLs.
    section_nav_budget = _env_bounded_int("OBS_SECTION_QUEUE_CAP", 40, 5, 800)
    section_navs = 0

    for section_url in section_queue:
        if section_navs >= section_nav_budget:
            logger.warning(
                "OBS_SECTION_QUEUE_CAP=%s reached; stopping section discovery early "
                "(raise cap if you need more programme URLs).",
                section_nav_budget,
            )
            break
        if section_url in seen_section:
            continue
        if not _preferred_lang(section_url, target_lang):
            continue
        seen_section.add(section_url)
        section_navs += 1
        try:
            driver_get_obs(driver, section_url)
            time.sleep(delay)
            wait_for_page_ready(driver, timeout=_page_ready_timeout_for_url(section_url))
        except WebDriverException as e:
            logger.warning("Section GET failed %s: %s", section_url, e)
            continue

        sp, ot = _gather_from_all_frames(driver, driver.current_url or base)
        all_showpac |= sp
        for u in ot:
            if u not in seen_section and _preferred_lang(u, target_lang):
                all_other.add(u)
                if u not in section_queue and (
                    "dynconpage" in u.lower()
                    or "unitselection" in u.lower()
                ):
                    section_queue.append(u)

        spx, otx = _urls_from_html_regex(driver.page_source, driver.current_url or base)
        all_showpac |= spx
        all_other |= otx

    # Back to index for tree expansion + menu clicks (main nav lives on index).
    try:
        driver_get_obs(driver, BOLOGNA_INDEX_URL)
        time.sleep(delay)
        wait_for_page_ready(driver)
    except WebDriverException as e:
        logger.warning("Return to index failed: %s", e)

    expand_passes = _env_bounded_int(
        "OBS_EXPAND_PASSES", MAX_EXPAND_PASSES, 2, max(24, MAX_EXPAND_PASSES)
    )
    for pass_idx in range(expand_passes):
        before = len(all_showpac) + len(all_other)

        sp, ot = _gather_from_all_frames(driver, driver.current_url or base)
        all_showpac |= sp
        for u in ot:
            if _preferred_lang(u, target_lang):
                all_other.add(u)

        _expand_menu_close_links(driver, delay, skip_parts, clicked_fingerprints)
        sp, ot = _gather_from_all_frames(driver, driver.current_url or base)
        all_showpac |= sp
        for u in ot:
            if _preferred_lang(u, target_lang):
                all_other.add(u)

        _expand_postback_candidates(driver, delay, skip_parts, clicked_fingerprints)
        sp, ot = _gather_from_all_frames(driver, driver.current_url or base)
        all_showpac |= sp
        for u in ot:
            if _preferred_lang(u, target_lang):
                all_other.add(u)

        after = len(all_showpac) + len(all_other)
        if after == before and pass_idx > 0:
            break

    merged = all_showpac | all_other
    merged = {u for u in merged if _preferred_lang(u, target_lang)}
    ordered = sorted(merged)
    if BOLOGNA_INDEX_URL in merged:
        ordered.remove(BOLOGNA_INDEX_URL)
        ordered.insert(0, BOLOGNA_INDEX_URL)
    return ordered


def fetch_page_extract(
    driver: WebDriver,
    url: str,
    delay: float,
    retries: int = 1,
    *,
    target_lang: str = "en",
) -> tuple[str, str, list[str]]:
    """Navigate to url and return (title, text, embedding_units)."""
    last_exc: Exception | None = None
    for attempt in range(retries + 1):
        try:
            logger.debug("OBS fetch start: %s", url)
            driver_get_obs(driver, url)
            time.sleep(delay)
            wait_for_page_ready(driver, timeout=_page_ready_timeout_for_url(url))
            html = driver.page_source
            title, text, units = extract_title_text_and_embedding_units(html)
            # showPac pages are mostly a nav shell; dynConPage bodies carry programme text.
            # Use HTTP (not extra Selenium navigations) so Docker without Chrome still works.
            if "showpac" in (url or "").lower():
                nav_html = _expand_showpac_sections_and_capture_html(driver, max(0.05, delay / 2))
                html_for_dyncon = f"{html}\n\n{nav_html}" if nav_html else html
                extra = append_showpac_dyncon_via_http(
                    html_for_dyncon,
                    url,
                    target_lang=target_lang or "en",
                    between_sleep=max(0.05, delay / 4),
                )
                if extra:
                    text = f"{(text or '').strip()}\n\n{extra}".strip()
            if not title:
                title = url[:500]
            return title, text, units
        except WebDriverException as e:
            last_exc = e
            logger.warning("fetch_page_extract attempt %s failed for %s: %s", attempt, url, e)
            time.sleep(delay)
    if last_exc:
        logger.warning("Giving up on %s: %s", url, last_exc)
    return url[:500], "", []
