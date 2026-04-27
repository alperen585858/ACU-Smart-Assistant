"""Shared HTML → title + plain text extraction for crawlers."""
import os
import re

from bs4 import BeautifulSoup, NavigableString, Tag

# Keep enough page body so late sections (e.g. scholarships) are not truncated.
MAX_CONTENT_CHARS = max(3000, int(os.environ.get("CRAWL_MAX_CONTENT_CHARS", "20000")))
# Bounds for JSON storage and embedding prep (per unit / per page).
_MAX_UNIT_CHARS = max(300, int(os.environ.get("CRAWL_MAX_UNIT_CHARS", "1500")))
_MAX_UNITS_PER_PAGE = max(80, int(os.environ.get("CRAWL_MAX_UNITS_PER_PAGE", "320")))


def _strip_noise_tags(soup: BeautifulSoup) -> None:
    for tag in soup(["script", "style", "noscript", "svg"]):
        tag.decompose()


def _normalize_unit_text(s: str) -> str:
    t = re.sub(r"\n{3,}", "\n\n", (s or "").strip())
    if len(t) > _MAX_UNIT_CHARS:
        t = t[:_MAX_UNIT_CHARS].rstrip()
    return t


def _dedupe_key(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip().lower()


def _extract_embedding_units_from_root(root: Tag) -> list[str]:
    """Collect one string per DOM record (table row, list item, dl entry)."""
    seen: set[str] = set()
    out: list[str] = []

    def add(raw: str) -> None:
        t = _normalize_unit_text(raw)
        if len(t) < 3:
            return
        k = _dedupe_key(t)
        if not k or k in seen:
            return
        seen.add(k)
        out.append(t)
        if len(out) >= _MAX_UNITS_PER_PAGE:
            return

    for table in root.find_all("table"):
        for tr in table.find_all("tr"):
            if tr.find_parent("table") is not table:
                continue
            if not tr.find("td"):
                continue
            if len(out) >= _MAX_UNITS_PER_PAGE:
                break
            add(tr.get_text(separator="\n", strip=True))
        if len(out) >= _MAX_UNITS_PER_PAGE:
            break

    for lst in root.find_all(["ul", "ol"]):
        if len(out) >= _MAX_UNITS_PER_PAGE:
            break
        for li in lst.find_all("li", recursive=False):
            if not isinstance(li, Tag):
                continue
            if li.find_parent(["ul", "ol"]) is not lst:
                continue
            if li.find_parent("table") is not None:
                continue
            add(li.get_text(separator="\n", strip=True))

    for dl in root.find_all("dl"):
        if len(out) >= _MAX_UNITS_PER_PAGE:
            break
        for dt in dl.find_all("dt", recursive=False):
            if not isinstance(dt, Tag):
                continue
            if dt.find_parent("dl") is not dl:
                continue
            bits: list[str] = [dt.get_text(separator="\n", strip=True)]
            for sib in dt.next_siblings:
                if isinstance(sib, NavigableString):
                    continue
                if not isinstance(sib, Tag):
                    continue
                if sib.name == "dd":
                    bits.append(sib.get_text(separator="\n", strip=True))
                elif sib.name == "dt":
                    break
            add("\n".join(b for b in bits if b))

    return out


def extract_title_text_and_embedding_units(html: str) -> tuple[str, str, list[str]]:
    """
    Parse HTML once: document title, flattened body text (capped), and DOM record units.
    """
    soup = BeautifulSoup(html or "", "lxml")
    raw_title = ""
    if soup.title and soup.title.string:
        raw_title = soup.title.string.strip()
    _strip_noise_tags(soup)
    root = soup.find("main") or soup.find("article") or soup.body
    if root is None or not isinstance(root, Tag):
        text = ""
        units: list[str] = []
    else:
        text = root.get_text(separator="\n", strip=True)
        units = _extract_embedding_units_from_root(root)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    text = text[:MAX_CONTENT_CHARS]
    title = (raw_title or "")[:500]
    return title, text, units


def extract_title_and_text(html: str) -> tuple[str, str]:
    title, text, _units = extract_title_text_and_embedding_units(html)
    return title, text
