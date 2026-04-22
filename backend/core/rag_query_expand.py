"""
Rule-based RAG query expansion for asymmetric person-name retrieval.

When users ask e.g. "Who is X?" the embedding of the bare question can miss faculty
"message from the head" pages; we add a few no-LLM string variants (role/department
phrasing) and merge them in vector search. See rag_retrieval._embedding_variants.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Optional

# 1:1 map so string indices are preserved (for snippet windows) while matching TR/Latin names.
_TR_TO_LATIN: dict[str, str] = {
    "ı": "i",
    "İ": "i",
    "ş": "s",
    "Ş": "s",
    "ğ": "g",
    "Ğ": "g",
    "ü": "u",
    "Ü": "u",
    "ö": "o",
    "Ö": "o",
    "ç": "c",
    "Ç": "c",
}


def fold_for_whois_match(s: str) -> str:
    """Lowercase, NFKC, Turkish letters folded to Latin so 'Ziraksima' matches 'Zıraksıma'."""
    u = unicodedata.normalize("NFKC", s or "")
    return "".join(_TR_TO_LATIN.get(c, c) for c in u).lower()

# English: "Who is X?", "Who's X?" — not anchored to line start
_RE_WHO_EN = re.compile(
    r"\bwho(?:'s|\s+is)\s+(.+?)(?:\s*[\?\.!]|$)",
    re.IGNORECASE,
)
_RE_KIMDIR_SUF = re.compile(r"^\s*([\w\s\.\'’-]+?)\s+kimdir\b", re.IGNORECASE)
_RE_KIM_AT_END = re.compile(
    r"^\s*([\w\s\.\'’-]+?)\s+kim\s*[\?\.!]?\s*$",
    re.IGNORECASE,
)
_RE_KIMDIR_PRE = re.compile(r"^\s*kimdir\s+([\w\s\.\'’-]+?)\s*[\?\.!]?\s*$", re.IGNORECASE)
_RE_ROLE_FOCUSED = re.compile(
    r"\b(head|chair|dean|director|bolum|bölüm|baskan|başkan|müdür|mudur|dekan)\b",
    re.IGNORECASE,
)
_STRIP_TITLES = re.compile(
    r"^(?:dr\.?|doç\.?|doc\.?|prof\.?|yrd\.?|asst\.?|assoc\.?|öğr\.?|assistant|associate)\s*",
    re.IGNORECASE,
)


def _nfkc(s: str) -> str:
    return unicodedata.normalize("NFKC", s or "")


def _collapse_ws(s: str) -> str:
    return " ".join((s or "").split())


def _strip_leading_titles(s: str) -> str:
    t = _collapse_ws(s)
    for _ in range(6):
        n = _STRIP_TITLES.sub("", t, count=1).strip()
        if n == t:
            break
        t = n
    return t


def _is_plausible_name(parts: list[str], _raw_blob: str) -> bool:
    if not parts:
        return False
    if parts[0].lower() in {
        "the",
        "a",
        "an",
        "our",
        "this",
        "that",
        "what",
        "which",
        "acu",
        "acibadem",
        "acıbadem",
    }:
        return False
    joined = " ".join(parts)
    if len(joined) < 5 or (len(parts) == 1 and len(joined) < 8):
        return False
    return True


def _extract_name_turkish(t: str, blob: str) -> Optional[str]:
    m = _RE_KIMDIR_SUF.match(t) or _RE_KIMDIR_PRE.match(t) or _RE_KIM_AT_END.match(t)
    if not m:
        return None
    c = _strip_leading_titles(m.group(1))
    parts = [p for p in c.split() if p][:4]
    if not _is_plausible_name(parts, blob):
        return None
    if _RE_ROLE_FOCUSED.search(" ".join(parts).lower()) and len(parts) < 2:
        return None
    return " ".join(parts)


def _extract_name(blob: str) -> Optional[str]:
    t = _nfkc(_collapse_ws(blob))
    if not t:
        return None
    n = _extract_name_turkish(t, blob)
    if n:
        return n
    m3 = _RE_WHO_EN.search(t)
    if m3:
        c = re.sub(r"[\?\.!]+$", "", m3.group(1).strip())
        c = _strip_leading_titles(c)
        parts = [p for p in c.split() if p][:4]
        if not _is_plausible_name(parts, blob):
            return None
        return " ".join(parts)
    return None


def whois_name_from_queries(composed: str, raw: Optional[str]) -> Optional[str]:
    """
    If the message looks like a person-identity question ("who is X", "X kimdir", "X kim"),
    return a cleaned name string; otherwise None.

    The RAG *composed* query often appends "Acibadem... University" (see compose_rag_search_query);
    that breaks multiline "who is X" matching. We therefore extract the name from the short
    *raw* user message when present, and only fall back to composed.
    """
    raw_s = (raw or "").strip()
    comp = (composed or "").strip()
    if not raw_s and not comp:
        return None
    combined = f"{raw_s}\n{comp}".strip()
    # Prefer plain user line for English / TR name regex (avoids "who is X\\n...long ACU line")
    primary = raw_s or comp
    # Pure role questions without a person: do not run expensive boosts
    if _RE_ROLE_FOCUSED.search(combined) and not re.search(
        r"(\bwho\s+is\b|\'s|kimdir|^\s*[\w\s\.\'’-]+\s+kimdir|\bkim\s*[\?\.!]?\s*$)",
        primary,
        re.IGNORECASE | re.MULTILINE,
    ):
        if not re.search(
            r"[A-ZÇĞİÖŞÜa-zçğıöşü]{2,}\s+[A-ZÇĞİÖŞÜa-zçğıöşü]{2,}", combined
        ):
            return None
    n = _extract_name(primary)
    if n and 5 <= len(n) <= 120:
        return n
    return None


def snippet_around_phrase(text: str, phrase: str | None, max_len: int) -> str:
    """
    If phrase appears in text, return a window of up to max_len characters centered
    on the first occurrence. Otherwise the first max_len chars (RAG default behavior).

    Long faculty/chunk text often has the person name after the default prefix slice
    (e.g. 1100 chars), so the LLM would not see the name in CONTEXT.
    """
    t = text or ""
    if not t or max_len < 1:
        return ""
    L = min(max_len, len(t))
    if not phrase or len(phrase.strip()) < 2:
        return t[:L]
    p = phrase.strip()
    t_fold = fold_for_whois_match(t)
    p_fold = fold_for_whois_match(p)
    i = t_fold.find(p_fold)
    if i < 0:
        for tok in sorted([x for x in p.split() if len(x) >= 3], key=len, reverse=True):
            tf = fold_for_whois_match(tok)
            j = t_fold.find(tf)
            if j >= 0:
                i = j
                p = tok
                p_fold = tf
                break
    if i < 0:
        return t[:L]
    if len(t) <= L:
        return t
    # Same byte length for fold / original, so i aligns with t for 1:1 map
    span = max(len(p_fold), 1)
    center = i + span // 2
    start = max(0, min(center - L // 2, len(t) - L))
    return t[start : start + L]


def whois_vector_variants(name: str, max_variants: int = 3) -> list[str]:
    extra = _collapse_ws(_nfkc(name or ""))
    if len(extra) < 4:
        return []
    out = [
        f"{extra} head of department message from head faculty staff",
        f"{extra} computer engineering department chair message faculty",
        f"{extra} bolum baskan department chair associate professor",
        f"{extra} Acibadem academic staff contact",
    ]
    return out[: max(0, max_variants)]


def whois_name_in_content(content: str, anchor: str) -> bool:
    """
    True if the chunk text plausibly mentions this who-is name.
    Uses fold_for_whois_match so 'Ziraksima' (user/Latin) matches 'Zıraksıma' (page Turkish).
    """
    a = (anchor or "").strip()
    if not a or not (content or "").strip():
        return False
    fc = fold_for_whois_match(content)
    fa = fold_for_whois_match(a)
    if fa in fc:
        return True
    parts = [p for p in a.split() if len(p) >= 2]
    if len(parts) >= 2:
        return all(fold_for_whois_match(p) in fc for p in parts)
    return fa in fc
