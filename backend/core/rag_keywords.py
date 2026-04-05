"""Keyword and intent helpers for RAG compose + retrieval boosts."""

import re

_RAG_KEYWORD_STOP = frozenset(
    """
    what when where which who how why the and for with about from that this have does did you your
    are was were please dont not tell can could would should university universite universitesi
    acibadem acıbadem mehmet ali aydinlar aydınlar tell lie know please more some any very just
    like into than then them they their there here hakkında nedir nelerdir nasıl hangi şey
    """.split()
)

RAG_DEPT_OR_FACULTY_INTENT_RE = re.compile(
    r"depart|fakülte|fakulte|fakult\b|bölüm|bolum|facult(y|ies)|schools?\b|"
    r"program\s*list|bölümler|yüksekokul|anabilim|meslek\s*yüksek|mühendis|muhendis|tıp\s*fak",
    re.IGNORECASE,
)

RAG_STEM_OR_ENGINEERING_INTENT_RE = re.compile(
    r"\bcomputer\b|software|electrical|mechanical|civil|chemical|industrial|biomedical|"
    r"\bengineering\b|\bengineer\b|programming|informatics|"
    r"bilgisayar|yazılım|yazilim|elektrik|elektronik|makine|inşaat|insaat|yapay\s*zeka|veri\s*bilim",
    re.IGNORECASE,
)


def rag_keywords_from_query(text: str, max_terms: int = 5) -> list[str]:
    raw = (text or "").lower()
    words = re.findall(r"[a-zA-ZğüşıöçĞÜŞİÖÇ]{4,}", raw)
    out: list[str] = []
    for w in words:
        if w in _RAG_KEYWORD_STOP:
            continue
        if w not in out:
            out.append(w)
        if len(out) >= max_terms:
            break
    return out


def stem_engineering_boost_terms(query: str) -> list[str]:
    if not (query or "").strip():
        return []
    if not RAG_STEM_OR_ENGINEERING_INTENT_RE.search(query):
        return []
    ql = query.lower()
    out: list[str] = []
    if re.search(r"computer|bilgisayar|informatics|yazılım|yazilim|software", ql):
        # Phrases first (icontains order uses list order). Avoid lone "Bilgisayar" — matches too many pages.
        out.extend(
            [
                "Bilgisayar Mühendisliği",
                "Computer Engineering",
                "computer engineering",
            ]
        )
    if re.search(r"electrical|elektrik|elektronik|electronics", ql):
        out.extend(["Electrical", "Elektrik", "Electronics", "Elektronik"])
    if re.search(r"mechanical|makine", ql):
        out.extend(["Mechanical", "Makine"])
    if re.search(r"civil|inşaat|insaat", ql):
        out.extend(["Civil", "İnşaat", "Insaat"])
    if not out:
        out = ["Mühendisliği", "Engineering", "Faculty of Engineering", "Mühendislik"]
    seen: set[str] = set()
    deduped: list[str] = []
    for t in out:
        if t.lower() not in seen:
            seen.add(t.lower())
            deduped.append(t)
    return deduped


def structured_list_boost_terms(query: str) -> list[str]:
    if not (query or "").strip():
        return []
    q = query.strip()
    if not RAG_DEPT_OR_FACULTY_INTENT_RE.search(q):
        return []
    return [
        "Fakülte",
        "Faculty",
        "Department",
        "Yüksekokul",
        "Meslek Yüksekokulu",
        "School of",
    ]
