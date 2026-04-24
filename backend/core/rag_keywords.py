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

# "Who are the teachers?" / "faculty" — prefer department /academic-staff/ pages in RAG.
RAG_FACULTY_ROSTER_INTENT_RE = re.compile(
    r"\bteachers?\b|teaching\s+staff|"
    r"\b(prof|faculty|professors?|instructors?|lecturers?)\b|"
    r"öğretim|ogretim|akademik\s+kadro|eğitim\s+kadrosu|egitim\s+kadrosu|"
    r"\bhocalar|ders\s+veren|educators?\b|academic\s+staff\b|"
    r"kadro(ya)?\b",
    re.IGNORECASE,
)

# Query substring → URL path segment under .../departments/<segment>/ (see site structure).
_FACULTY_ROSTER_PATHS: list[tuple[tuple[str, ...], str]] = [
    (
        (
            "computer engineering",
            "bilgisayar mühendisliği",
            "bilgisayar",
            "mühendisliği bölüm",
            "bolum bilgisayar",
        ),
        "computer-engineering",
    ),
    (("electrical", "elektrik", "elektronik", "electronics"), "electrical"),
    (("mechanical", "makine"), "mechanical"),
    (("civil", "inşaat", "insaat", "mimarlık inşaat"), "civil"),
    (("industrial", "endüstri", "endustri"), "industrial"),
    (("biomedical", "biyomedikal", "biyom"), "biomedical"),
    (("software", "yazılım", "yazilim", "yazilim m"), "software"),
]


def faculty_roster_path_filter(query: str) -> str | None:
    """URL segment e.g. computer-engineering, or None if no department match."""
    ql = (query or "").lower()
    for needles, path_seg in _FACULTY_ROSTER_PATHS:
        if any(n in ql for n in needles):
            return path_seg
    return None


# Deans, rector, university leadership (not the same as “teachers / academic staff list”).
RAG_LEADERSHIP_INTENT_RE = re.compile(
    r"\b(dean|deans|dekan|rector|rektör|rectorate|dekanlık|dekanlik|provost)\b",
    re.IGNORECASE,
)

# Tuition / price (shared by compose_rag_search_query + retrieval filters)
RAG_FEE_TUITION_INTENT_RE = re.compile(
    r"\b(price|prices|fee|fees|tuition|ücret|ücreti|ücretler|"
    r"scholarship|burs|how\s+much|what\s+.*\s+cost|costs?|"
    r"annual|yıllık|yillik|öğrenim|ogrenim|payment|ödeme|odeme)\b",
    re.IGNORECASE,
)

# User wants university-wide or full fee list, not a single department.
RAG_BROAD_FEE_LIST_INTENT_RE = re.compile(
    r"\b(tüm|bütün|butun|hepsi|all|every|"
    r"tüm\s+ücret|bütün\s+ücret|tum\s+ücret|tüm\s+program|bütün\s+program|"
    r"tüm\s+bölüm|bütün\s+bölüm|tüm\s+fakülte|bütün\s+fakülte|"
    r"all\s+programs?|all\s+fees?|all\s+departments?|all\s+tuition|all\s+prices?|"
    r"list\s+of\s+fees?|fee\s+list|ücret\s+listesi|ücretler\s+ne|"
    r"what\s+are\s+the\s+fees|tüm\s+öğrenim|bütün\s+öğrenim)\b",
    re.IGNORECASE,
)


def fee_tuition_intent(text: str) -> bool:
    return bool(RAG_FEE_TUITION_INTENT_RE.search(text or ""))


def is_university_wide_fee_rag_query(text: str) -> bool:
    """
    True: user asked for all / general program fees, not a single known department path.
    Used for extra embedding + compose line (retrieve schedule-style pages, not one dept only).
    """
    t = text or ""
    if not RAG_FEE_TUITION_INTENT_RE.search(t):
        return False
    if RAG_BROAD_FEE_LIST_INTENT_RE.search(t):
        return True
    return faculty_roster_path_filter(t) is None


def leadership_embedding_phrase(query: str) -> str | None:
    """Steer retrieval toward pages that name deans/rector/faculty leadership, not generic contact."""
    q = f"{query or ''}".strip()
    if not q or not RAG_LEADERSHIP_INTENT_RE.search(q):
        return None
    return (
        "Acıbadem Mehmet Ali Aydınlar University faculty deans rector leadership "
        "organization schools management board vice rector"
    )


def faculty_list_embedding_phrase(query: str) -> str | None:
    """Extra embedding line so vector search hits /academic-staff/ pages (not generic /about)."""
    q = f"{query or ''}".strip()
    if not q or not RAG_FACULTY_ROSTER_INTENT_RE.search(q):
        return None
    if not RAG_STEM_OR_ENGINEERING_INTENT_RE.search(q) and not faculty_roster_path_filter(q):
        return None
    seg = faculty_roster_path_filter(q) or "engineering"
    label = seg.replace("-", " ")
    return (
        f"Acıbadem University {label} department academic staff list faculty members "
        f"professors and teaching staff"
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
