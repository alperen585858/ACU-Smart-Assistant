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

RAG_LOCATION_CONTACT_INTENT_RE = re.compile(
    r"\b(where\s+is|location|address|postal|contact|how\s+to\s+get|how\s+can\s+i\s+reach|reach\b|directions?|"
    r"transport|transportation|campus\s+location|communication\s+and\s+transportation|"
    r"nerede|konum|adres|iletişim|iletisim|ulaşım|ulasim|kamp[uü]s\s+konum)\b",
    re.IGNORECASE,
)

RAG_ACADEMIC_OBS_INTENT_RE = re.compile(
    r"\b(obs|oibs|bologna|course|courses|curriculum|syllabus|ects|catalog|"
    r"programme?\s+structure|program\s+outcomes?|learning\s+outcomes?|"
    r"ders|dersler|müfredat|mufredat|akts|katalog|kazan[ıi]m)\b",
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
# Faculty of Medicine before generic "medicine" heuristics; see faculty_roster_path_filter.
_FACULTY_ROSTER_PATHS: list[tuple[tuple[str, ...], str]] = [
    (
        (
            "faculty of health sciences",
            "health sciences",
            "health sciences faculty",
            "sağlık bilimleri fakültesi",
            "saglik bilimleri fakultesi",
            "sağlık bilimleri",
            "saglik bilimleri",
        ),
        "faculty-of-health-sciences",
    ),
    (
        (
            "faculty of medicine",
            "tıp fakültesi",
            "tıp fakultesi",
            "tip fakultesi",
            "hekimlik",
            "medical school",
            "doctor of medicine",
        ),
        "faculty-of-medicine",
    ),
    (
        (
            "computer programming",
            "computer programmer",
            "bilgisayar programcılığı",
            "bilgisayar programciligi",
            "programming",
        ),
        "computer-programming",
    ),
    (
        (
            "computer engineering",
            "bilgisayar mühendisliği",
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

_ENTITY_ALIAS_GROUPS: dict[str, tuple[str, ...]] = {
    "computer-programming": (
        "computer programming",
        "computer programmer",
        "bilgisayar programcılığı",
        "bilgisayar programciligi",
    ),
    "computer-engineering": (
        "computer engineering",
        "bilgisayar mühendisliği",
        "bilgisayar muhendisligi",
    ),
    "faculty-of-health-sciences": (
        "faculty of health sciences",
        "health sciences",
        "sağlık bilimleri fakültesi",
        "saglik bilimleri fakultesi",
    ),
    "faculty-of-medicine": (
        "faculty of medicine",
        "medical school",
        "tıp fakültesi",
        "tip fakultesi",
        "hekimlik",
    ),
}

_ENTITY_ORDER: tuple[str, ...] = (
    "computer-programming",
    "computer-engineering",
    "faculty-of-health-sciences",
    "faculty-of-medicine",
)


def extract_target_entity_key(query: str) -> str | None:
    ql = (query or "").lower()
    if not ql:
        return None
    for key in _ENTITY_ORDER:
        aliases = _ENTITY_ALIAS_GROUPS.get(key, ())
        if any(a in ql for a in aliases):
            return key
    return None


def target_entity_aliases(entity_key: str | None) -> tuple[str, ...]:
    if not entity_key:
        return ()
    aliases = _ENTITY_ALIAS_GROUPS.get(entity_key)
    return tuple(aliases) if aliases else ()


def target_entity_competitor_aliases(entity_key: str | None) -> tuple[str, ...]:
    if not entity_key:
        return ()
    out: list[str] = []
    for k, aliases in _ENTITY_ALIAS_GROUPS.items():
        if k == entity_key:
            continue
        out.extend(list(aliases))
    return tuple(dict.fromkeys(out))


def faculty_roster_path_filter(query: str) -> str | None:
    """URL segment e.g. computer-engineering, or None if no department match."""
    ql = (query or "").lower()
    key = extract_target_entity_key(ql)
    if key:
        return key
    for needles, path_seg in _FACULTY_ROSTER_PATHS:
        if any(n in ql for n in needles):
            return path_seg
    # "medicine" + price → Tıp Fakültesi / MD, not MYO "Medical X Techniques" rows on the same fee page
    if re.search(r"(?i)\bmedicine\b", ql) and re.search(
        r"(?i)\b(price|fee|fees|tuition|ücret|costs?|how\s+much|payment|ödeme|odeme)\b", ql
    ):
        if re.search(
            r"(?i)medical\s+education|t[ıi]p\s*e[ğg]itimi\s*(yüksek|master)|master[’']?s\s+in\s+medical",
            ql,
        ):
            return None
        if not re.search(
            r"(?i)(biomedic|biyomed|laborator|techniques?|imaging|podolog|podiatry|radiother|pathology|"
            r"secretar|documentation|veterinary|dental\s+nurs|myo\b|ön\s*lisans|on\s*lisans)",
            ql,
        ):
            return "faculty-of-medicine"
    return None


def department_snippet_anchor_phrases(query: str) -> list[str]:
    """
    Phrases to center RAG snippets on when the user names a department (esp. fee rows
    below the default chunk prefix). Longer / more specific phrases first.
    """
    seg = faculty_roster_path_filter(query)
    if not seg:
        return []
    for needles, path_seg in _FACULTY_ROSTER_PATHS:
        if path_seg != seg:
            continue
        phrases: list[str] = []
        # Prefer multi-word names so we land on the fee table row, not random "Bilgisayar" hits.
        for n in needles:
            n = n.strip()
            if len(n) >= 4:
                phrases.append(n)
        # Title-style label from URL segment
        label = " ".join(w.capitalize() for w in seg.split("-") if w)
        if label and label not in phrases:
            phrases.insert(0, label)
        seen: set[str] = set()
        out: list[str] = []
        for p in sorted(phrases, key=len, reverse=True):
            k = p.casefold()
            if k not in seen:
                seen.add(k)
                out.append(p)
        return out[:10]
    label = " ".join(w.capitalize() for w in seg.split("-") if w)
    return [label] if label else []


def fee_snippet_anchor_phrases(query: str) -> list[str]:
    """
    Phrases to center fee-related snippets on for broad queries where target text
    can sit far from chunk starts (e.g., scholarship sections on tuition pages).
    """
    q = (query or "").lower()
    if not q or not fee_tuition_intent(q):
        return []
    phrases: list[str] = []
    if re.search(r"\bscholar(ship|ships)?\b|\bburs(lar[ıi]?)?\b", q):
        phrases.extend(
            [
                "Scholarship",
                "Scholarships",
                "Tuition Fees and Scholarships",
                "Burs",
                "Burslar",
                "Ücret ve Burs",
                "Öğrenim Ücretleri ve Burslar",
            ]
        )
    if re.search(r"\bpayment|ödeme|odeme|installment|taksit\b", q):
        phrases.extend(["Payment", "Ödeme", "Taksit", "Payment Plan"])
    seen: set[str] = set()
    out: list[str] = []
    for p in phrases:
        k = p.casefold()
        if k not in seen:
            seen.add(k)
            out.append(p)
    return out


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

# User explicitly asked about master’s/PhD/graduate — do not default to UG intl. admission rules.
RAG_GRADUATE_ADMISSIONS_INTENT_RE = re.compile(
    r"\b(graduate|post-?grad|postgraduate|master[’']?s\b|\bmba\b|m\.?\s*sc\.?\b|m\.?a\.?\b|"
    r"ph\.?d|doktora|tez|yüksek\s+lisans|yuksek\s+lisans|lisansüstü|lisansustu|"
    r"doctoral|doctorate|graduate\s+school|graduate\s+program)\b|"
    r"\b(masters?|ms\b|phd|graduate|doktora|yüksek)\b.*\b(apply|admission|applicant|requirement)\b|"
    r"\b(apply|admission|applicant|requirement|international)\b.*\b("
    r"masters?|phd|graduate|doktora|master[’']s|yüksek|postgrad|post-?grad"
    r")\b",
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


def graduate_or_postgrad_admissions_intent(text: str) -> bool:
    """User is asking about graduate / master’s / PhD admissions (not the default UG intl. case)."""
    return bool(RAG_GRADUATE_ADMISSIONS_INTENT_RE.search(text or ""))


def international_admissions_default_undergraduate_only(text: str) -> bool:
    """
    International admission/requirements question without explicit master’s/PhD/graduate focus.
    In this mode, do not treat graduate-program requirements as applying to the user’s question.
    """
    t = (text or "").strip()
    if not t or graduate_or_postgrad_admissions_intent(t):
        return False
    if international_application_requirements_page_intent(t):
        return True
    return bool(international_student_apply_intent(t))


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


def international_student_apply_intent(text: str) -> bool:
    """
    International/foreign-student questions about: applying, admissions, requirements,
    or eligibility (not “international student tuition / fees” alone).
    """
    t = (text or "").strip()
    if not t:
        return False
    tl = t.lower()
    intl = bool(
        re.search(
            r"\binternational\s+(students?|applicants?|applicant)\b|"
            r"\bforeign\s+(students?|applicants?)\b|\boverseas\s+students?|"
            r"yabanc[ıi]\s*ö[ğg]renci|yabanci\s*ogrenci",
            tl,
        )
    ) or re.search(
        r"\b(international|foreign|yabanc[ıi]|overseas)\b.+\b("
        r"apply|application|admissions?|admission|başvuru|basvuru|"
        r"requirement|requirements|eligibility|criteria"
        r")\b|"
        r"\b(apply|application|admissions?|admission|başvuru|basvuru|"
        r"requirement|requirements|eligibility|criteria)\b.+\b("
        r"international|foreign|yabanc[ıi]|overseas"
        r")\b",
        tl,
    )
    topic = bool(
        re.search(
            r"\b(apply|application|admissions?|admission|enroll|başvuru|basvuru|kabul|"
            r"eligible|eligibility|requirement|requirements|criteria|qualif|prerequisite|documents?)\b|"
            r"what\s+are\s+the\s+requirements|what\s+do\s+(i|we)\s+need|"
            r"who\s+can\s+apply|who\s+cannot\s+apply",
            tl,
        )
    ) or re.search(
        r"can\s+(i|we|they|you)\b.*\bapply\b|"
        r"can\s+international\s+students?\b.*\bapply|"
        r"\b(international|foreign)\b.*\bapply\??$",
        tl,
    )
    if not (intl and topic):
        return False
    # Pure fee/price without an apply/admission angle — not this intent
    if re.search(
        r"\b(tuition|ücret|fee|fees|how\s+much|price|prices|cost|payment|ödeme|odeme)\b",
        tl,
    ) and not re.search(
        r"\b(apply|application|admissions?|admission|başvuru|basvuru|kabul|eligible|enroll|"
        r"requirement|requirements|eligibility|criteria)\b",
        tl,
    ):
        return False
    return True


def international_application_requirements_page_intent(text: str) -> bool:
    """
    User asks for concrete international admission / application requirements
    (diplomas, exams, scores — the main table on the Application Requirements page).
    """
    if not international_student_apply_intent(text):
        return False
    tl = (text or "").lower()
    return bool(
        re.search(
            r"\b(requirement|requirements|diploma|diplomas|exams?|entrance|tests?|scores?|"
            r"sat\b|gce|act\b|\bap\b|tr-y|yös|yos|tawjihi|baccalaur|abitur|matura|"
            r"transcripts?|national\s+high\s+school|how\s+to\s+get\s+in)\b|"
            r"what\s+are\s+the\s+requirements|what\s+do\s+(i|we)\s+need",
            tl,
        )
    )


def international_admissions_embedding_phrase(query: str) -> str | None:
    if not (query or "").strip():
        return None
    if not international_student_apply_intent(query):
        return None
    if international_application_requirements_page_intent(query):
        return (
            "Acıbadem University international students undergraduate Application Requirements "
            "required diploma exam table SAT GCE ACT AP IB National High School TR-YÖS School of Medicine English Turkish program minimum scores"
        )
    return (
        "Acıbadem Mehmet Ali Aydınlar University international students admission application "
        "how to apply requirements deadlines English language proficiency yabancı öğrenci başvuru kabul"
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
    if re.search(r"computer\s+programming|bilgisayar\s+programc", ql):
        out.extend(
            [
                "Computer Programming",
                "Bilgisayar Programcılığı",
                "associate degree",
                "ön lisans",
            ]
        )
        seen_prog: set[str] = set()
        deduped_prog: list[str] = []
        for t in out:
            if t.lower() not in seen_prog:
                seen_prog.add(t.lower())
                deduped_prog.append(t)
        return deduped_prog
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
