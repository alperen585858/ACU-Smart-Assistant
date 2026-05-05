"""
Vector RAG retrieval: multi-query embeddings, merged distances, wide candidate pool,
rerank (word overlap + optional pg_trgm + cross-encoder), then char-budget fill.
"""

from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass, field
from urllib.parse import parse_qs, urlparse

from django.contrib.postgres.search import SearchQuery, SearchRank, SearchVector
from django.db import connection
from django.db.models import Q
from pgvector.django import CosineDistance

from core.embeddings import embed_texts
from core.models import DocumentChunk, Page
from core.rag_config import (
    RAG_ACADEMIC_OBS_FALLBACK_LIMIT,
    RAG_BM25_HYBRID,
    RAG_CROSS_ENCODER_RERANK,
    RAG_CROSS_ENCODER_WEIGHT,
    RAG_KEYWORD_BOOST,
    RAG_LEXICAL_WEIGHT,
    RAG_MAX_CHARS,
    RAG_MAX_CHUNKS_PER_URL,
    RAG_MAX_DISTANCE,
    RAG_MULTI_EMBED,
    RAG_MULTI_EMBED_KEYWORD_LINE,
    RAG_OBS_VECTOR_PREFILTER,
    RAG_RELAX_ON_EMPTY,
    RAG_RERANK_OVERLAP_WEIGHT,
    RAG_SNIPPET_CHARS,
    RAG_STEM_NOISE_URL_PENALTY,
    RAG_TOP_K,
    RAG_VECTOR_CANDIDATE_POOL,
    RAG_VECTOR_FILL_EXTRA,
    RAG_WHOIS_EXTRA_EMBEDS,
    RAG_WHOIS_QUERY_EXPAND,
    is_rag_source_url_blocked,
    rag_source_url_blocklist_substrings,
)
from core.rag_keywords import (
    RAG_ACADEMIC_OBS_INTENT_RE,
    RAG_FACULTY_ROSTER_INTENT_RE,
    RAG_LEADERSHIP_INTENT_RE,
    RAG_LOCATION_CONTACT_INTENT_RE,
    RAG_STEM_OR_ENGINEERING_INTENT_RE,
    department_snippet_anchor_phrases,
    extract_target_entity_key,
    fee_snippet_anchor_phrases,
    faculty_list_embedding_phrase,
    faculty_roster_path_filter,
    fee_tuition_intent,
    international_admissions_default_undergraduate_only,
    international_admissions_embedding_phrase,
    international_application_requirements_page_intent,
    international_student_apply_intent,
    is_university_wide_fee_rag_query,
    leadership_embedding_phrase,
    rag_keywords_from_query,
    stem_engineering_boost_terms,
    structured_list_boost_terms,
    target_entity_aliases,
    target_entity_competitor_aliases,
)
from core.rag_query_expand import (
    snippet_around_phrase,
    whois_name_in_content,
    whois_name_from_queries,
    whois_vector_variants,
)

logger = logging.getLogger("core.rag")


def _load_whois_name_chunks(anchor: str, out_limit: int, scan_limit: int = 200) -> list[DocumentChunk]:
    """
    Rows mentioning this person: DB prefilter + whois_name_in_content (Turkish/Latin fold).

    Important: for multi-token names we require *all* tokens in SQL (AND), not only the
    first name. Short first tokens like "Ata" match huge unrelated sets; scanning the first
    N rows by pk often misses the real faculty chunk entirely.
    """
    a = (anchor or "").strip()
    if not a or out_limit < 1:
        return []
    parts = [p for p in a.split() if len(p) >= 2]
    if not parts:
        q = DocumentChunk.objects.filter(content__icontains=a)[:out_limit]
        return [
            ch
            for ch in q
            if not is_rag_source_url_blocked(str(ch.source_url or ""))
            and whois_name_in_content(str(ch.content or ""), anchor)
        ]

    out: list[DocumentChunk] = []
    seen: set[int] = set()
    wide = max(scan_limit, min(1200, scan_limit * 6))

    def try_append(ch: DocumentChunk) -> None:
        if ch.pk in seen:
            return
        if is_rag_source_url_blocked(str(ch.source_url or "")):
            return
        if not whois_name_in_content(str(ch.content or ""), anchor):
            return
        seen.add(ch.pk)
        out.append(ch)

    if len(parts) >= 2:
        q_all = Q(content__icontains=parts[0])
        for p in parts[1:]:
            q_all &= Q(content__icontains=p)
        for ch in DocumentChunk.objects.filter(q_all).order_by("pk")[:wide]:
            if len(out) >= out_limit:
                break
            try_append(ch)
        if out:
            return out
        for ch in (
            DocumentChunk.objects.filter(content__icontains=parts[-1])
            .order_by("pk")[:wide]
        ):
            if len(out) >= out_limit:
                break
            try_append(ch)
        if out:
            return out

    for ch in (
        DocumentChunk.objects.filter(content__icontains=parts[0]).order_by("pk")[:scan_limit]
    ):
        if len(out) >= out_limit:
            break
        try_append(ch)
    return out


_pg_trgm_cache: bool | None = None


def _pg_trgm_available() -> bool:
    global _pg_trgm_cache
    if _pg_trgm_cache is not None:
        return _pg_trgm_cache
    if connection.vendor != "postgresql":
        _pg_trgm_cache = False
        return False
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'pg_trgm')"
            )
            row = cursor.fetchone()
            _pg_trgm_cache = bool(row and row[0])
    except Exception:
        _pg_trgm_cache = False
    return _pg_trgm_cache


def _query_token_set(composed: str, raw_user: str | None) -> set[str]:
    blob = f"{composed} {raw_user or ''}".lower()
    return set(re.findall(r"[a-zA-ZğüşıöçĞÜŞİÖÇ]{4,}", blob))


def _word_overlap_count(chunk_text: str, tokens: set[str]) -> int:
    if not tokens or not chunk_text:
        return 0
    low = chunk_text.lower()
    return sum(1 for t in tokens if t in low)


_DEPT_LEADER_QUERY_RE = re.compile(
    r"head|chair|director|dean|başkan|baskan|dekan|müdür|mudur",
    re.IGNORECASE,
)
_DEPT_LEADER_CONTENT_RE = re.compile(
    r"head of department|message from head|department head|chair of|"
    r"bölüm başkanı|bolum baskan|dekan|müdür",
    re.IGNORECASE,
)
_LEADERSHIP_CONTENT_RE = re.compile(
    r"dean\s+of|faculty\s+dean|\bdean\b|dekan|rector|rektör|dean'?s?\s+office|dekanlık|"
    r"vice\s*rector|yardımcı\s*rektör|fakülte|faculty\s+of",
    re.IGNORECASE,
)
# "Who is X?" — commissions / quota announcements often mention names in passing; deprioritize
# vs academic-staff / message-from-head / department about pages (high-signal identity sources).
_WHOIS_HIGH_SIGNAL_RE = re.compile(
    r"academic-staff|message-from-head|head-of-department|/head-of-department|"
    r"message\s+from\s+head|from\s+the\s+head|ode-to-the-department|"
    r"bolum-baskan|bölüm\s*başkan|department\s+head\s+message",
    re.IGNORECASE,
)
_WHOIS_LOW_SIGNAL_RE = re.compile(
    r"commissions?\b|board-?of-?education|board_of_education|education-and-commissions|"
    r"application[-\s]quotas|evaluation[-\s]schedule|quotas.*evaluation|"
    r"have\s+been\s+announced|"
    r"fall[-\s]semester.*20\d\d|academic[-\s]year.*announced|"
    r"graduate\s+school\s+of\s+natural.*announc|"
    r"alumni[-_\s]videos|double[-\s]major|minor[-\s]program|"
    r"accreditation[-\s]certificate|"
    r"klinik-arastirmalar|yo[ğg]un-bakim",
    re.IGNORECASE,
)


def _whois_high_low_signal(url: str, page_title: str) -> tuple[bool, bool]:
    blob = f"{url or ''}\n{page_title or ''}"
    return (
        bool(_WHOIS_HIGH_SIGNAL_RE.search(blob)),
        bool(_WHOIS_LOW_SIGNAL_RE.search(blob)),
    )


def _whois_chunk_allowed_for_identity(ch: DocumentChunk, anchor: str) -> bool:
    """
    For person-identity questions: drop commission/quota/announcement noise even when the
    name appears on a long roster; keep high-signal staff/head URLs; otherwise require the
    name in title or body so vector "near misses" (dept home, unrelated pages) are removed.
    """
    u = str(ch.source_url or "")
    t = str(ch.page_title or "")
    hi, lo = _whois_high_low_signal(u, t)
    if lo and not hi:
        return False
    if hi:
        return True
    blob = f"{t}\n{ch.content or ''}"
    return whois_name_in_content(blob, anchor)


def _filter_whois_identity_chunks(
    pairs: list[tuple[DocumentChunk, float]],
    anchor: str | None,
) -> list[tuple[DocumentChunk, float]]:
    if not anchor:
        return pairs
    return [(ch, d) for ch, d in pairs if _whois_chunk_allowed_for_identity(ch, anchor)]


_DEPT_NAMES: list[tuple[str, str]] = [
    ("computer engineering", "bilgisayar mühendisliği"),
    ("electrical", "elektrik"),
    ("electronics", "elektronik"),
    ("mechanical engineering", "makine mühendisliği"),
    ("civil engineering", "inşaat mühendisliği"),
    ("industrial engineering", "endüstri mühendisliği"),
    ("biomedical engineering", "biyomedikal mühendisliği"),
    ("software engineering", "yazılım mühendisliği"),
    ("medicine", "tıp fakültesi"),
    ("nursing", "hemşirelik"),
    ("pharmacy", "eczacılık"),
    ("dentistry", "diş hekimliği"),
    ("health sciences", "sağlık bilimleri"),
]


def _intent_boost(composed_query: str, url: str, title: str, content: str) -> int:
    q = (composed_query or "").lower()
    blob = f"{url or ''} {title or ''} {content or ''}".lower()
    boost = 0
    location_intent = bool(RAG_LOCATION_CONTACT_INTENT_RE.search(q))
    academic_obs_intent = bool(RAG_ACADEMIC_OBS_INTENT_RE.search(q))
    if RAG_LEADERSHIP_INTENT_RE.search(q):
        if _LEADERSHIP_CONTENT_RE.search(blob):
            boost += 14
    if _DEPT_LEADER_QUERY_RE.search(q):
        if _DEPT_LEADER_CONTENT_RE.search(blob):
            boost += 8
        for en_name, tr_name in _DEPT_NAMES:
            if en_name in q or tr_name in q:
                if en_name in blob or tr_name in blob:
                    boost += 5
                break
    if location_intent and _LOCATION_AUTH_HINT_RE.search(blob):
        boost += 10
    if location_intent and _OBS_URL_RE.search(blob):
        boost -= 8
    if academic_obs_intent and _OBS_URL_RE.search(blob):
        boost += 7
    return boost


def _lexical_fallback_from_chunks(
    composed_query: str,
    raw_user_query: str | None,
) -> tuple[str, list[dict], bool, bool]:
    """
    Best-effort fallback when embeddings are unavailable:
    score chunks by query token overlap and build a bounded context.
    """
    tokens = _query_token_set(composed_query, raw_user_query)
    if not tokens:
        return "", [], True, False

    max_candidates = max(RAG_TOP_K + RAG_VECTOR_FILL_EXTRA, 40) * 5
    candidates: list[tuple[int, DocumentChunk]] = []
    for ch in DocumentChunk.objects.only(
        "pk", "content", "page_title", "source_url"
    ).iterator(chunk_size=2000):
        score = _word_overlap_count(str(ch.content or ""), tokens)
        score += _word_overlap_count(str(ch.page_title or ""), tokens) * 2
        score += _word_overlap_count(str(ch.source_url or ""), tokens) * 2
        score += _intent_boost(
            composed_query,
            str(ch.source_url or ""),
            str(ch.page_title or ""),
            str(ch.content or ""),
        )
        if score > 0 and not is_rag_source_url_blocked(str(ch.source_url or "")):
            candidates.append((score, ch))
            if len(candidates) >= max_candidates:
                break
    if not candidates:
        return "", [], True, False

    candidates.sort(key=lambda x: (-x[0], x[1].pk))
    ranked = [ch for _, ch in candidates[: max(RAG_TOP_K + RAG_VECTOR_FILL_EXTRA, 40)]]

    context_parts: list[str] = []
    sources: list[dict] = []
    total = 0
    url_counts: dict[str, int] = {}

    # For head/chair/dean questions, prioritize exact department pages from Page rows.
    q = (composed_query or "").lower()
    if _DEPT_LEADER_QUERY_RE.search(q):
        url_keyword = None
        for en_name, tr_name in _DEPT_NAMES:
            if en_name in q or tr_name in q:
                url_keyword = en_name.split()[0]  # e.g. "computer", "electrical"
                break
        if url_keyword:
            page_hits: list[Page] = []
            for p in Page.objects.filter(url__icontains=url_keyword).iterator():
                blob = f"{str(p.url or '')} {str(p.content or '')}".lower()
                if _DEPT_LEADER_CONTENT_RE.search(blob):
                    page_hits.append(p)
                if len(page_hits) >= 3:
                    break
            for p in page_hits:
                if total >= RAG_MAX_CHARS:
                    break
                u = str(p.url or "")
                if is_rag_source_url_blocked(u):
                    continue
                if url_counts.get(u, 0) >= RAG_MAX_CHUNKS_PER_URL:
                    continue
                snippet = str(p.content or "")[:RAG_SNIPPET_CHARS]
                if not snippet or total + len(snippet) > RAG_MAX_CHARS:
                    continue
                title = str(p.title or p.url or "")
                context_parts.append(f"[{title}]\n{snippet}")
                total += len(snippet)
                url_counts[u] = url_counts.get(u, 0) + 1
                sources.append(
                    {
                        "url": u,
                        "title": title[:200],
                        "cosine_distance": 0.9997,
                    }
                )
    for ch in ranked:
        if total >= RAG_MAX_CHARS:
            break
        u = str(ch.source_url or "")
        if url_counts.get(u, 0) >= RAG_MAX_CHUNKS_PER_URL:
            continue
        snippet = str(ch.content or "")[:RAG_SNIPPET_CHARS]
        if not snippet:
            continue
        if total + len(snippet) > RAG_MAX_CHARS:
            continue
        title = str(ch.page_title or ch.source_url or "")
        context_parts.append(f"[{title}]\n{snippet}")
        total += len(snippet)
        url_counts[u] = url_counts.get(u, 0) + 1
        sources.append(
            {
                "url": ch.source_url,
                "title": (title or "")[:200],
                "cosine_distance": 0.9999,
            }
        )
    return "\n\n".join(context_parts), sources, True, False


def _lexical_fallback_from_pages(
    composed_query: str,
    raw_user_query: str | None,
) -> tuple[str, list[dict], bool, bool]:
    """
    Second fallback when chunk embeddings were never built for some pages.
    """
    tokens = _query_token_set(composed_query, raw_user_query)
    if not tokens:
        return "", [], True, False

    max_candidates = max(RAG_TOP_K, 12) * 5
    candidates: list[tuple[int, Page]] = []
    for p in Page.objects.only("id", "url", "title", "content").iterator(chunk_size=2000):
        blob = f"{str(p.title or '')}\n{str(p.content or '')}"
        score = _word_overlap_count(blob, tokens)
        score += _word_overlap_count(str(p.url or ""), tokens) * 2
        score += _intent_boost(
            composed_query, str(p.url or ""), str(p.title or ""), str(p.content or "")
        )
        purl = str(p.url or "")
        if score > 0 and not is_rag_source_url_blocked(purl):
            candidates.append((score, p))
            if len(candidates) >= max_candidates:
                break
    if not candidates:
        return "", [], True, False

    candidates.sort(key=lambda x: (-x[0], x[1].id))
    context_parts: list[str] = []
    sources: list[dict] = []
    total = 0
    for _, p in candidates[: max(RAG_TOP_K, 12)]:
        if total >= RAG_MAX_CHARS:
            break
        snippet = str(p.content or "")[:RAG_SNIPPET_CHARS]
        if not snippet:
            continue
        if total + len(snippet) > RAG_MAX_CHARS:
            continue
        title = str(p.title or p.url or "")
        context_parts.append(f"[{title}]\n{snippet}")
        total += len(snippet)
        sources.append(
            {
                "url": p.url,
                "title": (title or "")[:200],
                "cosine_distance": 1.0,
            }
        )
    return "\n\n".join(context_parts), sources, True, False


def _embedding_variants(composed: str, raw_user: str | None) -> list[str]:
    composed = (composed or "").strip()
    variants: list[str] = []
    if composed:
        variants.append(composed)
    if RAG_MULTI_EMBED:
        raw = (raw_user or "").strip()
        if raw and raw.casefold() != composed.casefold():
            variants.append(raw)
        if RAG_MULTI_EMBED_KEYWORD_LINE:
            kw = rag_keywords_from_query(composed)
            if kw:
                line = " ".join(kw) + " Acibadem Mehmet Ali Aydinlar University"
                if line.casefold() not in {v.casefold() for v in variants}:
                    variants.append(line)
    # "Who is X?" / "X kimdir" — extra vectors to match faculty/head pages (independent of RAG_MULTI_EMBED)
    if RAG_WHOIS_QUERY_EXPAND:
        wn = whois_name_from_queries(composed, raw_user)
        if wn:
            for line in whois_vector_variants(wn, RAG_WHOIS_EXTRA_EMBEDS):
                if line.casefold() not in {v.casefold() for v in variants}:
                    variants.append(line)
    # Department faculty roster: steer embeddings toward .../academic-staff/ (not /about, /news, …).
    fl_blob = f"{composed} {raw_user or ''}".strip()
    fl_phrase = faculty_list_embedding_phrase(fl_blob)
    if fl_phrase and fl_phrase.casefold() not in {v.casefold() for v in variants}:
        variants.append(fl_phrase)
    lead_phrase = leadership_embedding_phrase(fl_blob)
    if lead_phrase and lead_phrase.casefold() not in {v.casefold() for v in variants}:
        variants.append(lead_phrase)
    intl_phrase = international_admissions_embedding_phrase(fl_blob)
    if intl_phrase and intl_phrase.casefold() not in {v.casefold() for v in variants}:
        variants.append(intl_phrase)
    # English "medicine" + fees → Tıp Fakültesi / MD, not only MYO rows on a generic fee page
    if fee_tuition_intent(fl_blob) and faculty_roster_path_filter(fl_blob) == "faculty-of-medicine":
        med_line = (
            "Acıbadem University Faculty of Medicine Tıp Fakültesi lisans six year MD hekimlik "
            "undergraduate tuition not Medical Education master program not tıp eğitimi yüksek lisans"
        )
        if med_line.casefold() not in {v.casefold() for v in variants}:
            variants.append(med_line)
    if (
        fee_tuition_intent(fl_blob)
        and faculty_roster_path_filter(fl_blob) == "faculty-of-health-sciences"
    ):
        hs_line = (
            "Acıbadem University Faculty of Health Sciences Sağlık Bilimleri "
            "Physiotherapy Nursing Nutrition Dietetics Healthcare Management "
            "undergraduate program tuition fee per year USD not Medicine not vocational school"
        )
        if hs_line.casefold() not in {v.casefold() for v in variants}:
            variants.append(hs_line)
    # All-program tuition pages (not a single /computer-engineering/ path).
    if is_university_wide_fee_rag_query(fl_blob):
        fee_all = (
            "Acıbadem University full tuition and fee schedule all faculties programs "
            "undergraduate graduate Medicine Engineering Health Law Dentistry Pharmacy "
            "vocational school associate degree price list"
        )
        if fee_all.casefold() not in {v.casefold() for v in variants}:
            variants.append(fee_all)
    # OBS Computer Engineering: steer vector search toward Bologna course/module vocabulary.
    if (
        RAG_ACADEMIC_OBS_INTENT_RE.search(fl_blob)
        and extract_target_entity_key(fl_blob) == "computer-engineering"
    ):
        obs_ce = (
            "Acıbadem University Computer Engineering undergraduate Bologna programme "
            "course structure curriculum semester modules ECTS course codes syllabus "
            "programme learning outcomes degree requirements"
        )
        if obs_ce.casefold() not in {v.casefold() for v in variants}:
            variants.append(obs_ce)
    # Dedupe preserving order
    seen: set[str] = set()
    out: list[str] = []
    for v in variants:
        key = v.casefold()
        if key not in seen:
            seen.add(key)
            out.append(v)
    return out


def _bm25_search(query: str, top_n: int = 20) -> list[tuple[int, float]]:
    """PostgreSQL full-text search (BM25-like ranking) on chunk content + title."""
    if not query or not query.strip():
        return []
    try:
        words = re.findall(r"[a-zA-ZğüşıöçĞÜŞİÖÇ]{3,}", query)
        if not words:
            return []
        search_str = " | ".join(words[:10])
        sq = SearchQuery(search_str, search_type="raw")
        sv = SearchVector("content", weight="A") + SearchVector("page_title", weight="B")
        results = (
            DocumentChunk.objects.annotate(rank=SearchRank(sv, sq))
            .filter(rank__gt=0.01)
            .order_by("-rank")[:top_n]
        )
        return [(ch.pk, float(ch.rank)) for ch in results]
    except Exception:
        logger.debug("BM25 search failed, skipping", exc_info=True)
        return []


def _merge_best_distances(vectors: list[list[float]], per_vector_pool: int) -> dict[int, float]:
    best: dict[int, float] = {}
    for vec in vectors:
        if not vec:
            continue
        qs = (
            DocumentChunk.objects.annotate(distance=CosineDistance("embedding", vec))
            .order_by("distance")[:per_vector_pool]
        )
        for ch in qs:
            if is_rag_source_url_blocked(str(ch.source_url or "")):
                continue
            d = float(ch.distance)
            pk = ch.pk
            prev = best.get(pk)
            if prev is None or d < prev:
                best[pk] = d
    return best


def _merge_obs_host_vector_pool(primary_vector: list[float], limit: int) -> dict[int, float]:
    """Top cosine-distance hits restricted to obs.acibadem.edu.tr chunks."""
    if not primary_vector or limit < 1:
        return {}
    out: dict[int, float] = {}
    qs = (
        DocumentChunk.objects.filter(source_url__icontains="obs.acibadem.edu.tr")
        .annotate(distance=CosineDistance("embedding", primary_vector))
        .order_by("distance")[:limit]
    )
    for ch in qs:
        u = str(ch.source_url or "")
        if is_rag_source_url_blocked(u):
            continue
        out[ch.pk] = float(ch.distance)
    return out


# STEM: de-prioritize generic portal / events (not /fees/ or "tuition" in URL — that hurt fee questions).
_STEM_LOW_VALUE_PAGE_RE = re.compile(
    r"event|etkinlik|kariyer|career|duyur|announce|haber|news|kongre|congress|"
    r"life-sciences|mezun|alumni|ilan|job|kampus\s*gez|graduate-fair|e-bulten|ebulten|"
    r"tubitak|tübitak",
    re.IGNORECASE,
)

# Fee questions: a chunk is usable only if the URL or body clearly concerns fees/tuition/currency.
_FEE_PATH_OR_TITLE_HINT = re.compile(
    r"fee|ucret|ücret|tuition|ogrenim|öğrenim|burs|scholarship|financial|"
    r"pricing|kay[ıi]t.*ucret|kayıt.*ücret|ucreti|ucret-bilgi|academic-fee|"
    r"tuition-fee|ogrenim-ucreti|admissions.*fee|fees-and|fees-and-tuition|/fees/|/ucret",
    re.IGNORECASE,
)
_FEE_TEXT_EVIDENCE = re.compile(
    r"tuition|ücret|ucret|öğrenim|ogrenim|program\s+fee|annual\s+fee|"
    r"\busd\b|\btry\b|₺|\$\s*[\d,\.]+|vat|kdv|y[ıi]ll[ıi]k|taksit|per\s+year|/year|"
    r"scholarship|burs|financial\s+aid|ödeme|odeme|payment\s+plan|pe[şs]in|pesin",
    re.IGNORECASE,
)

_SCHOLARSHIP_QUERY_RE = re.compile(r"\bscholar(ship|ships)?\b|\bburs(lar[ıi]?)?\b", re.IGNORECASE)
_SCHOLARSHIP_TEXT_EVIDENCE = re.compile(
    r"\bscholar(ship|ships)?\b|\bburs(lar[ıi]?)?\b|discount|indirim|financial\s+aid",
    re.IGNORECASE,
)

# International UG default: demote / skip graduate-application sources when the user did not ask for graduate.
_GRADUATE_INTL_SOURCE_HINT = re.compile(
    r"graduate|post-?grad|yuksek\s*lisans|yüksek\s+lisans|master[’'s]?\s*program|"
    r"phd|doktora|/graduate/|/graduate-|\btez\b|mba\s*admission|post-?grad",
    re.IGNORECASE,
)

_ENTITY_NOISE_SOURCE_RE = re.compile(
    r"career|kariyer|event|etkinlik|news|haber|announcement|duyuru",
    re.IGNORECASE,
)

_OBS_URL_RE = re.compile(
    r"obs\.acibadem\.edu\.tr|/oibs/|bologna|dynconpage|course",
    re.IGNORECASE,
)

_OBS_PROGRAM_DETAIL_RE = re.compile(
    r"showpac|curop=showpac|cursunit=\d+|curunit=\d+",
    re.IGNORECASE,
)

_CE_CURRICULUM_QUERY_RE = re.compile(
    r"\blessons?\b|\bcourses?\b|curriculum|modules?|\bects\b|syllabus|müfredat|ders",
    re.IGNORECASE,
)

# Entity-specific OBS program code hints.
# These codes are stable enough to de-prioritize clearly wrong program links.
# curUnit=14 Computer Engineering uses several curSunit shells (overview vs sections).
# Live URLs look like: .../bologna/index.aspx?lang=en&curOp=showPac&curUnit=14&curSunit=6246
_OBS_ENTITY_CODE_HINTS: dict[str, dict[str, set[str]]] = {
    "computer-engineering": {
        "curunit": {"14"},
        "cursunit": {"6246", "6247", "6248"},
    },
}


def _obs_entity_source_url_q(hints: dict[str, set[str]] | None) -> Q | None:
    """
    OR of URL substring filters for known OBS curUnit / curSunit params.
    Matches index.aspx?...&curOp=showPac&curUnit=14&... style URLs.
    """
    if not hints:
        return None
    parts: list[Q] = []
    for u in hints.get("curunit") or ():
        su = str(u).strip()
        if not su:
            continue
        parts.append(Q(source_url__icontains=f"curUnit={su}"))
        parts.append(Q(source_url__icontains=f"curunit={su}"))
    for s in hints.get("cursunit") or ():
        ss = str(s).strip()
        if not ss:
            continue
        parts.append(Q(source_url__icontains=f"curSunit={ss}"))
        parts.append(Q(source_url__icontains=f"cursunit={ss}"))
    if not parts:
        return None
    combined = parts[0]
    for p in parts[1:]:
        combined |= p
    return combined


_LOCATION_AUTH_HINT_RE = re.compile(
    r"contact|iletisim|iletişim|address|adres|transport|ula[şs][ıi]m|campus|konum|location",
    re.IGNORECASE,
)


def _chunk_bears_fee_grounding(ch: DocumentChunk) -> bool:
    u = f"{ch.source_url or ''} {ch.page_title or ''}"
    if _FEE_PATH_OR_TITLE_HINT.search(u):
        return True
    blob = f"{ch.page_title or ''}\n{ch.content or ''}"[:80000]
    return bool(_FEE_TEXT_EVIDENCE.search(blob))


def _chunk_bears_scholarship_grounding(ch: DocumentChunk) -> bool:
    blob = f"{ch.page_title or ''}\n{ch.content or ''}"[:80000]
    return bool(_SCHOLARSHIP_TEXT_EVIDENCE.search(blob))


def _entity_alignment_score(
    text: str, target_aliases: tuple[str, ...], competitor_aliases: tuple[str, ...]
) -> tuple[int, int]:
    low = (text or "").lower()
    pos = sum(1 for a in target_aliases if a and a.lower() in low)
    neg = sum(1 for a in competitor_aliases if a and a.lower() in low)
    return pos, neg


def _stem_noise_penalty(url: str, title: str) -> float:
    if RAG_STEM_NOISE_URL_PENALTY <= 0:
        return 0.0
    blob = f"{url or ''} {title or ''}"
    if _STEM_LOW_VALUE_PAGE_RE.search(blob):
        return RAG_STEM_NOISE_URL_PENALTY
    return 0.0


def _obs_priority_adjustment(url: str, title: str, *, location_intent: bool, academic_obs_intent: bool) -> float:
    """
    Positive => penalize (demote), negative => promote.
    OBS remains available globally; only priority changes by intent.
    """
    blob = f"{url or ''} {title or ''}"
    if not _OBS_URL_RE.search(blob):
        return 0.0
    if location_intent:
        return 0.22
    if academic_obs_intent:
        # For curriculum questions, reward program-detail OBS pages more than
        # generic index/info pages.
        if _OBS_PROGRAM_DETAIL_RE.search(blob):
            return -0.16
        return -0.10
    return 0.03


def _obs_entity_code_adjustment(url: str, intents: QueryIntents) -> float:
    """
    OBS program-detail URL tuning by target entity code hints.
    Negative => promote, positive => demote.
    """
    if not intents.academic_obs or not intents.target_entity:
        return 0.0
    if not _OBS_PROGRAM_DETAIL_RE.search(url or ""):
        return 0.0
    hints = _OBS_ENTITY_CODE_HINTS.get(intents.target_entity)
    if not hints:
        return 0.0
    parsed = urlparse(str(url or ""))
    q = parse_qs(parsed.query or "")
    curunit = (q.get("curUnit") or q.get("curunit") or [""])[0].strip()
    cursunit = (q.get("curSunit") or q.get("cursunit") or [""])[0].strip()

    score = 0.0
    unit_hints = hints.get("curunit") or set()
    sunit_hints = hints.get("cursunit") or set()

    if curunit:
        score += -0.20 if curunit in unit_hints else 0.50
    if cursunit:
        score += -0.14 if cursunit in sunit_hints else 0.34
    return score


def _obs_url_matches_target_entity(url: str, intents: QueryIntents) -> bool:
    """
    True when OBS URL query params match known target entity codes.
    If no hints exist for that entity, return False.
    """
    if not intents.target_entity:
        return False
    hints = _OBS_ENTITY_CODE_HINTS.get(intents.target_entity)
    if not hints:
        return False
    parsed = urlparse(str(url or ""))
    q = parse_qs(parsed.query or "")
    curunit = (q.get("curUnit") or q.get("curunit") or [""])[0].strip()
    cursunit = (q.get("curSunit") or q.get("cursunit") or [""])[0].strip()
    unit_hints = hints.get("curunit") or set()
    sunit_hints = hints.get("cursunit") or set()
    return bool(
        (curunit and curunit in unit_hints)
        or (cursunit and cursunit in sunit_hints)
    )


def _obs_url_entity_match_state(url: str, intents: QueryIntents) -> bool | None:
    """
    Tri-state entity match for OBS URLs:
    - True  : URL matches known entity code hints
    - False : URL has comparable codes but mismatches hints
    - None  : no reliable hint or no comparable URL params
    """
    if not intents.target_entity:
        return None
    hints = _OBS_ENTITY_CODE_HINTS.get(intents.target_entity)
    if not hints:
        return None
    parsed = urlparse(str(url or ""))
    q = parse_qs(parsed.query or "")
    curunit = (q.get("curUnit") or q.get("curunit") or [""])[0].strip()
    cursunit = (q.get("curSunit") or q.get("cursunit") or [""])[0].strip()
    if not curunit and not cursunit:
        return None
    return _obs_url_matches_target_entity(url, intents)


def _academic_obs_fallback_chunks(
    intents: QueryIntents, limit: int | None = None,
) -> list[DocumentChunk]:
    """
    Last-resort OBS fallback for curriculum queries when ranking yields no usable context.
    When ``_OBS_ENTITY_CODE_HINTS`` has an entry for ``intents.target_entity``, narrows the
    queryset with curUnit/curSunit URL filters (plus dynConPage bodies) instead of relying
    on primary-key iteration order.
    """
    if not intents.academic_obs:
        return []

    lim = max(1, min(24, int(limit if limit is not None else RAG_ACADEMIC_OBS_FALLBACK_LIMIT)))

    base_obs = Q(source_url__icontains="obs.acibadem.edu.tr")
    # Programme shells use curOp=showPac on index.aspx; substring match covers both.
    showpac_q = Q(source_url__icontains="showPac")
    dyn_q = Q(source_url__icontains="dynConPage.aspx")

    hints_dict: dict[str, set[str]] | None = None
    if intents.target_entity:
        hints_dict = _OBS_ENTITY_CODE_HINTS.get(intents.target_entity)
    entity_q = _obs_entity_source_url_q(hints_dict)

    candidates: list[DocumentChunk] = []
    if entity_q is not None:
        scoped = (
            DocumentChunk.objects.filter(base_obs & (showpac_q | dyn_q) & entity_q)
            .only("pk", "content", "page_title", "source_url")
            .order_by("pk")[: max(lim * 12, 120)]
        )
        candidates = list(scoped)
    if not candidates:
        candidates = list(
            DocumentChunk.objects.filter(base_obs & showpac_q)
            .only("pk", "content", "page_title", "source_url")
            .order_by("pk")[: max(lim * 12, 120)]
        )

    def _sort_key(ch: DocumentChunk) -> tuple[int, int, int]:
        u = str(ch.source_url or "")
        blob = f"{ch.page_title or ''}\n{ch.content or ''}\n{u}".lower()
        ent_match = (
            1
            if intents.target_entity
            and hints_dict
            and _obs_url_matches_target_entity(u, intents)
            else 0
        )
        alias_match = (
            1
            if intents.target_alias
            and any(a and a.lower() in blob for a in intents.target_alias)
            else 0
        )
        # Lower tuple sorts first: prefer entity URL match, then alias in body, stable pk.
        return (-ent_match, -alias_match, ch.pk)

    candidates.sort(key=_sort_key)

    out: list[DocumentChunk] = []
    for ch in candidates:
        u = str(ch.source_url or "")
        if is_rag_source_url_blocked(u):
            continue
        if intents.target_alias:
            blob = f"{ch.page_title or ''}\n{ch.content or ''}\n{u}".lower()
            has_alias = any(a and a.lower() in blob for a in intents.target_alias)
            match_state = _obs_url_entity_match_state(u, intents)
            if match_state is False and not has_alias:
                continue
        out.append(ch)
        if len(out) >= lim:
            break
    return out


def _rerank_items(
    items: list[tuple[DocumentChunk, float]],
    composed: str,
    raw_user: str | None,
    stem_query: bool = False,
) -> list[tuple[DocumentChunk, float]]:
    if len(items) <= 1:
        return items
    tokens = _query_token_set(composed, raw_user)
    trigram_q = f"{composed} {(raw_user or '').strip()}".strip()[:400]

    sim_map: dict[int, float] = {}
    if (
        RAG_LEXICAL_WEIGHT > 0
        and trigram_q
        and _pg_trgm_available()
    ):
        from django.contrib.postgres.search import TrigramSimilarity

        pks = [ch.pk for ch, _ in items]
        for ch in (
            DocumentChunk.objects.filter(pk__in=pks)
            .annotate(sim=TrigramSimilarity("content", trigram_q))
            .iterator()
        ):
            sim_map[ch.pk] = float(getattr(ch, "sim", 0.0) or 0.0)

    # Cross-encoder reranking (only top candidates to keep latency low)
    ce_score_map: dict[int, float] = {}
    CE_TOP_N = 15
    if RAG_CROSS_ENCODER_RERANK:
        from core.embeddings import rerank_passages
        raw_q = (raw_user or composed or "").strip()[:300]
        ce_items = items[:CE_TOP_N]
        passages = [str(ch.content or "")[:500] for ch, _ in ce_items]
        t_ce = time.time()
        ce_scores = rerank_passages(raw_q, passages)
        logger.info("RAG cross-encoder rerank: %.2fs (%d items)", time.time() - t_ce, len(ce_items))
        if ce_scores:
            max_s = max(abs(s) for s in ce_scores) or 1.0
            for i, (ch, _) in enumerate(ce_items):
                ce_score_map[ch.pk] = ce_scores[i] / max_s

    def sort_key(it: tuple[DocumentChunk, float]) -> float:
        ch, d = it
        ov = _word_overlap_count(str(ch.content or ""), tokens)
        sim = sim_map.get(ch.pk, 0.0)
        ce = ce_score_map.get(ch.pk, 0.0)
        # Lower is better: pull down score when overlap/sim/cross-encoder is high
        key = (
            d
            - RAG_RERANK_OVERLAP_WEIGHT * min(ov, 30) / 30.0
            - RAG_LEXICAL_WEIGHT * sim
            - RAG_CROSS_ENCODER_WEIGHT * ce
        )
        if stem_query:
            key += _stem_noise_penalty(
                str(ch.source_url or ""), str(ch.page_title or "")
            )
        return key

    return sorted(items, key=sort_key)


def _cosine_distance_by_pk(
    pks: list[int], primary_vector: list[float]
) -> dict[int, float]:
    """Real cosine distance for metadata (avoids misleading constant 'boost' distances)."""
    if not pks or not primary_vector:
        return {}
    out: dict[int, float] = {}
    for ch in DocumentChunk.objects.filter(pk__in=pks).annotate(
        _cd=CosineDistance("embedding", primary_vector)
    ).iterator():
        out[ch.pk] = float(ch._cd)
    return out

# ──────────────────────────────────────────────────────────────────────────────
# Structured types for pipeline stages
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class QueryIntents:
    """Parsed intent signals extracted from the user query."""

    q_blob: str
    fee: bool
    scholarship: bool
    intl_apply: bool
    appreq: bool
    intl_ug_only: bool
    location: bool
    academic_obs: bool
    stem: bool
    apply_stem_noise: bool
    leadership: bool
    faculty_roster: bool
    whois_anchor: str | None
    target_entity: str
    target_alias: tuple[str, ...]
    competitor_alias: tuple[str, ...]


@dataclass
class IntentPageBlocks:
    """Full-page content blocks fetched for intent-specific injection."""

    inject_block: str = ""
    inject_skip_url: str | None = None
    inject_title: str = ""
    faculty_path: str | None = None
    mgmt_block: str = ""
    mgmt_url: str = ""
    mgmt_title: str = ""
    loc_block: str = ""
    loc_sources: list[dict] = field(default_factory=list)
    intl_block: str = ""
    intl_sources: list[dict] = field(default_factory=list)
    appreq_block: str = ""
    appreq_sources: list[dict] = field(default_factory=list)
    appreq_skip: set[str] = field(default_factory=set)
    scholarship_block: str = ""
    scholarship_sources: list[dict] = field(default_factory=list)
    scholarship_skip: set[str] = field(default_factory=set)
    whois_block: str = ""
    whois_sources: list[dict] = field(default_factory=list)


# ──────────────────────────────────────────────────────────────────────────────
# Stage 1 — Intent detection
# ──────────────────────────────────────────────────────────────────────────────


def _detect_query_intents(
    composed_query: str, raw_user_query: str | None
) -> QueryIntents:
    """Extract all intent signals from the user query."""
    q_blob = f"{composed_query} {raw_user_query or ''}".strip()
    fee = fee_tuition_intent(q_blob)
    stem = bool(RAG_STEM_OR_ENGINEERING_INTENT_RE.search(q_blob))
    whois_anchor: str | None = None
    if RAG_WHOIS_QUERY_EXPAND:
        wn = whois_name_from_queries(composed_query, raw_user_query)
        if wn and len(wn) >= 8:
            whois_anchor = wn
    target_entity = extract_target_entity_key(q_blob)
    return QueryIntents(
        q_blob=q_blob,
        fee=fee,
        scholarship=bool(_SCHOLARSHIP_QUERY_RE.search(q_blob)),
        intl_apply=international_student_apply_intent(q_blob),
        appreq=international_application_requirements_page_intent(q_blob),
        intl_ug_only=international_admissions_default_undergraduate_only(q_blob),
        location=bool(RAG_LOCATION_CONTACT_INTENT_RE.search(q_blob)),
        academic_obs=bool(RAG_ACADEMIC_OBS_INTENT_RE.search(q_blob)),
        stem=stem,
        apply_stem_noise=stem and not fee,
        leadership=bool(RAG_LEADERSHIP_INTENT_RE.search(q_blob)),
        faculty_roster=bool(RAG_FACULTY_ROSTER_INTENT_RE.search(q_blob)),
        whois_anchor=whois_anchor,
        target_entity=target_entity,
        target_alias=target_entity_aliases(target_entity),
        competitor_alias=target_entity_competitor_aliases(target_entity),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Stage 2 — Vector search + BM25 merge → candidate list
# ──────────────────────────────────────────────────────────────────────────────


def _vector_search_and_merge(
    vectors: list[list[float]],
    composed_query: str,
    raw_user_query: str | None,
    *,
    intents: QueryIntents | None = None,
) -> tuple[dict[int, float], list[tuple[DocumentChunk, float]], bool, int] | None:
    """
    Run cosine-distance vector search + BM25 hybrid, merge distances,
    and fetch the candidate ``DocumentChunk`` objects.

    Returns ``(dist_map, merged_order, has_rows, per_pool)`` or ``None``
    when there are zero candidates after merging.
    """
    has_rows = DocumentChunk.objects.exists()
    per_pool = max(RAG_VECTOR_CANDIDATE_POOL, RAG_TOP_K + RAG_VECTOR_FILL_EXTRA + 8)

    t_vec = time.time()
    best = _merge_best_distances(vectors, per_pool)
    if (
        RAG_OBS_VECTOR_PREFILTER
        and intents
        and intents.academic_obs
        and vectors
        and vectors[0]
    ):
        obs_limit = min(per_pool, 64)
        obs_rows = _merge_obs_host_vector_pool(vectors[0], obs_limit)
        for pk, d in obs_rows.items():
            prev = best.get(pk)
            if prev is None or d < prev:
                best[pk] = d
        if obs_rows:
            logger.info(
                "RAG OBS vector prefilter: merged obs host rows=%d, pool cap=%d",
                len(obs_rows),
                obs_limit,
            )
    logger.info("RAG vector search: %.2fs (pool=%d)", time.time() - t_vec, per_pool)

    # BM25 hybrid
    bm25_hits: list[tuple[int, float]] = []
    t_bm25 = time.time()
    if RAG_BM25_HYBRID:
        bm25_raw_q = (raw_user_query or composed_query or "").strip()
        bm25_hits = _bm25_search(bm25_raw_q, top_n=20)
    for pk, rank in bm25_hits:
        if pk not in best:
            best[pk] = 0.55
    if bm25_hits:
        logger.info(
            "RAG BM25 hybrid: %.2fs (%d hits, %d new)",
            time.time() - t_bm25,
            len(bm25_hits),
            sum(1 for pk, _ in bm25_hits if pk not in best),
        )

    if not best:
        return None

    sorted_pairs = sorted(best.items(), key=lambda x: x[1])[:per_pool]
    pk_order = [pk for pk, _ in sorted_pairs]
    dist_map = dict(sorted_pairs)
    chunk_map = {c.pk: c for c in DocumentChunk.objects.filter(pk__in=pk_order)}
    merged_order: list[tuple[DocumentChunk, float]] = []
    for pk in pk_order:
        ch = chunk_map.get(pk)
        if ch is None:
            continue
        if is_rag_source_url_blocked(str(ch.source_url or "")):
            continue
        merged_order.append((ch, dist_map[pk]))

    return dist_map, merged_order, has_rows, per_pool


# ──────────────────────────────────────────────────────────────────────────────
# Stage 3 — Reranking, distance threshold, fee-intent boosting
# ──────────────────────────────────────────────────────────────────────────────


def _rerank_threshold_and_fee_boost(
    merged_order: list[tuple[DocumentChunk, float]],
    intents: QueryIntents,
    composed_query: str,
    raw_user_query: str | None,
    has_rows: bool,
    per_pool: int,
) -> tuple[list[tuple[DocumentChunk, float]], list[tuple[DocumentChunk, float]], bool]:
    """
    Rerank candidates, apply distance threshold, and boost fee-related chunks.

    Returns ``(vector_block, reranked, used_relaxed)``.
    """
    candidate_slice = merged_order[:per_pool]

    t_rerank = time.time()
    reranked = _rerank_items(
        candidate_slice, composed_query, raw_user_query,
        stem_query=intents.apply_stem_noise,
    )
    reranked = [
        (ch, d) for ch, d in reranked
        if not is_rag_source_url_blocked(str(ch.source_url or ""))
    ]
    if intents.target_alias:
        tuned: list[tuple[DocumentChunk, float]] = []
        for ch, d in reranked:
            blob = f"{ch.page_title or ''}\n{ch.content or ''}\n{ch.source_url or ''}"
            pos, neg = _entity_alignment_score(
                blob, intents.target_alias, intents.competitor_alias,
            )
            tuned.append((ch, d - min(0.07, 0.02 * pos) + min(0.05, 0.01 * neg)))
        reranked = tuned
    logger.info(
        "RAG rerank: %.2fs (%d candidates)",
        time.time() - t_rerank, len(candidate_slice),
    )

    # ── Distance threshold ──
    thresh_hits = [(ch, d) for ch, d in reranked if d <= RAG_MAX_DISTANCE]
    used_relaxed = False
    if len(thresh_hits) >= RAG_TOP_K:
        vector_block = thresh_hits[:RAG_TOP_K]
    elif thresh_hits:
        vector_block = thresh_hits
    elif RAG_RELAX_ON_EMPTY and has_rows and not intents.fee:
        vector_block = reranked[:RAG_TOP_K]
        used_relaxed = bool(vector_block)
    elif RAG_RELAX_ON_EMPTY and has_rows and intents.fee:
        relaxed_fee = [
            (c, d) for c, d in reranked[:per_pool] if _chunk_bears_fee_grounding(c)
        ]
        if relaxed_fee:
            vector_block = relaxed_fee[:RAG_TOP_K]
            used_relaxed = True
        else:
            vector_block = []
    else:
        vector_block = []

    # ── Fee-intent: prepend fee / scholarship URL chunks ──
    if intents.fee and has_rows:
        seen_vb = {c.pk for c, _ in vector_block}
        fee_url_q = (
            Q(source_url__icontains="tuition")
            | Q(source_url__icontains="ucret")
            | Q(source_url__icontains="ücret")
            | Q(source_url__icontains="scholarship")
            | Q(source_url__icontains="/burs/")
            | Q(source_url__icontains="/fees")
            | Q(source_url__icontains="ogrenim-ucret")
            | Q(source_url__icontains="kayit-ucret")
            | Q(source_url__icontains="ogrenim-ucreti")
            | Q(source_url__icontains="ucretlendirme")
            | Q(source_url__icontains="ucret-")
            | Q(source_url__icontains="fee-schedule")
            | Q(source_url__icontains="fee-information")
            | Q(source_url__icontains="price-list")
            | Q(source_url__icontains="fiyat")
            | Q(source_url__icontains="tarife")
            | Q(source_url__icontains="admissions")
            | Q(source_url__icontains="kabul")
        )
        fee_qs = DocumentChunk.objects.filter(fee_url_q)
        bl = rag_source_url_blocklist_substrings()
        if bl:
            block_q = Q()
            for sub in bl:
                block_q |= Q(source_url__icontains=sub)
            fee_qs = fee_qs.exclude(block_q)
        extra_fee = list(
            fee_qs.only("pk", "content", "page_title", "source_url", "embedding")[:24]
        )
        fee_prepend: list[tuple[DocumentChunk, float]] = []
        for ch in extra_fee:
            if ch.pk in seen_vb or not _chunk_bears_fee_grounding(ch):
                continue
            seen_vb.add(ch.pk)
            fee_prepend.append((ch, 0.12))
        if fee_prepend:
            vector_block = fee_prepend + list(vector_block)
        if intents.scholarship:
            scholar_hits = [
                ch for ch in extra_fee
                if _chunk_bears_scholarship_grounding(ch) and ch.pk not in seen_vb
            ][:10]
            scholar_prepend = [(ch, 0.08) for ch in scholar_hits]
            for ch in scholar_hits:
                seen_vb.add(ch.pk)
            if scholar_prepend:
                vector_block = scholar_prepend + list(vector_block)

    # ── Fee grounding filter ──
    if intents.fee:
        filtered = [(c, d) for c, d in vector_block if _chunk_bears_fee_grounding(c)]
        if filtered:
            vector_block = filtered
        else:
            vector_block = []
            used_relaxed = False

    return vector_block, reranked, used_relaxed


# ──────────────────────────────────────────────────────────────────────────────
# Stage 4 — Keyword boosting + final sort
# ──────────────────────────────────────────────────────────────────────────────


def _effective_sort_distance(
    ch: DocumentChunk,
    nominal: float,
    real_dist: dict[int, float],
    intents: QueryIntents,
    faculty_roster_pks: set[int],
) -> float:
    """Compute effective sort distance for a chunk with all intent adjustments."""
    d = real_dist.get(ch.pk, float(nominal))
    if intents.apply_stem_noise:
        d += _stem_noise_penalty(str(ch.source_url or ""), str(ch.page_title or ""))
    if intents.target_alias:
        blob = f"{ch.page_title or ''}\n{ch.content or ''}\n{ch.source_url or ''}"
        pos, neg = _entity_alignment_score(
            blob, intents.target_alias, intents.competitor_alias,
        )
        d = d - min(0.06, 0.02 * pos) + min(0.04, 0.01 * neg)
        if _ENTITY_NOISE_SOURCE_RE.search(str(ch.source_url or "")):
            d += 0.04
        # For OBS/curriculum questions, prefer chunks that explicitly mention the target
        # entity (e.g., Computer Engineering) and demote generic information-package pages.
        if intents.academic_obs:
            low_blob = blob.lower()
            has_target = any(a and a.lower() in low_blob for a in intents.target_alias)
            if has_target and _OBS_URL_RE.search(low_blob):
                d -= 0.08
            elif (
                "information package" in low_blob
                or "program learning outcomes" in low_blob
                or "course structure" in low_blob
            ):
                # Enriched OBS programme pages legitimately contain these headings; do not
                # demote when the URL is already a known target-program shell.
                if not (
                    intents.target_entity
                    and _obs_url_matches_target_entity(str(ch.source_url or ""), intents)
                ):
                    d += 0.22
    if intents.whois_anchor and whois_name_in_content(
        str(ch.content or ""), intents.whois_anchor,
    ):
        hi, lo = _whois_high_low_signal(
            str(ch.source_url or ""), str(ch.page_title or ""),
        )
        if lo and not hi:
            d += 0.36
        elif hi:
            d = min(d, 0.05)
        else:
            d = min(d, 0.12)
    if int(ch.pk) in faculty_roster_pks:
        d = min(d, 0.12)
    # If the user asks for curriculum/course content (OBS intent), generic staff pages
    # should not outrank Bologna/course sources.
    if intents.academic_obs and "academic-staff" in str(ch.source_url or "").lower():
        d += 0.18
    # For curriculum/course requests, non-OBS pages should generally rank below
    # OBS/Bologna pages unless OBS has no useful evidence.
    obs_blob = f"{ch.source_url or ''} {ch.page_title or ''}".lower()
    if intents.academic_obs and not _OBS_URL_RE.search(obs_blob):
        d += 0.24
        # Strongly demote generic department intro pages that often hallucinate "lesson" answers.
        if (
            "message from head of department" in obs_blob
            or "about | acıbadem" in obs_blob
            or "about | acibadem" in obs_blob
            or "faculty of engineering and natural sciences" in obs_blob
        ):
            d += 0.20
    if intents.academic_obs and _OBS_URL_RE.search(obs_blob):
        # Prefer concrete Bologna program endpoints (e.g. showPac, curSunit pages)
        # over generic category pages under the same OBS host.
        if _OBS_PROGRAM_DETAIL_RE.search(obs_blob):
            d -= 0.10
            # For entity-specific curriculum questions, penalize program-detail URLs
            # that still do not mention the requested target (e.g. wrong curUnit/curSunit).
            if intents.target_alias:
                obs_target_blob = (
                    f"{ch.source_url or ''}\n{ch.page_title or ''}\n{ch.content or ''}"
                ).lower()
                if not any(
                    a and a.lower() in obs_target_blob for a in intents.target_alias
                ):
                    match_state = _obs_url_entity_match_state(str(ch.source_url or ""), intents)
                    if match_state is False:
                        d += 0.24
        elif "information package" in obs_blob:
            d += 0.08
    d += _obs_priority_adjustment(
        str(ch.source_url or ""), str(ch.page_title or ""),
        location_intent=intents.location,
        academic_obs_intent=intents.academic_obs,
    )
    d += _obs_entity_code_adjustment(str(ch.source_url or ""), intents)
    # Lesson/course-list questions: same programme may use curSunit=6247 for structure detail.
    if (
        intents.academic_obs
        and intents.target_entity == "computer-engineering"
        and _CE_CURRICULUM_QUERY_RE.search(intents.q_blob)
    ):
        ulow = (ch.source_url or "").lower()
        if "cursunit=6247" in ulow:
            d -= 0.09
    if intents.intl_ug_only and _GRADUATE_INTL_SOURCE_HINT.search(
        f"{ch.source_url or ''} {ch.page_title or ''}"
    ):
        d += 0.18
    return d


def _keyword_boost_and_sort(
    vector_block: list[tuple[DocumentChunk, float]],
    intents: QueryIntents,
    composed_query: str,
    has_rows: bool,
    vectors: list[list[float]],
    dist_map: dict[int, float],
) -> tuple[
    list[tuple[DocumentChunk, float]], dict[int, float], set[int], str | None
]:
    """
    Merge keyword-boosted chunks into the ranking, compute real cosine
    distances, and produce the final sorted list.

    Returns ``(ranked, real_dist, faculty_roster_pks, faculty_path_for_inject)``.
    """
    ranked: list[tuple[DocumentChunk, float]] = []
    seen_pk: set[int] = set()
    faculty_roster_pks: set[int] = set()
    faculty_path_for_inject: str | None = None

    def push_kw(ch: DocumentChunk, nominal: float) -> None:
        if is_rag_source_url_blocked(str(ch.source_url or "")):
            return
        if ch.pk not in seen_pk:
            seen_pk.add(ch.pk)
            ranked.append((ch, nominal))

    whois_chunks: list[DocumentChunk] = []
    t_kw = time.time()

    # Whois keyword boost
    if has_rows and intents.whois_anchor:
        whois_chunks = _load_whois_name_chunks(intents.whois_anchor, 20)
        for ch in whois_chunks[:8]:
            push_kw(ch, 0.59)

    # Faculty roster keyword boost
    if has_rows and intents.faculty_roster:
        path_seg = faculty_roster_path_filter(intents.q_blob)
        if path_seg:
            faculty_path_for_inject = path_seg
            for ch in DocumentChunk.objects.filter(
                Q(source_url__icontains="academic-staff")
                & Q(source_url__icontains=path_seg)
            )[:20]:
                faculty_roster_pks.add(int(ch.pk))
                push_kw(ch, 0.12)

    # Leadership keyword boost
    if has_rows and intents.leadership:
        q_ld = (
            Q(content__icontains="Dean of")
            | Q(content__icontains="Faculty of")
            | Q(page_title__icontains="dean")
            | Q(page_title__icontains="Dekan")
            | Q(content__icontains="Dekan")
            | Q(content__icontains="rector")
            | Q(content__icontains="Rektör")
            | Q(content__icontains="Rektor")
            | Q(content__icontains="Fakülte")
        )
        for ch in DocumentChunk.objects.filter(q_ld)[:22]:
            push_kw(ch, 0.2)

    # STEM keyword boost
    if RAG_KEYWORD_BOOST and has_rows and not intents.fee:
        stem_terms = stem_engineering_boost_terms(composed_query)
        if stem_terms:
            q_filter = Q()
            for term in stem_terms:
                q_filter |= Q(content__icontains=term)
            for ch in DocumentChunk.objects.filter(q_filter)[
                : len(stem_terms) * 6
            ]:
                push_kw(ch, 0.62)

    # Structured list keyword boost
    if RAG_KEYWORD_BOOST and has_rows and not intents.stem:
        struct_terms = structured_list_boost_terms(composed_query)
        if struct_terms:
            q_filter = Q()
            for term in struct_terms:
                q_filter |= Q(content__icontains=term)
            for ch in DocumentChunk.objects.filter(q_filter)[
                : len(struct_terms) * 5
            ]:
                push_kw(ch, 0.65)

    # Vector block chunks
    for ch, d in vector_block:
        push_kw(ch, float(d))

    # General keyword boost
    if RAG_KEYWORD_BOOST and has_rows and not intents.stem:
        kw_terms = rag_keywords_from_query(composed_query)
        if kw_terms:
            q_filter = Q()
            for term in kw_terms:
                q_filter |= Q(content__icontains=term)
            for ch in DocumentChunk.objects.filter(q_filter)[
                : len(kw_terms) * 4
            ]:
                push_kw(ch, 0.72)
    logger.info("RAG keyword boost: %.2fs", time.time() - t_kw)

    # Whois prepend: person-name chunks must compete even if scored poorly
    if has_rows and intents.whois_anchor and whois_chunks:
        have_pk = {ch.pk for ch, _ in ranked}
        prepend: list[tuple[DocumentChunk, float]] = []
        for ch in whois_chunks:
            if ch.pk not in have_pk:
                prepend.append((ch, 0.1))
                have_pk.add(ch.pk)
        if prepend:
            ranked = prepend + ranked

    # Faculty roster prepend
    if has_rows and faculty_roster_pks:
        have_pk = {c.pk for c, _ in ranked}
        fr_pre: list[tuple[DocumentChunk, float]] = []
        for ch in DocumentChunk.objects.filter(pk__in=faculty_roster_pks).order_by(
            "pk"
        ):
            if ch.pk not in have_pk:
                fr_pre.append((ch, 0.1))
                have_pk.add(ch.pk)
        if fr_pre:
            ranked = fr_pre + ranked

    # Real cosine distances for accurate metadata
    primary_vec = vectors[0]
    ranked_pks = [ch.pk for ch, _ in ranked]
    t_dist = time.time()
    real_dist = _cosine_distance_by_pk(ranked_pks, primary_vec)
    logger.info(
        "RAG cosine re-dist: %.2fs (%d pks)", time.time() - t_dist, len(ranked_pks),
    )

    # Final sort by effective distance
    ranked.sort(
        key=lambda it: (
            _effective_sort_distance(
                it[0], it[1], real_dist, intents, faculty_roster_pks,
            ),
            it[0].pk,
        )
    )

    return ranked, real_dist, faculty_roster_pks, faculty_path_for_inject


# ──────────────────────────────────────────────────────────────────────────────
# Stage 5 — Fetch intent-specific full-page blocks
# ──────────────────────────────────────────────────────────────────────────────


def _fetch_faculty_inject(faculty_path: str | None) -> tuple[str, str | None, str]:
    """Returns ``(inject_block, inject_skip_url, inject_title)``."""
    if not faculty_path:
        return "", None, ""
    staff_page = (
        Page.objects.filter(
            Q(url__icontains="academic-staff") & Q(url__icontains=faculty_path)
        )
        .only("url", "title", "content")
        .first()
    )
    if not staff_page or not (staff_page.content or "").strip():
        return "", None, ""
    body = (staff_page.content or "").strip()
    title = (staff_page.title or "")[:200]
    block = (
        f"[{staff_page.title} — faculty listing; name every person and title below]\n"
        f"{body[:10000]}"
    )
    return block, str(staff_page.url), title


def _fetch_management_inject(
    intents: QueryIntents, faculty_path: str | None,
) -> tuple[str, str, str]:
    """Returns ``(mgmt_block, mgmt_url, mgmt_title)``."""
    if not intents.leadership or intents.whois_anchor:
        return "", "", ""
    dept_teacher_list = bool(intents.faculty_roster and faculty_path)
    if dept_teacher_list:
        return "", "", ""
    um = (
        Page.objects.filter(
            Q(url__icontains="university-management")
            & Q(url__icontains="instructors-handbook")
        )
        .only("url", "title", "content")
        .first()
    )
    if not um or not (um.content or "").strip():
        return "", "", ""
    body = (um.content or "").strip()
    title = (um.title or "")[:200]
    block = (
        f"[{um.title} — deans, rector, boards; use only this list for names and titles]\n"
        f"{body[:12000]}"
    )
    return block, str(um.url), title


def _fetch_location_pages(intents: QueryIntents) -> tuple[str, list[dict]]:
    """Returns ``(loc_block, loc_sources)``."""
    if not intents.location or intents.fee:
        return "", []
    loc_q = (
        Q(url__icontains="communication-and-transportation")
        | Q(url__icontains="/kayit/iletisim/ulasim")
        | Q(url__icontains="contact-details")
        | Q(url__icontains="/contact-us")
        | Q(url__icontains="/contact")
        | Q(url__icontains="/iletisim")
        | Q(url__icontains="transport")
        | Q(title__icontains="Communication and Transportation")
        | Q(title__icontains="Contact Details")
        | Q(title__icontains="Contact")
    )
    loc_pages_all = list(
        Page.objects.filter(loc_q)
        .exclude(url__icontains="obs.")
        .exclude(url__icontains="/news/")
        .exclude(url__icontains="/events/")
        .only("url", "title", "content")[:30]
    )

    def _priority(p: Page) -> tuple[int, int]:
        u = str(p.url or "").lower()
        t = str(p.title or "").lower()
        pri = 90
        if "communication-and-transportation" in u or "/kayit/iletisim/ulasim" in u:
            pri = 0
        elif "contact-details" in u:
            pri = 1
        elif "contact details" in t:
            pri = 2
        elif "/contact-us" in u:
            pri = 3
        elif "/contact" in u or "/iletisim" in u:
            pri = 4
        elif "transport" in u:
            pri = 5
        return (pri, len(u))

    loc_pages = sorted(loc_pages_all, key=_priority)[:2]
    parts: list[str] = []
    sources: list[dict] = []
    for p in loc_pages:
        body = (p.content or "").strip()
        if not body:
            continue
        title = str(p.title or p.url or "")[:200]
        parts.append(f"[{title} — location/contact source]\n{body[:1200]}")
        sources.append(
            {"url": str(p.url or ""), "title": title, "cosine_distance": 0.0}
        )
    return "\n\n".join(parts), sources


def _fetch_appreq_page(
    intents: QueryIntents,
) -> tuple[str, list[dict], set[str]]:
    """Returns ``(appreq_block, appreq_sources, appreq_skip)``."""
    if not intents.appreq:
        return "", [], set()
    p_ar = (
        Page.objects.filter(url__icontains="application-requirements")
        .exclude(url__icontains="obs.")
        .only("url", "title", "content")
        .first()
    )
    if not p_ar or not (p_ar.content or "").strip():
        return "", [], set()
    body = (p_ar.content or "").strip()
    t_ar = str(p_ar.title or p_ar.url or "")[:200]
    block = (
        f"[{p_ar.title} — international undergraduate application requirements; "
        f"answer using every required diploma/exam and score in the text below]\n{body[:14000]}"
    )
    sources = [{"url": str(p_ar.url), "title": t_ar, "cosine_distance": 0.0}]
    return block, sources, {str(p_ar.url)}


def _fetch_scholarship_page(
    intents: QueryIntents,
) -> tuple[str, list[dict], set[str]]:
    """Returns ``(scholarship_block, scholarship_sources, scholarship_skip)``."""
    if not intents.scholarship:
        return "", [], set()
    p_sc = (
        Page.objects.filter(
            Q(url__icontains="scholarship-opportunities")
            | Q(url__icontains="/burs/")
            | Q(url__icontains="scholarship")
        )
        .exclude(url__icontains="obs.")
        .only("url", "title", "content")
        .first()
    )
    if not p_sc or not (p_sc.content or "").strip():
        return "", [], set()
    body = (p_sc.content or "").strip()
    t_sc = str(p_sc.title or p_sc.url or "")[:200]
    block = (
        f"[{p_sc.title} — scholarship opportunities source; "
        f"answer from this page first]\n{body[:14000]}"
    )
    sources = [{"url": str(p_sc.url), "title": t_sc, "cosine_distance": 0.0}]
    return block, sources, {str(p_sc.url)}


def _fetch_intl_pages(
    intents: QueryIntents, appreq_skip: set[str],
) -> tuple[str, list[dict]]:
    """Returns ``(intl_block, intl_sources)``."""
    if not intents.intl_apply:
        return "", []
    intl_q = (
        Q(url__icontains="application-requirements")
        | Q(url__icontains="international-student")
        | Q(url__icontains="international-students")
        | Q(url__icontains="international/admission")
        | (Q(url__icontains="international") & Q(url__icontains="admission"))
        | (Q(url__icontains="international") & Q(url__icontains="apply"))
        | Q(url__icontains="yabanci-ogrenci")
        | Q(url__icontains="foreign-student")
        | Q(title__icontains="International Student")
        | Q(title__icontains="International Students")
    )
    intl_pages_all = list(
        Page.objects.filter(intl_q)
        .exclude(url__icontains="obs.")
        .exclude(url__icontains="/news/")
        .exclude(url__icontains="/events/")
        .only("url", "title", "content")[:40]
    )
    if intents.intl_ug_only:
        intl_pages_all = [
            p
            for p in intl_pages_all
            if not _GRADUATE_INTL_SOURCE_HINT.search(
                f"{p.url} {p.title or ''}"
            )
        ]

    def _priority(p: Page) -> tuple[int, int]:
        u = str(p.url or "").lower()
        t = str(p.title or "").lower()
        pri = 90
        if "application-requirements" in u:
            pri = 0
        elif "international" in u and "admission" in u:
            pri = 1
        elif "international-student" in u or "international-students" in u:
            pri = 2
        elif "international" in u and (
            "apply" in u or "basvuru" in u or "kayit" in u
        ):
            pri = 3
        elif "admission" in t and "international" in t:
            pri = 4
        elif "international" in t and (
            "student" in t or "admission" in t
        ):
            pri = 5
        elif "ucret" in u or "fee" in u or "tuition" in u or "fiyat" in u:
            pri = 12
        elif "visa" in u or "residence" in u or "ikamet" in u:
            pri = 15
        return (pri, len(u))

    intl_pages: list[Page] = []
    for p in sorted(intl_pages_all, key=_priority):
        if str(p.url) in appreq_skip:
            continue
        intl_pages.append(p)
        if len(intl_pages) >= 2:
            break
    parts: list[str] = []
    sources: list[dict] = []
    for p in intl_pages:
        body = (p.content or "").strip()
        if not body:
            continue
        title = str(p.title or p.url or "")[:200]
        parts.append(
            f"[{title} — international admission / application source]\n{body[:2200]}"
        )
        sources.append(
            {"url": str(p.url or ""), "title": title, "cosine_distance": 0.0}
        )
    return "\n\n".join(parts), sources


WHOIS_FULL_PAGE_MAX = int(os.environ.get("RAG_WHOIS_FULL_PAGE_MAX", "14000"))


def _fetch_whois_identity_block(anchor: str) -> tuple[str, list[dict]]:
    """
    Pull full HTML-stripped Page bodies that mention this person so “who is …?” answers
    can cite title, department head message, and staff lists—not only small vector chunks.
    """
    a = (anchor or "").strip()
    if not a:
        return "", []
    parts = [p for p in a.split() if len(p) >= 2]
    if not parts:
        return "", []

    def _page_score(p: Page) -> tuple[int, int]:
        u = (p.url or "").lower()
        if "academic-staff" in u:
            return (0, -len(u))
        if "message-from-head" in u or "head-of-department" in u:
            return (2, -len(u))
        if "obs.acibadem.edu.tr" in u:
            return (14, -len(u))
        if "faculty" in u or "department" in u or "bolum" in u:
            return (5, -len(u))
        return (10, -len(u))

    qs = Page.objects.all()
    for p in parts:
        qs = qs.filter(content__icontains=p)
    candidates: list[Page] = list(
        qs.exclude(url__icontains="robots").only("url", "title", "content").order_by("id")[:150],
    )
    candidates = [
        p for p in candidates if whois_name_in_content(str(p.content or ""), a)
    ]
    if not candidates and len(parts) >= 2:
        fallback = (
            Page.objects.filter(content__icontains=parts[-1])
            .exclude(url__icontains="robots")
            .only("url", "title", "content")
            .order_by("id")[:200]
        )
        candidates = [
            p for p in fallback if whois_name_in_content(str(p.content or ""), a)
        ]
    if not candidates:
        return "", []

    candidates.sort(key=_page_score)
    blocks: list[str] = []
    sources: list[dict] = []
    total_chars = 0
    max_total = max(8000, WHOIS_FULL_PAGE_MAX)
    per_page_cap = min(7000, max_total)

    for page in candidates[:8]:
        body = (page.content or "").strip()
        if not body:
            continue
        title = str(page.title or page.url or "")[:220]
        window = snippet_around_phrase(body, a, per_page_cap)
        if not window.strip():
            window = body[:per_page_cap]
        block = (
            f"[Full page extract — identity; use titles, roles, and department names from this text]\n"
            f"[{title}]\n{window}"
        )
        if total_chars + len(block) > max_total and blocks:
            break
        blocks.append(block)
        sources.append(
            {
                "url": str(page.url or ""),
                "title": title,
                "cosine_distance": 0.0,
            }
        )
        total_chars += len(block)
        if total_chars >= max_total:
            break

    return "\n\n".join(blocks), sources


def _fetch_intent_page_blocks(
    intents: QueryIntents, faculty_path: str | None,
) -> IntentPageBlocks:
    """Fetch full-page content blocks for all detected intents."""
    inject_block, inject_skip_url, inject_title = _fetch_faculty_inject(
        faculty_path,
    )
    mgmt_block, mgmt_url, mgmt_title = _fetch_management_inject(
        intents, faculty_path,
    )
    loc_block, loc_sources = _fetch_location_pages(intents)
    appreq_block, appreq_sources, appreq_skip = _fetch_appreq_page(intents)
    scholarship_block, scholarship_sources, scholarship_skip = (
        _fetch_scholarship_page(intents)
    )
    intl_block, intl_sources = _fetch_intl_pages(intents, appreq_skip)
    whois_block, whois_sources = ("", [])
    if intents.whois_anchor:
        whois_block, whois_sources = _fetch_whois_identity_block(intents.whois_anchor)

    return IntentPageBlocks(
        inject_block=inject_block,
        inject_skip_url=inject_skip_url,
        inject_title=inject_title,
        faculty_path=faculty_path,
        mgmt_block=mgmt_block,
        mgmt_url=mgmt_url,
        mgmt_title=mgmt_title,
        loc_block=loc_block,
        loc_sources=loc_sources,
        intl_block=intl_block,
        intl_sources=intl_sources,
        appreq_block=appreq_block,
        appreq_sources=appreq_sources,
        appreq_skip=appreq_skip,
        scholarship_block=scholarship_block,
        scholarship_sources=scholarship_sources,
        scholarship_skip=scholarship_skip,
        whois_block=whois_block,
        whois_sources=whois_sources,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Stage 6 — Context assembly
# ──────────────────────────────────────────────────────────────────────────────


def _assemble_context(
    ranked: list[tuple[DocumentChunk, float]],
    reranked: list[tuple[DocumentChunk, float]],
    blocks: IntentPageBlocks,
    intents: QueryIntents,
    real_dist: dict[int, float],
    dist_map: dict[int, float],
    composed_query: str,
    raw_user_query: str | None,
    used_relaxed: bool,
    faculty_roster_pks: set[int],
) -> tuple[str, list[dict], bool, bool]:
    """
    Build the final context text within character budget, handle inject-only
    early returns, and prepend intent-specific blocks.
    """
    # ── Inject-only early returns ──
    faculty_inject_only = bool(
        blocks.inject_block
        and blocks.faculty_path
        and intents.faculty_roster
        and not intents.whois_anchor
        and not intents.fee
    )
    if faculty_inject_only and blocks.inject_skip_url:
        src = [
            {
                "url": blocks.inject_skip_url,
                "title": blocks.inject_title,
                "cosine_distance": 0.0,
            }
        ]
        logger.info(
            "RAG faculty inject-only context (path=%s, chars=%d)",
            blocks.faculty_path,
            len(blocks.inject_block),
        )
        return blocks.inject_block, src, used_relaxed, True

    if blocks.mgmt_block and blocks.mgmt_url and not intents.fee:
        src_mgmt = [
            {
                "url": blocks.mgmt_url,
                "title": blocks.mgmt_title,
                "cosine_distance": 0.0,
            }
        ]
        logger.info(
            "RAG leadership university-management inject-only (chars=%d)",
            len(blocks.mgmt_block),
        )
        return blocks.mgmt_block, src_mgmt, used_relaxed, True

    if blocks.loc_block and blocks.loc_sources:
        logger.info(
            "RAG location inject-only context (pages=%d, chars=%d)",
            len(blocks.loc_sources),
            len(blocks.loc_block),
        )
        return blocks.loc_block, blocks.loc_sources, used_relaxed, True

    # ── Character budget ──
    max_context_chars = (
        max(RAG_MAX_CHARS, 9000) if blocks.inject_block else RAG_MAX_CHARS
    )
    if blocks.appreq_block:
        max_context_chars = max(max_context_chars, 12000)

    context_parts: list[str] = []
    sources: list[dict] = []
    total = 0
    url_counts: dict[str, int] = {}
    seen_chunk_ids: set[int] = set()

    def try_add(ch: DocumentChunk, nominal_dist: float) -> None:
        nonlocal total
        if intents.fee and not _chunk_bears_fee_grounding(ch):
            return
        if total >= max_context_chars:
            return
        cid = ch.pk
        if cid in seen_chunk_ids:
            return
        u = str(ch.source_url or "")
        if is_rag_source_url_blocked(u):
            return
        if blocks.inject_skip_url and u == blocks.inject_skip_url:
            return
        if u in blocks.appreq_skip:
            return
        if u in blocks.scholarship_skip:
            return
        if intents.intl_ug_only and _GRADUATE_INTL_SOURCE_HINT.search(
            f"{u} {ch.page_title or ''}"
        ):
            return
        max_per_url = RAG_MAX_CHUNKS_PER_URL + (
            1 if intents.scholarship else 0
        )
        if url_counts.get(u, 0) >= max_per_url:
            return
        raw = str(ch.content or "")
        if intents.target_alias:
            blob = f"{ch.page_title or ''}\n{raw}\n{u}"
            pos, neg = _entity_alignment_score(
                blob, intents.target_alias, intents.competitor_alias,
            )
            if pos == 0 and neg > 0:
                return
        # Curriculum/course intent: keep context grounded on OBS program pages.
        # For known entities (e.g. computer-engineering), prefer matching curUnit/curSunit.
        if intents.academic_obs and intents.target_alias:
            blob = f"{ch.page_title or ''}\n{raw}\n{u}".lower()
            has_target_alias = any(
                a and a.lower() in blob for a in intents.target_alias
            )
            if _OBS_URL_RE.search(u):
                if _OBS_PROGRAM_DETAIL_RE.search(u):
                    match_state = _obs_url_entity_match_state(u, intents)
                    if match_state is False and not has_target_alias:
                        return
                elif not has_target_alias:
                    return
            else:
                return
        # Snippet extraction
        if intents.whois_anchor and whois_name_in_content(
            raw, intents.whois_anchor,
        ):
            snippet = snippet_around_phrase(
                raw, intents.whois_anchor, RAG_SNIPPET_CHARS,
            )
        elif intents.fee:
            q_for_anchor = f"{raw_user_query or ''} {composed_query or ''}"
            snippet = raw[:RAG_SNIPPET_CHARS]
            low = raw.lower()
            anchor_candidates = (
                fee_snippet_anchor_phrases(q_for_anchor)
                + department_snippet_anchor_phrases(q_for_anchor)
            )
            for ph in anchor_candidates:
                p = (ph or "").strip()
                if len(p) >= 4 and p.lower() in low:
                    snippet = snippet_around_phrase(raw, p, RAG_SNIPPET_CHARS)
                    break
        elif intents.intl_apply and not intents.fee:
            snippet = raw[:RAG_SNIPPET_CHARS]
            low = raw.lower()
            for ph in (
                "International students",
                "International student",
                "Yabancı",
                "Yabanci",
                "Application",
                "Admission",
                "Apply",
            ):
                if ph.lower() in low:
                    snippet = snippet_around_phrase(raw, ph, RAG_SNIPPET_CHARS)
                    break
        else:
            snippet = raw[:RAG_SNIPPET_CHARS]
        if total + len(snippet) > max_context_chars:
            return
        title = str(ch.page_title or ch.source_url or "")
        context_parts.append(f"[{title}]\n{snippet}")
        total += len(snippet)
        url_counts[u] = url_counts.get(u, 0) + 1
        seen_chunk_ids.add(cid)
        display_d = real_dist.get(cid, nominal_dist)
        sources.append(
            {
                "url": ch.source_url,
                "title": (title or "")[:200],
                "cosine_distance": round(float(display_d), 4),
            }
        )

    # Fill from ranked chunks
    for chunk, dist_val in ranked:
        try_add(chunk, dist_val)

    # Fill extra from reranked pool
    if total < max_context_chars and RAG_VECTOR_FILL_EXTRA > 0:
        pool = [
            (ch, d) for ch, d in reranked if ch.pk not in seen_chunk_ids
        ][: RAG_VECTOR_FILL_EXTRA * 4]
        if intents.fee:
            pool = [(c, d) for c, d in pool if _chunk_bears_fee_grounding(c)]
        fill_slice = pool[:RAG_VECTOR_FILL_EXTRA]
        for ch, _ in fill_slice:
            if ch.pk not in real_dist:
                real_dist[ch.pk] = dist_map.get(ch.pk, 1.0)
        fill_slice.sort(
            key=lambda it: (
                _effective_sort_distance(
                    it[0], it[1], real_dist, intents, faculty_roster_pks,
                ),
                it[0].pk,
            )
        )
        for ch, d in fill_slice:
            try_add(ch, float(d))
            if total >= max_context_chars:
                break

    # If strict filters yielded no programme-detail OBS evidence for curriculum queries,
    # force a small OBS-only fallback to avoid generic / selector-only context.
    #
    # Rationale: OBS often serves multiple sections (Course Structure, Outcomes, etc.)
    # under the same UI; vector search may first hit unitSelection/index shells.
    have_obs_programme_detail = any(
        _OBS_PROGRAM_DETAIL_RE.search(str(s.get("url") or ""))
        for s in sources
    )
    if intents.academic_obs and (not context_parts or not have_obs_programme_detail):
        for ch in _academic_obs_fallback_chunks(intents):
            u = str(ch.source_url or "")
            raw = str(ch.content or "")
            if not raw or not u:
                continue
            title = str(ch.page_title or ch.source_url or "")
            snippet = raw[:RAG_SNIPPET_CHARS]
            context_parts.append(f"[{title}]\n{snippet}")
            sources.append(
                {
                    "url": ch.source_url,
                    "title": (title or "")[:200],
                    "cosine_distance": round(float(real_dist.get(ch.pk, 0.9999)), 4),
                }
            )

    # ── Prepend intent blocks ──
    out = "\n\n".join(context_parts)
    if blocks.scholarship_block:
        out = (
            f"{blocks.scholarship_block}\n\n{out}"
            if out
            else blocks.scholarship_block
        )
    if blocks.intl_block:
        out = f"{blocks.intl_block}\n\n{out}" if out else blocks.intl_block
    if blocks.appreq_block:
        out = f"{blocks.appreq_block}\n\n{out}" if out else blocks.appreq_block
    if blocks.appreq_block or blocks.intl_block or blocks.scholarship_block:
        surls = (
            {str(s.get("url") or "") for s in blocks.scholarship_sources}
            if blocks.scholarship_block
            else set()
        )
        aurls = (
            {str(s.get("url") or "") for s in blocks.appreq_sources}
            if blocks.appreq_block
            else set()
        )
        iurls = (
            {str(s.get("url") or "") for s in blocks.intl_sources}
            if blocks.intl_block
            else set()
        )
        head_u = (surls | aurls | iurls) - {""}
        tail = [
            s for s in sources if str(s.get("url") or "") not in head_u
        ]
        sources = (
            (list(blocks.scholarship_sources) if blocks.scholarship_block else [])
            + (list(blocks.appreq_sources) if blocks.appreq_block else [])
            + (list(blocks.intl_sources) if blocks.intl_block else [])
            + tail
        )
    if blocks.loc_block:
        out = f"{blocks.loc_block}\n\n{out}" if out else blocks.loc_block
        existing_urls = {str(s.get("url") or "") for s in sources}
        loc_prepend = [
            s
            for s in blocks.loc_sources
            if str(s.get("url") or "") not in existing_urls
        ]
        if loc_prepend:
            sources = loc_prepend + sources
    if blocks.inject_block:
        out = f"{blocks.inject_block}\n\n{out}" if out else blocks.inject_block
        if blocks.inject_skip_url and not any(
            str(s.get("url") or "") == blocks.inject_skip_url for s in sources
        ):
            sources.insert(
                0,
                {
                    "url": blocks.inject_skip_url,
                    "title": blocks.inject_title,
                    "cosine_distance": 0.0,
                },
            )

    if blocks.whois_block:
        out = f"{blocks.whois_block}\n\n{out}" if out else blocks.whois_block
        head_u = {str(s.get("url") or "") for s in blocks.whois_sources if s.get("url")}
        tail = [s for s in sources if str(s.get("url") or "") not in head_u]
        sources = list(blocks.whois_sources) + tail

    return out, sources, used_relaxed, True


# ──────────────────────────────────────────────────────────────────────────────
# Public entry point (orchestrator)
# ──────────────────────────────────────────────────────────────────────────────


def search_document_chunks(
    composed_query: str, raw_user_query: str | None = None
) -> tuple[str, list[dict], bool, bool]:
    """
    Returns (context_text, sources, used_relaxed_fallback, embedding_ok).
    """
    t0 = time.time()
    composed_query = (composed_query or "").strip()
    if not composed_query:
        return "", [], False, True

    # Stage 1: detect query intents
    intents = _detect_query_intents(composed_query, raw_user_query)

    # Stage 2a: embed query variants
    variants = _embedding_variants(composed_query, raw_user_query)
    try:
        t_emb = time.time()
        vectors = embed_texts(variants) if variants else []
        logger.info(
            "RAG embed: %.2fs (%d variants)", time.time() - t_emb, len(variants),
        )
    except Exception:
        logger.warning(
            "RAG embed failed, falling back to lexical search", exc_info=True,
        )
        context, sources, relaxed, emb_ok = _lexical_fallback_from_chunks(
            composed_query, raw_user_query,
        )
        if context:
            return context, sources, relaxed, emb_ok
        return _lexical_fallback_from_pages(composed_query, raw_user_query)

    vectors = [v for v in vectors if v]
    if not vectors:
        return "", [], False, False

    # Stage 2b: vector search + BM25 merge
    retrieval = _vector_search_and_merge(
        vectors, composed_query, raw_user_query, intents=intents,
    )
    if retrieval is None:
        return "", [], False, True

    dist_map, merged_order, has_rows, per_pool = retrieval

    # Stage 3: rerank, threshold, fee boost
    vector_block, reranked, used_relaxed = _rerank_threshold_and_fee_boost(
        merged_order, intents, composed_query, raw_user_query, has_rows, per_pool,
    )

    # Stage 4: keyword boosting + final sort
    ranked, real_dist, faculty_roster_pks, faculty_path = (
        _keyword_boost_and_sort(
            vector_block, intents, composed_query, has_rows, vectors, dist_map,
        )
    )

    if intents.whois_anchor:
        ranked = _filter_whois_identity_chunks(ranked, intents.whois_anchor)
        reranked = _filter_whois_identity_chunks(reranked, intents.whois_anchor)

    # Stage 5: fetch intent-specific full-page blocks
    page_blocks = _fetch_intent_page_blocks(intents, faculty_path)

    # Stage 6: assemble context within character budget
    result = _assemble_context(
        ranked, reranked, page_blocks, intents,
        real_dist, dist_map, composed_query, raw_user_query,
        used_relaxed, faculty_roster_pks,
    )

    logger.info(
        "RAG total: %.2fs, chunks=%d, chars=%d",
        time.time() - t0, len(result[1]), len(result[0]),
    )
    return result
