"""
Vector RAG retrieval: multi-query embeddings, merged distances, wide candidate pool,
rerank (word overlap + optional pg_trgm + cross-encoder), then char-budget fill.
"""

from __future__ import annotations

import logging
import re
import time

from django.contrib.postgres.search import SearchQuery, SearchRank, SearchVector
from django.db import connection
from django.db.models import Q
from pgvector.django import CosineDistance

from core.embeddings import embed_texts
from core.models import DocumentChunk, Page
from core.rag_config import (
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
    Rows mentioning this person, using first-name (or only token) as a DB filter, then
    whois_name_in_content (Turkish/Latin fold) in Python. This catches 'Ziraksima' query
    vs 'Zıraksıma' in HTML where SQL AND on ASCII tokens fails.
    """
    a = (anchor or "").strip()
    if not a or out_limit < 1:
        return []
    parts = [p for p in a.split() if len(p) >= 2]
    if not parts:
        q = DocumentChunk.objects.filter(content__icontains=a)[:out_limit]
        return [ch for ch in q if not is_rag_source_url_blocked(str(ch.source_url or ""))]
    out: list[DocumentChunk] = []
    for ch in DocumentChunk.objects.filter(content__icontains=parts[0])[:scan_limit]:
        if is_rag_source_url_blocked(str(ch.source_url or "")):
            continue
        if whois_name_in_content(str(ch.content or ""), anchor):
            out.append(ch)
        if len(out) >= out_limit:
            break
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
        return -0.10
    return 0.03


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

    q_blob = f"{composed_query} {raw_user_query or ''}".strip()
    fee_intent = fee_tuition_intent(q_blob)
    intl_apply_intent = international_student_apply_intent(q_blob)
    appreq_intent = international_application_requirements_page_intent(q_blob)
    intl_ug_only = international_admissions_default_undergraduate_only(q_blob)
    scholarship_intent = bool(_SCHOLARSHIP_QUERY_RE.search(q_blob))
    location_intent = bool(RAG_LOCATION_CONTACT_INTENT_RE.search(q_blob))
    academic_obs_intent = bool(RAG_ACADEMIC_OBS_INTENT_RE.search(q_blob))
    target_entity = extract_target_entity_key(q_blob)
    target_alias = target_entity_aliases(target_entity)
    competitor_alias = target_entity_competitor_aliases(target_entity)

    whois_anchor: str | None = None
    if RAG_WHOIS_QUERY_EXPAND:
        _wn = whois_name_from_queries(composed_query, raw_user_query)
        if _wn and len(_wn) >= 8:
            whois_anchor = _wn

    variants = _embedding_variants(composed_query, raw_user_query)
    try:
        t_emb = time.time()
        vectors = embed_texts(variants) if variants else []
        logger.info("RAG embed: %.2fs (%d variants)", time.time() - t_emb, len(variants))
    except Exception:
        logger.warning("RAG embed failed, falling back to lexical search", exc_info=True)
        context, sources, relaxed, emb_ok = _lexical_fallback_from_chunks(
            composed_query, raw_user_query
        )
        if context:
            return context, sources, relaxed, emb_ok
        return _lexical_fallback_from_pages(composed_query, raw_user_query)
    vectors = [v for v in vectors if v]
    if not vectors:
        return "", [], False, False

    has_rows = DocumentChunk.objects.exists()
    per_pool = max(
        RAG_VECTOR_CANDIDATE_POOL,
        RAG_TOP_K + RAG_VECTOR_FILL_EXTRA + 8,
    )

    t_vec = time.time()
    best = _merge_best_distances(vectors, per_pool)
    logger.info("RAG vector search: %.2fs (pool=%d)", time.time() - t_vec, per_pool)

    # BM25 hybrid: merge full-text search results into vector pool
    bm25_hits = []
    t_bm25 = time.time()
    if RAG_BM25_HYBRID:
        bm25_raw_q = (raw_user_query or composed_query or "").strip()
        bm25_hits = _bm25_search(bm25_raw_q, top_n=20)
    for pk, rank in bm25_hits:
        if pk not in best:
            best[pk] = 0.55  # inject with moderate distance so reranker can promote
    if bm25_hits:
        logger.info("RAG BM25 hybrid: %.2fs (%d hits, %d new)", time.time() - t_bm25, len(bm25_hits), sum(1 for pk, _ in bm25_hits if pk not in best))

    if not best:
        return "", [], False, True

    sorted_pairs = sorted(best.items(), key=lambda x: x[1])[:per_pool]
    pk_order = [pk for pk, _ in sorted_pairs]
    dist_map = dict(sorted_pairs)
    chunk_map = {
        c.pk: c for c in DocumentChunk.objects.filter(pk__in=pk_order)
    }
    merged_order: list[tuple[DocumentChunk, float]] = []
    for pk in pk_order:
        ch = chunk_map.get(pk)
        if ch is None:
            continue
        if is_rag_source_url_blocked(str(ch.source_url or "")):
            continue
        d = dist_map[pk]
        merged_order.append((ch, d))

    candidate_slice = merged_order[:per_pool]
    stem_intent = bool(
        RAG_STEM_OR_ENGINEERING_INTENT_RE.search(
            f"{composed_query} {raw_user_query or ''}"
        )
    )
    # STEM URL noise penalty would punish real /fees/ URLs; also fee questions are not "STEM list" queries.
    apply_stem_noise = stem_intent and not fee_intent

    t_rerank = time.time()
    reranked = _rerank_items(
        candidate_slice, composed_query, raw_user_query, stem_query=apply_stem_noise
    )
    reranked = [
        (ch, d) for ch, d in reranked if not is_rag_source_url_blocked(str(ch.source_url or ""))
    ]
    if target_alias:
        tuned: list[tuple[DocumentChunk, float]] = []
        for ch, d in reranked:
            blob = f"{ch.page_title or ''}\n{ch.content or ''}\n{ch.source_url or ''}"
            pos, neg = _entity_alignment_score(blob, target_alias, competitor_alias)
            d2 = d - min(0.07, 0.02 * pos) + min(0.05, 0.01 * neg)
            tuned.append((ch, d2))
        reranked = tuned
    logger.info("RAG rerank: %.2fs (%d candidates)", time.time() - t_rerank, len(candidate_slice))

    thresh_hits = [(ch, d) for ch, d in reranked if d <= RAG_MAX_DISTANCE]
    used_relaxed = False
    if len(thresh_hits) >= RAG_TOP_K:
        vector_block = thresh_hits[:RAG_TOP_K]
    elif thresh_hits:
        vector_block = thresh_hits
    elif RAG_RELAX_ON_EMPTY and has_rows and not fee_intent:
        vector_block = reranked[:RAG_TOP_K]
        used_relaxed = bool(vector_block)
    elif RAG_RELAX_ON_EMPTY and has_rows and fee_intent:
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

    if fee_intent and has_rows:
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
        if scholarship_intent:
            scholar_hits = [
                ch for ch in extra_fee if _chunk_bears_scholarship_grounding(ch) and ch.pk not in seen_vb
            ][:10]
            scholar_prepend = [(ch, 0.08) for ch in scholar_hits]
            for ch in scholar_hits:
                seen_vb.add(ch.pk)
            if scholar_prepend:
                vector_block = scholar_prepend + list(vector_block)

    if fee_intent:
        filtered = [(c, d) for c, d in vector_block if _chunk_bears_fee_grounding(c)]
        if filtered:
            vector_block = filtered
        else:
            vector_block = []
            used_relaxed = False

    ranked: list[tuple[DocumentChunk, float]] = []
    seen_pk: set[int] = set()

    def push_kw(ch: DocumentChunk, nominal: float) -> None:
        if is_rag_source_url_blocked(str(ch.source_url or "")):
            return
        pk = ch.pk
        if pk not in seen_pk:
            seen_pk.add(pk)
            ranked.append((ch, nominal))

    whois_chunks: list[DocumentChunk] = []
    t_kw = time.time()
    faculty_roster_pks: set[int] = set()
    faculty_path_for_inject: str | None = None
    if has_rows and whois_anchor:
        whois_chunks = _load_whois_name_chunks(whois_anchor, 20)
        for ch in whois_chunks[:8]:
            push_kw(ch, 0.59)

    if has_rows and RAG_FACULTY_ROSTER_INTENT_RE.search(q_blob):
        path_seg = faculty_roster_path_filter(q_blob)
        if path_seg:
            faculty_path_for_inject = path_seg
            for ch in (
                DocumentChunk.objects.filter(
                    Q(source_url__icontains="academic-staff")
                    & Q(source_url__icontains=path_seg)
                )[:20]
            ):
                faculty_roster_pks.add(int(ch.pk))
                push_kw(ch, 0.12)

    if has_rows and RAG_LEADERSHIP_INTENT_RE.search(q_blob):
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

    if RAG_KEYWORD_BOOST and has_rows and not fee_intent:
        stem_terms = stem_engineering_boost_terms(composed_query)
        if stem_terms:
            q_filter = Q()
            for term in stem_terms:
                q_filter |= Q(content__icontains=term)
            for ch in DocumentChunk.objects.filter(q_filter)[:len(stem_terms) * 6]:
                push_kw(ch, 0.62)

    if RAG_KEYWORD_BOOST and has_rows and not stem_intent:
        struct_terms = structured_list_boost_terms(composed_query)
        if struct_terms:
            q_filter = Q()
            for term in struct_terms:
                q_filter |= Q(content__icontains=term)
            for ch in DocumentChunk.objects.filter(q_filter)[:len(struct_terms) * 5]:
                push_kw(ch, 0.65)

    for ch, d in vector_block:
        push_kw(ch, float(d))

    if RAG_KEYWORD_BOOST and has_rows and not stem_intent:
        kw_terms = rag_keywords_from_query(composed_query)
        if kw_terms:
            q_filter = Q()
            for term in kw_terms:
                q_filter |= Q(content__icontains=term)
            for ch in DocumentChunk.objects.filter(q_filter)[:len(kw_terms) * 4]:
                push_kw(ch, 0.72)
    logger.info("RAG keyword boost: %.2fs", time.time() - t_kw)

    # Who-is: any chunk that literally contains the person name must compete in the pool,
    # even if the primary query vector scored it poorly (asymmetric retrieval).
    if has_rows and whois_anchor and whois_chunks:
        have_pk = {ch.pk for ch, _ in ranked}
        prepend: list[tuple[DocumentChunk, float]] = []
        for ch in whois_chunks:
            if ch.pk not in have_pk:
                prepend.append((ch, 0.1))
                have_pk.add(ch.pk)
        if prepend:
            ranked = prepend + ranked

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

    primary_vec = vectors[0]
    ranked_pks = [ch.pk for ch, _ in ranked]

    t_dist = time.time()
    real_dist = _cosine_distance_by_pk(ranked_pks, primary_vec)
    logger.info("RAG cosine re-dist: %.2fs (%d pks)", time.time() - t_dist, len(ranked_pks))

    def _effective_sort_distance(ch: DocumentChunk, nominal: float) -> float:
        d = real_dist.get(ch.pk, float(nominal))
        if apply_stem_noise:
            d += _stem_noise_penalty(
                str(ch.source_url or ""), str(ch.page_title or "")
            )
        if target_alias:
            blob = f"{ch.page_title or ''}\n{ch.content or ''}\n{ch.source_url or ''}"
            pos, neg = _entity_alignment_score(blob, target_alias, competitor_alias)
            d = d - min(0.06, 0.02 * pos) + min(0.04, 0.01 * neg)
            if _ENTITY_NOISE_SOURCE_RE.search(str(ch.source_url or "")):
                d += 0.04
        if whois_anchor and whois_name_in_content(str(ch.content or ""), whois_anchor):
            d = min(d, 0.08)
        if int(ch.pk) in faculty_roster_pks:
            d = min(d, 0.12)
        d += _obs_priority_adjustment(
            str(ch.source_url or ""),
            str(ch.page_title or ""),
            location_intent=location_intent,
            academic_obs_intent=academic_obs_intent,
        )
        if intl_ug_only and _GRADUATE_INTL_SOURCE_HINT.search(
            f"{ch.source_url or ''} {ch.page_title or ''}"
        ):
            d += 0.18
        return d

    ranked.sort(key=lambda it: (_effective_sort_distance(it[0], it[1]), it[0].pk))

    # Full Page row: embeddings/chunks can miss card-style names; inject raw HTML-stripped text.
    inject_block = ""
    inject_skip_url: str | None = None
    inject_title = ""
    if faculty_path_for_inject:
        staff_page = (
            Page.objects.filter(
                Q(url__icontains="academic-staff")
                & Q(url__icontains=faculty_path_for_inject)
            )
            .only("url", "title", "content")
            .first()
        )
        if staff_page and (staff_page.content or "").strip():
            body = (staff_page.content or "").strip()
            inject_title = (staff_page.title or "")[:200]
            inject_skip_url = str(staff_page.url)
            cap = 10000
            inject_block = (
                f"[{staff_page.title} — faculty listing; name every person and title below]\n"
                f"{body[:cap]}"
            )

    # Official rector + deans + boards (English "University Management" page). Vector search
    # often returns irrelevant news; inject the authoritative page for generic leadership questions.
    mgmt_block = ""
    mgmt_url: str = ""
    mgmt_title = ""
    if RAG_LEADERSHIP_INTENT_RE.search(q_blob) and not whois_anchor:
        dept_teacher_list = bool(
            RAG_FACULTY_ROSTER_INTENT_RE.search(q_blob) and faculty_path_for_inject
        )
        if not dept_teacher_list:
            um = (
                Page.objects.filter(
                    Q(url__icontains="university-management")
                    & Q(url__icontains="instructors-handbook")
                )
                .only("url", "title", "content")
                .first()
            )
            if um and (um.content or "").strip():
                body_um = (um.content or "").strip()
                mgmt_title = (um.title or "")[:200]
                mgmt_url = str(um.url)
                mgmt_block = (
                    f"[{um.title} — deans, rector, boards; use only this list for names and titles]\n"
                    f"{body_um[:12000]}"
                )

    loc_block = ""
    loc_sources: list[dict] = []
    intl_block = ""
    intl_sources: list[dict] = []
    appreq_block = ""
    appreq_sources: list[dict] = []
    appreq_skip: set[str] = set()
    scholarship_block = ""
    scholarship_sources: list[dict] = []
    scholarship_skip: set[str] = set()
    if location_intent and not fee_intent:
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
        def _loc_priority(p: Page) -> tuple[int, int]:
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

        loc_pages = sorted(loc_pages_all, key=_loc_priority)[:2]
        loc_parts: list[str] = []
        for p in loc_pages:
            body = (p.content or "").strip()
            if not body:
                continue
            title = str(p.title or p.url or "")[:200]
            loc_parts.append(f"[{title} — location/contact source]\n{body[:1200]}")
            loc_sources.append(
                {
                    "url": str(p.url or ""),
                    "title": title,
                    "cosine_distance": 0.0,
                }
            )
        if loc_parts:
            loc_block = "\n\n".join(loc_parts)

    if appreq_intent:
        p_ar = (
            Page.objects.filter(url__icontains="application-requirements")
            .exclude(url__icontains="obs.")
            .only("url", "title", "content")
            .first()
        )
        if p_ar and (p_ar.content or "").strip():
            body = (p_ar.content or "").strip()
            t_ar = str(p_ar.title or p_ar.url or "")[:200]
            appreq_block = (
                f"[{p_ar.title} — international undergraduate application requirements; "
                f"answer using every required diploma/exam and score in the text below]\n{body[:14000]}"
            )
            appreq_sources = [
                {
                    "url": str(p_ar.url),
                    "title": t_ar,
                    "cosine_distance": 0.0,
                }
            ]
            appreq_skip.add(str(p_ar.url))

    if scholarship_intent:
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
        if p_sc and (p_sc.content or "").strip():
            body = (p_sc.content or "").strip()
            t_sc = str(p_sc.title or p_sc.url or "")[:200]
            scholarship_block = (
                f"[{p_sc.title} — scholarship opportunities source; answer from this page first]\n{body[:14000]}"
            )
            scholarship_sources = [
                {
                    "url": str(p_sc.url),
                    "title": t_sc,
                    "cosine_distance": 0.0,
                }
            ]
            scholarship_skip.add(str(p_sc.url))

    if intl_apply_intent:
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
        if intl_ug_only:
            intl_pages_all = [
                p
                for p in intl_pages_all
                if not _GRADUATE_INTL_SOURCE_HINT.search(
                    f"{p.url} {p.title or ''}"
                )
            ]

        def _intl_priority(p: Page) -> tuple[int, int]:
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
        for p in sorted(intl_pages_all, key=_intl_priority):
            if str(p.url) in appreq_skip:
                continue
            intl_pages.append(p)
            if len(intl_pages) >= 2:
                break
        intl_parts: list[str] = []
        for p in intl_pages:
            body = (p.content or "").strip()
            if not body:
                continue
            title = str(p.title or p.url or "")[:200]
            intl_parts.append(
                f"[{title} — international admission / application source]\n{body[:2200]}"
            )
            intl_sources.append(
                {
                    "url": str(p.url or ""),
                    "title": title,
                    "cosine_distance": 0.0,
                }
            )
        if intl_parts:
            intl_block = "\n\n".join(intl_parts)

    # Full academic-staff page + "list all teachers" query: do not mix other URLs (news, other
    # departments, site-wide "Akademik" pages)—the model otherwise blends unrelated names.
    faculty_inject_only = bool(
        inject_block
        and faculty_path_for_inject
        and RAG_FACULTY_ROSTER_INTENT_RE.search(q_blob)
        and not whois_anchor
        and not fee_intent
    )
    if faculty_inject_only and inject_skip_url:
        src = [
            {
                "url": inject_skip_url,
                "title": inject_title,
                "cosine_distance": 0.0,
            }
        ]
        logger.info(
            "RAG faculty inject-only context (path=%s, chars=%d)",
            faculty_path_for_inject,
            len(inject_block or ""),
        )
        return inject_block, src, used_relaxed, True

    if mgmt_block and mgmt_url and not fee_intent:
        src_mgmt = [
            {
                "url": mgmt_url,
                "title": mgmt_title,
                "cosine_distance": 0.0,
            }
        ]
        logger.info(
            "RAG leadership university-management inject-only (chars=%d)",
            len(mgmt_block),
        )
        return mgmt_block, src_mgmt, used_relaxed, True

    if loc_block and loc_sources:
        logger.info(
            "RAG location inject-only context (pages=%d, chars=%d)",
            len(loc_sources),
            len(loc_block),
        )
        return loc_block, loc_sources, used_relaxed, True

    max_context_chars = (
        max(RAG_MAX_CHARS, 9000) if inject_block else RAG_MAX_CHARS
    )
    if appreq_block:
        max_context_chars = max(max_context_chars, 12000)

    context_parts: list[str] = []
    sources: list[dict] = []
    total = 0
    url_counts: dict[str, int] = {}
    seen_chunk_ids: set[int] = set()

    def try_add_context_chunk(ch: DocumentChunk, nominal_dist: float) -> None:
        nonlocal total
        if fee_intent and not _chunk_bears_fee_grounding(ch):
            return
        if total >= max_context_chars:
            return
        cid = ch.pk
        if cid in seen_chunk_ids:
            return
        u = str(ch.source_url or "")
        if is_rag_source_url_blocked(u):
            return
        if inject_skip_url and u == inject_skip_url:
            return
        if u in appreq_skip:
            return
        if u in scholarship_skip:
            return
        if intl_ug_only and _GRADUATE_INTL_SOURCE_HINT.search(
            f"{u} {ch.page_title or ''}"
        ):
            return
        max_chunks_per_url = RAG_MAX_CHUNKS_PER_URL + (1 if scholarship_intent else 0)
        if url_counts.get(u, 0) >= max_chunks_per_url:
            return
        raw = str(ch.content or "")
        if target_alias:
            blob = f"{ch.page_title or ''}\n{raw}\n{u}"
            pos, neg = _entity_alignment_score(blob, target_alias, competitor_alias)
            if pos == 0 and neg > 0:
                return
        if whois_anchor and whois_name_in_content(raw, whois_anchor):
            snippet = snippet_around_phrase(raw, whois_anchor, RAG_SNIPPET_CHARS)
        elif fee_intent:
            # Fee tables often list the program below the default chunk prefix; center on the department row.
            q_for_anchor = f"{raw_user_query or ''} {composed_query or ''}"
            snippet = raw[:RAG_SNIPPET_CHARS]
            low = raw.lower()
            anchor_candidates = fee_snippet_anchor_phrases(q_for_anchor) + department_snippet_anchor_phrases(
                q_for_anchor
            )
            for ph in anchor_candidates:
                p = (ph or "").strip()
                if len(p) >= 4 and p.lower() in low:
                    snippet = snippet_around_phrase(raw, p, RAG_SNIPPET_CHARS)
                    break
        elif intl_apply_intent and not fee_intent:
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

    for chunk, dist_val in ranked:
        try_add_context_chunk(chunk, dist_val)

    if total < max_context_chars and RAG_VECTOR_FILL_EXTRA > 0:
        pool = [
            (ch, d) for ch, d in reranked if ch.pk not in seen_chunk_ids
        ][: RAG_VECTOR_FILL_EXTRA * 4]
        if fee_intent:
            pool = [(c, d) for c, d in pool if _chunk_bears_fee_grounding(c)]
        fill_slice = pool[:RAG_VECTOR_FILL_EXTRA]
        for ch, _ in fill_slice:
            if ch.pk not in real_dist:
                real_dist[ch.pk] = dist_map.get(ch.pk, 1.0)
        fill_slice.sort(
            key=lambda it: (_effective_sort_distance(it[0], it[1]), it[0].pk)
        )
        for ch, d in fill_slice:
            try_add_context_chunk(ch, float(d))
            if total >= max_context_chars:
                break

    out = "\n\n".join(context_parts)
    if scholarship_block:
        out = f"{scholarship_block}\n\n{out}" if out else scholarship_block
    if intl_block:
        out = f"{intl_block}\n\n{out}" if out else intl_block
    if appreq_block:
        out = f"{appreq_block}\n\n{out}" if out else appreq_block
    if appreq_block or intl_block or scholarship_block:
        surls = {str(s.get("url") or "") for s in scholarship_sources} if scholarship_block else set()
        aurls = {str(s.get("url") or "") for s in appreq_sources} if appreq_block else set()
        iurls = {str(s.get("url") or "") for s in intl_sources} if intl_block else set()
        head_u = (surls | aurls | iurls) - {""}
        tail = [s for s in sources if str(s.get("url") or "") not in head_u]
        sources = (
            (list(scholarship_sources) if scholarship_block else [])
            + (list(appreq_sources) if appreq_block else [])
            + (list(intl_sources) if intl_block else [])
            + tail
        )
    if loc_block:
        out = f"{loc_block}\n\n{out}" if out else loc_block
        existing_urls = {str(s.get("url") or "") for s in sources}
        loc_prepend = [s for s in loc_sources if str(s.get("url") or "") not in existing_urls]
        if loc_prepend:
            sources = loc_prepend + sources
    if inject_block:
        out = f"{inject_block}\n\n{out}" if out else inject_block
        if inject_skip_url and not any(
            str(s.get("url") or "") == inject_skip_url for s in sources
        ):
            sources.insert(
                0,
                {
                    "url": inject_skip_url,
                    "title": inject_title,
                    "cosine_distance": 0.0,
                },
            )

    logger.info("RAG total: %.2fs, chunks=%d, chars=%d", time.time() - t0, len(sources), total)
    return out, sources, used_relaxed, True
