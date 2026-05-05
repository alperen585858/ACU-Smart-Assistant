"""RAG tuning from environment (shared by chat views and retrieval)."""

import os


def _env_bool(key: str, default: str = "true") -> bool:
    return os.environ.get(key, default).lower() in ("1", "true", "yes")


RAG_MAX_CHARS = max(800, int(os.environ.get("RAG_MAX_CHARS", "3000")))
RAG_TOP_K = int(os.environ.get("RAG_TOP_K", "8"))
RAG_VECTOR_FILL_EXTRA = max(0, int(os.environ.get("RAG_VECTOR_FILL_EXTRA", "4")))
RAG_MAX_CHUNKS_PER_URL = max(1, int(os.environ.get("RAG_MAX_CHUNKS_PER_URL", "2")))
RAG_MAX_DISTANCE = float(os.environ.get("RAG_MAX_DISTANCE", "0.68"))
RAG_RELAX_ON_EMPTY = _env_bool("RAG_RELAX_ON_EMPTY", "true")
RAG_KEYWORD_BOOST = _env_bool("RAG_KEYWORD_BOOST", "true")
RAG_SNIPPET_CHARS = min(
    max(400, int(os.environ.get("RAG_SNIPPET_CHARS", "700"))),
    RAG_MAX_CHARS,
)

# Recall-oriented (multi-embed, wide pool, rerank, optional lexical on Postgres)
RAG_MULTI_EMBED = _env_bool("RAG_MULTI_EMBED", "false")
RAG_MULTI_EMBED_KEYWORD_LINE = _env_bool("RAG_MULTI_EMBED_KEYWORD_LINE", "false")
RAG_VECTOR_CANDIDATE_POOL = max(20, min(500, int(os.environ.get("RAG_VECTOR_CANDIDATE_POOL", "80"))))
RAG_RERANK_OVERLAP_WEIGHT = float(os.environ.get("RAG_RERANK_OVERLAP_WEIGHT", "0.06"))
RAG_LEXICAL_WEIGHT = float(os.environ.get("RAG_LEXICAL_WEIGHT", "0.12"))
RAG_CROSS_ENCODER_RERANK = _env_bool("RAG_CROSS_ENCODER_RERANK", "false")
RAG_CROSS_ENCODER_WEIGHT = float(os.environ.get("RAG_CROSS_ENCODER_WEIGHT", "0.10"))
RAG_BM25_HYBRID = _env_bool("RAG_BM25_HYBRID", "true")
RAG_WHOIS_QUERY_EXPAND = _env_bool("RAG_WHOIS_QUERY_EXPAND", "true")
RAG_WHOIS_EXTRA_EMBEDS = max(0, min(4, int(os.environ.get("RAG_WHOIS_EXTRA_EMBEDS", "3"))))

# Added to effective cosine distance when sorting context (STEM queries); pushes news/events pages down.
RAG_STEM_NOISE_URL_PENALTY = float(os.environ.get("RAG_STEM_NOISE_URL_PENALTY", "0.14"))

# Academic OBS fallback (when strict context assembly yields no chunks): max chunks to inject.
RAG_ACADEMIC_OBS_FALLBACK_LIMIT = max(
    1, min(24, int(os.environ.get("RAG_ACADEMIC_OBS_FALLBACK_LIMIT", "5")))
)

# When true and query has academic OBS intent, merge top cosine hits from obs.acibadem.edu.tr only
# into the global candidate pool (helps programme pages rank before main-site noise).
RAG_OBS_VECTOR_PREFILTER = _env_bool("RAG_OBS_VECTOR_PREFILTER", "false")


def rag_source_url_blocklist_substrings() -> tuple[str, ...]:
    """
    Global source URL filters for low-value/noisy pages.
    Extend via RAG_SOURCE_URL_BLOCKLIST_SUBSTRINGS env.
    """
    default_items = (
        "/news",
        "/archive",
        "/events",
        "/event",
        "/haber",
        "/duyuru",
        "/blog",
    )
    raw = os.environ.get("RAG_SOURCE_URL_BLOCKLIST_SUBSTRINGS", "")
    if not raw.strip():
        return default_items
    extra = tuple(x.strip().lower() for x in raw.split(",") if x.strip())
    merged = []
    seen = set()
    for item in default_items + extra:
        key = item.lower()
        if key not in seen:
            seen.add(key)
            merged.append(key)
    return tuple(merged)


def is_rag_source_url_blocked(url: str | None) -> bool:
    u = (url or "").strip().lower()
    if not u:
        return False
    for sub in rag_source_url_blocklist_substrings():
        if sub and sub in u:
            return True
    return False
