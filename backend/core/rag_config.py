"""RAG tuning from environment (shared by chat views and retrieval)."""

import os
from functools import lru_cache


def _env_bool(key: str, default: str = "true") -> bool:
    return os.environ.get(key, default).lower() in ("1", "true", "yes")


RAG_MAX_CHARS = max(800, int(os.environ.get("RAG_MAX_CHARS", "2000")))
RAG_TOP_K = int(os.environ.get("RAG_TOP_K", "12"))
RAG_VECTOR_FILL_EXTRA = max(0, int(os.environ.get("RAG_VECTOR_FILL_EXTRA", "6")))
# Cap chunks per URL in one prompt (lower = less duplicate pages eating the char budget).
RAG_MAX_CHUNKS_PER_URL = max(1, int(os.environ.get("RAG_MAX_CHUNKS_PER_URL", "3")))
RAG_MAX_DISTANCE = float(os.environ.get("RAG_MAX_DISTANCE", "0.68"))
RAG_RELAX_ON_EMPTY = _env_bool("RAG_RELAX_ON_EMPTY", "true")
RAG_KEYWORD_BOOST = _env_bool("RAG_KEYWORD_BOOST", "true")
RAG_SNIPPET_CHARS = min(
    max(400, int(os.environ.get("RAG_SNIPPET_CHARS", "1100"))),
    RAG_MAX_CHARS,
)

# Recall-oriented (multi-embed, wide pool, rerank, optional lexical on Postgres)
RAG_MULTI_EMBED = _env_bool("RAG_MULTI_EMBED", "false")
RAG_MULTI_EMBED_KEYWORD_LINE = _env_bool("RAG_MULTI_EMBED_KEYWORD_LINE", "false")
RAG_VECTOR_CANDIDATE_POOL = max(20, min(500, int(os.environ.get("RAG_VECTOR_CANDIDATE_POOL", "56"))))
RAG_RERANK_OVERLAP_WEIGHT = float(os.environ.get("RAG_RERANK_OVERLAP_WEIGHT", "0.06"))
RAG_LEXICAL_WEIGHT = float(os.environ.get("RAG_LEXICAL_WEIGHT", "0.12"))

# Added to effective cosine distance when sorting context (STEM queries); pushes news/events pages down.
RAG_STEM_NOISE_URL_PENALTY = float(os.environ.get("RAG_STEM_NOISE_URL_PENALTY", "0.14"))

# Cross-encoder reranking: extra quality, notable CPU/latency on every chat (default off for speed)
RAG_CROSS_ENCODER_RERANK = _env_bool("RAG_CROSS_ENCODER_RERANK", "false")
RAG_CROSS_ENCODER_WEIGHT = float(os.environ.get("RAG_CROSS_ENCODER_WEIGHT", "0.4"))

# Asymmetric retrieval: "Who is {name}?" / "{name} kimdir" — add role-phrase vectors + name icontains boost (no re-embed of corpus)
RAG_WHOIS_QUERY_EXPAND = _env_bool("RAG_WHOIS_QUERY_EXPAND", "true")
RAG_WHOIS_EXTRA_EMBEDS = max(0, min(4, int(os.environ.get("RAG_WHOIS_EXTRA_EMBEDS", "3"))))

# Substrings: if source_url contains one, the chunk is excluded from RAG. Extra: RAG_SOURCE_URL_BLOCKLIST (comma-separated).
# RAG_DEFAULT_SOURCE_BLOCKLIST=false disables built-in defaults so only your list applies.
# Built-in: only the English "Tuition Fees and Scholarships" useful-info page
# (one canonical path — matches http/https, with or without www).
# https://acibadem.edu.tr/en/international-office/international-students/useful-information/tuition-fees-and-scholarships
_DEFAULT_BLOCKED_URL_SUBSTRINGS: tuple[str, ...] = (
    "/en/international-office/international-students/useful-information/tuition-fees-and-scholarships",
)


@lru_cache(maxsize=1)
def rag_source_url_blocklist_substrings() -> tuple[str, ...]:
    extra = (os.environ.get("RAG_SOURCE_URL_BLOCKLIST") or "").strip()
    extra_parts = [x.strip() for x in extra.split(",") if x.strip()]
    if not _env_bool("RAG_DEFAULT_SOURCE_BLOCKLIST", "true"):
        return tuple(extra_parts) if extra_parts else ()
    return tuple(
        dict.fromkeys([*list(_DEFAULT_BLOCKED_URL_SUBSTRINGS), *extra_parts])
    )


def is_rag_source_url_blocked(url: str | None) -> bool:
    u = (url or "").casefold()
    if not u:
        return False
    for sub in rag_source_url_blocklist_substrings():
        if sub.casefold() in u:
            return True
    return False
