import hashlib
import logging
import os
import threading
import time as _time
from collections import OrderedDict
from typing import Any

from sentence_transformers import SentenceTransformer

logger = logging.getLogger("core.embeddings")

DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
_embedding_model: SentenceTransformer | None = None
_embedding_model_lock = threading.Lock()

# Retry policy: after a load failure, wait _RETRY_COOLDOWN seconds before
# trying again.  After _MAX_RETRIES consecutive failures, give up permanently.
_EMBEDDING_RETRY_COOLDOWN = int(os.environ.get("EMBEDDING_RETRY_COOLDOWN_SECS", "300"))
_EMBEDDING_MAX_RETRIES = int(os.environ.get("EMBEDDING_MAX_RETRIES", "3"))
_embedding_fail_count: int = 0
_embedding_fail_time: float = 0.0

# ── Embedding cache ──────────────────────────────────────────────────────
_EMBED_CACHE_MAX = int(os.environ.get("EMBED_CACHE_MAX", "512"))
_embed_cache: OrderedDict[str, list[float]] = OrderedDict()
_cache_lock = threading.Lock()


def _cache_key(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def get_embedding_model() -> SentenceTransformer:
    """
    Load once, thread-safe.  On failure, retries after a cooldown period
    (default 300 s, env ``EMBEDDING_RETRY_COOLDOWN_SECS``) up to a maximum
    number of consecutive failures (default 3, env ``EMBEDDING_MAX_RETRIES``).
    After the limit is reached the error becomes permanent for this process.
    """
    global _embedding_model, _embedding_fail_count, _embedding_fail_time

    # Fast path — already loaded
    if _embedding_model is not None:
        return _embedding_model

    # Fast path — permanently failed (max retries exhausted)
    if _embedding_fail_count >= _EMBEDDING_MAX_RETRIES:
        raise RuntimeError(
            f"Embedding model permanently unavailable after "
            f"{_embedding_fail_count} failed attempts."
        )

    # In cooldown after a recent failure — skip without acquiring the lock
    if _embedding_fail_count > 0:
        elapsed = _time.monotonic() - _embedding_fail_time
        if elapsed < _EMBEDDING_RETRY_COOLDOWN:
            remaining = int(_EMBEDDING_RETRY_COOLDOWN - elapsed)
            raise RuntimeError(
                f"Embedding model unavailable (retry in {remaining}s, "
                f"attempt {_embedding_fail_count}/{_EMBEDDING_MAX_RETRIES})."
            )
        logger.info(
            "Embedding retry cooldown expired, attempting reload "
            "(attempt %d/%d)...",
            _embedding_fail_count + 1,
            _EMBEDDING_MAX_RETRIES,
        )

    with _embedding_model_lock:
        # Re-check inside lock (another thread may have loaded it)
        if _embedding_model is not None:
            return _embedding_model
        if _embedding_fail_count >= _EMBEDDING_MAX_RETRIES:
            raise RuntimeError(
                f"Embedding model permanently unavailable after "
                f"{_embedding_fail_count} failed attempts."
            )
        model_name = os.environ.get("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
        try:
            model = SentenceTransformer(
                model_name,
                device="cpu",
                model_kwargs={"low_cpu_mem_usage": False},
            )
            _embedding_model = model
            _embedding_fail_count = 0  # reset on success
            logger.info("Embedding model loaded: %s", model_name)
            return _embedding_model
        except Exception:
            _embedding_fail_count += 1
            _embedding_fail_time = _time.monotonic()
            logger.error(
                "Embedding model load failed (attempt %d/%d), "
                "next retry in %ds",
                _embedding_fail_count,
                _EMBEDDING_MAX_RETRIES,
                _EMBEDDING_RETRY_COOLDOWN,
            )
            raise


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []

    results: list[list[float] | None] = [None] * len(texts)
    to_encode: list[str] = []
    to_encode_idx: list[int] = []

    with _cache_lock:
        for i, t in enumerate(texts):
            key = _cache_key(t)
            cached = _embed_cache.get(key)
            if cached is not None:
                _embed_cache.move_to_end(key)
                results[i] = cached
            else:
                to_encode.append(t)
                to_encode_idx.append(i)

    if to_encode:
        model = get_embedding_model()
        vectors = model.encode(to_encode, normalize_embeddings=True)
        with _cache_lock:
            for j, idx in enumerate(to_encode_idx):
                vec = vectors[j].tolist()
                results[idx] = vec
                key = _cache_key(to_encode[j])
                _embed_cache[key] = vec
                if len(_embed_cache) > _EMBED_CACHE_MAX:
                    _embed_cache.popitem(last=False)

    cache_hits = len(texts) - len(to_encode)
    if cache_hits:
        logger.debug("Embedding cache: %d hits, %d misses", cache_hits, len(to_encode))

    failed = sum(1 for r in results if r is None)
    if failed:
        logger.warning("embed_texts: %d/%d texts failed to embed", failed, len(texts))
    return results  # type: ignore[return-value]


def embed_query(text: str) -> list[float]:
    vectors = embed_texts([text.strip()])
    return vectors[0] if vectors else []


# ── Cross-encoder reranker ──────────────────────────────────────────────
_RERANKER_MODEL_NAME = os.environ.get(
    "RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-2-v2"
)
_reranker_lock = threading.Lock()
_reranker_instance = None
_RERANKER_RETRY_COOLDOWN = int(os.environ.get("RERANKER_RETRY_COOLDOWN_SECS", "300"))
_RERANKER_MAX_RETRIES = int(os.environ.get("RERANKER_MAX_RETRIES", "3"))
_reranker_fail_count: int = 0
_reranker_fail_time: float = 0.0


def get_reranker():
    """
    Lazy-load cross-encoder reranker with retry.  Returns ``None`` when
    the model is unavailable (cooldown or permanent failure).
    """
    global _reranker_instance, _reranker_fail_count, _reranker_fail_time

    if _reranker_instance is not None:
        return _reranker_instance
    if _reranker_fail_count >= _RERANKER_MAX_RETRIES:
        return None
    if _reranker_fail_count > 0:
        elapsed = _time.monotonic() - _reranker_fail_time
        if elapsed < _RERANKER_RETRY_COOLDOWN:
            return None
        logger.info(
            "Reranker retry cooldown expired, attempting reload "
            "(attempt %d/%d)...",
            _reranker_fail_count + 1,
            _RERANKER_MAX_RETRIES,
        )

    with _reranker_lock:
        if _reranker_instance is not None:
            return _reranker_instance
        if _reranker_fail_count >= _RERANKER_MAX_RETRIES:
            return None
        try:
            from sentence_transformers import CrossEncoder
            reranker_kwargs: dict[str, Any] = {
                "model_kwargs": {"low_cpu_mem_usage": False},
            }
            _reranker_instance = CrossEncoder(_RERANKER_MODEL_NAME, **reranker_kwargs)
            _reranker_fail_count = 0
            logger.info("Reranker model loaded: %s", _RERANKER_MODEL_NAME)
            return _reranker_instance
        except Exception:
            _reranker_fail_count += 1
            _reranker_fail_time = _time.monotonic()
            logger.warning(
                "Reranker load failed (attempt %d/%d), "
                "next retry in %ds — falling back to lexical rerank",
                _reranker_fail_count,
                _RERANKER_MAX_RETRIES,
                _RERANKER_RETRY_COOLDOWN,
            )
            return None


def rerank_passages(query: str, passages: list[str]) -> list[float]:
    """Score query-passage pairs with cross-encoder. Returns list of scores."""
    reranker = get_reranker()
    if reranker is None or not passages:
        return []
    pairs = [[query, p] for p in passages]
    scores = reranker.predict(pairs)
    return [float(s) for s in scores]
