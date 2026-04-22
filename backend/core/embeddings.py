import hashlib
import logging
import os
import threading
from collections import OrderedDict

from sentence_transformers import SentenceTransformer

logger = logging.getLogger("core.embeddings")

DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
_EMBEDDING_LOAD_FAILED = False
_embedding_model: SentenceTransformer | None = None
_embedding_model_lock = threading.Lock()

# ── Embedding cache ──────────────────────────────────────────────────────
_EMBED_CACHE_MAX = int(os.environ.get("EMBED_CACHE_MAX", "512"))
_embed_cache: OrderedDict[str, list[float]] = OrderedDict()
_cache_lock = threading.Lock()


def _cache_key(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def get_embedding_model() -> SentenceTransformer:
    """
    Load once, thread-safe. Prevents concurrent SentenceTransformer() races (meta tensor
    / NotImplementedError on PyTorch 2.2+). Disables low_cpu_mem default meta init path.
    """
    global _embedding_model, _EMBEDDING_LOAD_FAILED
    if _EMBEDDING_LOAD_FAILED:
        raise RuntimeError("Embedding model is unavailable in this environment.")
    if _embedding_model is not None:
        return _embedding_model
    with _embedding_model_lock:
        if _embedding_model is not None:
            return _embedding_model
        model_name = os.environ.get("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
        try:
            # See huggingface/sentence-transformers#3396 — avoid default meta-device load + .to(cpu) failure
            model = SentenceTransformer(
                model_name,
                device="cpu",
                model_kwargs={"low_cpu_mem_usage": False},
            )
            _embedding_model = model
            logger.info("Embedding model loaded: %s", model_name)
            return _embedding_model
        except Exception:
            _EMBEDDING_LOAD_FAILED = True
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
_RERANKER_LOAD_FAILED = False


def get_reranker():
    """Lazy-load cross-encoder reranker (tiny model, ~20MB)."""
    global _reranker_instance, _RERANKER_LOAD_FAILED
    if _RERANKER_LOAD_FAILED:
        return None
    if _reranker_instance is not None:
        return _reranker_instance
    with _reranker_lock:
        if _reranker_instance is not None:
            return _reranker_instance
        try:
            from sentence_transformers import CrossEncoder
            _reranker_instance = CrossEncoder(
                _RERANKER_MODEL_NAME,
                model_kwargs={"low_cpu_mem_usage": False},
            )
            logger.info("Reranker model loaded: %s", _RERANKER_MODEL_NAME)
            return _reranker_instance
        except Exception:
            logger.warning("Reranker model failed to load, falling back to lexical rerank")
            _RERANKER_LOAD_FAILED = True
            return None


def rerank_passages(query: str, passages: list[str]) -> list[float]:
    """Score query-passage pairs with cross-encoder. Returns list of scores."""
    reranker = get_reranker()
    if reranker is None or not passages:
        return []
    pairs = [[query, p] for p in passages]
    scores = reranker.predict(pairs)
    return [float(s) for s in scores]
