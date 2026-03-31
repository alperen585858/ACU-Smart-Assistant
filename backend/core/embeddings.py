import os
from functools import lru_cache

from sentence_transformers import SentenceTransformer


DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


def chunk_text(text: str, chunk_size: int = 700, chunk_overlap: int = 120) -> list[str]:
    clean = " ".join((text or "").split())
    if not clean:
        return []
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    chunks: list[str] = []
    start = 0
    step = chunk_size - chunk_overlap
    while start < len(clean):
        end = min(len(clean), start + chunk_size)
        piece = clean[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(clean):
            break
        start += step
    return chunks


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    model_name = os.environ.get("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
    return SentenceTransformer(model_name)


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    model = get_embedding_model()
    vectors = model.encode(texts, normalize_embeddings=True)
    return [vec.tolist() for vec in vectors]


def embed_query(text: str) -> list[float]:
    vectors = embed_texts([text.strip()])
    return vectors[0] if vectors else []
