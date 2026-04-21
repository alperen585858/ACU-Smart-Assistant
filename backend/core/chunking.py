"""Text chunking for RAG: legacy fixed windows + entity-aware + hierarchical fallback."""
import re


def chunk_text(text: str, chunk_size: int = 700, chunk_overlap: int = 120) -> list[str]:
    """Whitespace-collapsing fixed windows (legacy; kept for compatibility)."""
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


def _split_oversized_unit(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split a long unit without collapsing internal newlines to a single space."""
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    out: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + chunk_size)
        piece = text[start:end]
        if end < n:
            cut = piece.rfind("\n")
            if cut >= chunk_size // 3:
                piece = piece[: cut + 1]
                end = start + cut + 1
            else:
                sp = piece.rfind(" ")
                if sp >= chunk_size // 3:
                    piece = piece[:sp]
                    end = start + sp
        piece = piece.strip()
        if piece:
            out.append(piece)
        if end >= n:
            break
        start = min(n - 1, end - chunk_overlap)
    return out if out else [text[:chunk_size].strip()]


def chunk_content_fallback(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Paragraph-first splitting on plain text (no DOM); preserves newlines within blocks."""
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")
    text = (text or "").strip()
    if not text:
        return []
    paras = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
    if not paras:
        return _split_oversized_unit(text, chunk_size, chunk_overlap)

    out: list[str] = []
    buf = ""
    for p in paras:
        if len(p) > chunk_size:
            if buf:
                out.append(buf)
                buf = ""
            out.extend(_split_oversized_unit(p, chunk_size, chunk_overlap))
            continue
        if not buf:
            buf = p
        elif len(buf) + 2 + len(p) <= chunk_size:
            buf = f"{buf}\n\n{p}"
        else:
            out.append(buf)
            buf = p
    if buf:
        out.append(buf)
    return out if out else _split_oversized_unit(text, chunk_size, chunk_overlap)


def _use_structural_units(page_text: str, units: list[str]) -> bool:
    """Decide whether DOM units cover the page well enough vs. noise-only matches."""
    cl = max(len(page_text.strip()), 1)
    joined = "\n\n".join(units)
    jl = len(joined)
    nu = len(units)
    if nu >= 4:
        return True
    if jl >= 0.32 * cl:
        return True
    if nu >= 2 and jl >= 0.18 * cl:
        return True
    if nu == 1 and jl >= 0.55 * cl:
        return True
    return False


def chunks_for_embedding(
    content: str,
    embedding_units: list[str] | None,
    chunk_size: int = 700,
    chunk_overlap: int = 120,
) -> list[str]:
    """
    Prefer one embedding chunk per DOM record when `embedding_units` is present
    and appears to represent the page; otherwise hierarchical plain-text chunking.
    """
    c_strip = (content or "").strip()
    if not c_strip:
        return []

    units = [str(u).strip() for u in (embedding_units or []) if str(u).strip()]
    if units and _use_structural_units(c_strip, units):
        pieces: list[str] = []
        for u in units:
            pieces.extend(_split_oversized_unit(u, chunk_size, chunk_overlap))
        return [p for p in pieces if p.strip()]

    return chunk_content_fallback(c_strip, chunk_size, chunk_overlap)
