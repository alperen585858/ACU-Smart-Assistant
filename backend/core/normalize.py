import hashlib
import json
import os
import re
import threading
import urllib.error
import urllib.request
from collections import OrderedDict


_TR_CHARS_RE = re.compile(r"[çğıöşüÇĞİÖŞÜ]")
_ASCII_LETTER_RE = re.compile(r"[A-Za-z]")
_WORD_RE = re.compile(r"[A-Za-zçğıöşüÇĞİÖŞÜ]{2,}")


def is_probably_turkish(text: str) -> bool:
    """
    Heuristic Turkish detection intended for short web chunks.
    Avoids adding external deps (langdetect) and is robust to mixed pages.
    """
    t = (text or "").strip()
    if not t:
        return False

    # Strong signal: Turkish-specific characters.
    if _TR_CHARS_RE.search(t):
        return True

    # Weak signal: common TR function words.
    # (Helps with Turkish content that happens to be ASCII-only.)
    low = t.casefold()
    tr_hits = 0
    for w in (" ve ", " için", " ile ", " olarak", " veya ", " değil", " bu ", " şu ", " bir "):
        if w in low:
            tr_hits += 1
            if tr_hits >= 2:
                return True

    # If it looks purely ASCII English-ish, treat as non-TR.
    if _ASCII_LETTER_RE.search(t) and not _TR_CHARS_RE.search(t):
        return False

    # Fallback: if there are no clear letters, don't translate.
    words = _WORD_RE.findall(t)
    return bool(words) and tr_hits >= 1


# ── Translation (best-effort, optional) ──────────────────────────────────
_TRANS_CACHE_MAX = int(os.environ.get("TRANS_CACHE_MAX", "512"))
_trans_cache: OrderedDict[str, str] = OrderedDict()
_trans_lock = threading.Lock()


def _cache_key(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def _ollama_translate_to_english(text: str) -> str | None:
    """
    Best-effort translation using the project's existing Ollama service (if available).
    Returns translated string, or None on failure/unavailable.
    """
    base = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
    model = os.environ.get("OLLAMA_TRANSLATE_MODEL") or os.environ.get("OLLAMA_MODEL", "llama3.2:3b")
    timeout_s = int(os.environ.get("OLLAMA_TRANSLATE_TIMEOUT", "45"))
    num_predict = int(os.environ.get("OLLAMA_TRANSLATE_NUM_PREDICT", "256"))

    system = (
        "You are a translation engine. Translate the user text to English.\n"
        "Rules: Output ONLY the English translation. Do not add explanations. "
        "Preserve URLs, names, and numbers. Keep formatting reasonable."
    )
    payload = json.dumps(
        {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": (text or "").strip()[:6000]},
            ],
            "stream": False,
            "options": {"temperature": 0.0, "top_p": 1.0, "num_predict": num_predict},
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        f"{base}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError):
        return None
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        return None
    msg = (obj.get("message") or {}).get("content", "")
    out = (msg or "").strip()
    if not out:
        return None
    return out


def translate_to_english(text: str) -> str:
    """
    Translate text to English (best-effort). If translation backend is unavailable,
    returns original text unchanged.
    """
    t = (text or "").strip()
    if not t:
        return ""

    enabled = os.environ.get("RAG_TRANSLATE_TR_CHUNKS", "1").strip().lower() not in ("0", "false", "no")
    if not enabled:
        return t

    key = _cache_key(t)
    with _trans_lock:
        cached = _trans_cache.get(key)
        if cached is not None:
            _trans_cache.move_to_end(key)
            return cached

    translated = _ollama_translate_to_english(t)
    out = translated if translated is not None else t

    with _trans_lock:
        _trans_cache[key] = out
        if len(_trans_cache) > _TRANS_CACHE_MAX:
            _trans_cache.popitem(last=False)
    return out


def normalize_for_embedding(text: str) -> str:
    """
    Normalize chunk text before embedding.
    - English chunks: unchanged
    - Turkish chunks: translate to English (index-time)
    """
    t = (text or "").strip()
    if not t:
        return ""
    return translate_to_english(t) if is_probably_turkish(t) else t

