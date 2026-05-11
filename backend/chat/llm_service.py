import json
import logging
import os
import re
import socket
import time
import urllib.error
import urllib.request

logger = logging.getLogger("chat.llm")

OLLAMA_BASE = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2:3b")
OLLAMA_NUM_PREDICT = int(os.environ.get("OLLAMA_NUM_PREDICT", "144"))
OLLAMA_NUM_CTX = int(os.environ.get("OLLAMA_NUM_CTX", "8192"))
OLLAMA_HTTP_TIMEOUT = int(os.environ.get("OLLAMA_HTTP_TIMEOUT", "120"))
OLLAMA_KEEP_ALIVE = os.environ.get("OLLAMA_KEEP_ALIVE", "30m")
OLLAMA_TEMPERATURE = float(os.environ.get("OLLAMA_TEMPERATURE", "0.15"))
OLLAMA_TOP_P = float(os.environ.get("OLLAMA_TOP_P", "0.85"))


def _sanitize_assistant_reply(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return t
    t = re.sub(
        r"(?i)according\s+to\s*,?\s*={3}\s*CONTEXT\s*={3}\s*[, ]*",
        "Based on the university website, ",
        t,
    )
    t = re.sub(
        r"(?i)based\s+on\s*,?\s*={3}\s*CONTEXT\s*={3}\s*[, ]*",
        "Based on the university website, ",
        t,
    )
    for leak in ("===CONTEXT===", "===QUESTION===", "===END_QUESTION==="):
        t = t.replace(leak, "")
    t = re.sub(r"(?i)^according\s+to\s*,\s*", "Based on the university website, ", t)
    t = re.sub(r"[ \t]{2,}", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def _call_ollama(
    ollama_messages: list, options: dict | None = None
) -> tuple[str | None, str | None]:
    opt = {
        "num_predict": OLLAMA_NUM_PREDICT,
        "num_ctx": OLLAMA_NUM_CTX,
        "temperature": OLLAMA_TEMPERATURE,
        "top_p": OLLAMA_TOP_P,
    }
    if options:
        opt.update({k: v for k, v in options.items() if v is not None})
    payload = json.dumps(
        {
            "model": OLLAMA_MODEL,
            "messages": ollama_messages,
            "stream": True,
            "keep_alive": OLLAMA_KEEP_ALIVE,
            "options": opt,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        f"{OLLAMA_BASE}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=OLLAMA_HTTP_TIMEOUT) as resp:
            raw = resp.read().decode()
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")
        logger.error("Ollama HTTP %d: %s", e.code, detail[:500])
        return None, "LLM service error. Please try again later."
    except urllib.error.URLError as e:
        err = str(e.reason)
        logger.error("Ollama connection error: %s", err)
        if isinstance(e.reason, TimeoutError) or "timed out" in err.lower():
            return None, "Response timed out. Please try again."
        return None, "LLM service is unreachable. Please try again later."
    except socket.timeout:
        logger.error("Ollama socket timeout")
        return None, "Response timed out. Please try again."

    # Ollama stream mode returns NDJSON chunks; concatenate message parts.
    reply_parts: list[str] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        piece = (obj.get("message") or {}).get("content", "")
        if piece:
            reply_parts.append(piece)

    # Fallback for non-streaming/single-object JSON responses.
    if not reply_parts:
        try:
            data = json.loads(raw)
            piece = (data.get("message") or {}).get("content", "")
            if piece:
                reply_parts.append(piece)
        except json.JSONDecodeError:
            pass

    reply = "".join(reply_parts).strip()
    if not reply:
        return None, "Empty model response"
    return reply, None


def call_llm(
    messages: list, *, ollama_options: dict | None = None
) -> tuple[str | None, str | None]:
    logger.info("LLM call: backend=ollama, messages=%d", len(messages))
    start = time.time()
    reply, err = _call_ollama(messages, options=ollama_options)

    elapsed = round(time.time() - start, 2)
    if err:
        logger.warning("LLM error: backend=ollama, elapsed=%.2fs, error=%s", elapsed, err[:200])
        return reply, err
    if not reply:
        logger.warning("LLM empty response: backend=ollama, elapsed=%.2fs", elapsed)
        return reply, err

    logger.info("LLM reply: backend=ollama, elapsed=%.2fs, chars=%d", elapsed, len(reply))
    return _sanitize_assistant_reply(reply), None
