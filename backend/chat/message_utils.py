import os
import uuid

CHAT_MESSAGE_MAX_CHARS = max(200, int(os.environ.get("CHAT_MESSAGE_MAX_CHARS", "900")))


def parse_client_id(raw: str | None):
    if not raw:
        return None
    try:
        return uuid.UUID(str(raw))
    except (ValueError, TypeError):
        return None


def trim_message_for_llm(text: str, max_chars: int = CHAT_MESSAGE_MAX_CHARS) -> str:
    t = (text or "").strip()
    if len(t) <= max_chars:
        return t
    return t[: max_chars - 1] + "…"


def trim_last_user_for_llm(content: str, rag_user_bubble_max_chars: int) -> str:
    if "===CONTEXT===" in content:
        if len(content) > rag_user_bubble_max_chars + 500:
            return content[: rag_user_bubble_max_chars + 499] + "…"
        return content
    return trim_message_for_llm(content)
