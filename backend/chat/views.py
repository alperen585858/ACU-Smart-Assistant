import json
import os
import socket
import urllib.error
import urllib.request
import uuid

from django.http import JsonResponse
from django.shortcuts import get_object_or_404
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET, require_http_methods

from django.db.models import Q

from .models import ChatMessage, ChatSession
from core.models import Page

LLM_BACKEND = os.environ.get("LLM_BACKEND", "ollama")
RAG_MAX_CHARS = 2000

# Ollama settings
OLLAMA_BASE = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "phi3:mini")
OLLAMA_NUM_PREDICT = int(os.environ.get("OLLAMA_NUM_PREDICT", "96"))
OLLAMA_NUM_CTX = int(os.environ.get("OLLAMA_NUM_CTX", "1024"))
OLLAMA_HTTP_TIMEOUT = int(os.environ.get("OLLAMA_HTTP_TIMEOUT", "120"))

# Claude API settings
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-haiku-4-5-20251001")

SYSTEM_BASE = (
    "You are ACU Smart Assistant for Acıbadem University. "
    "IMPORTANT: Always respond in Turkish only. "
    "Keep answers short: 2-3 sentences maximum. "
    "Use only the provided context to answer. "
    "If you don't know, say 'Bu konuda bilgim yok.'"
)


def _search_pages(query: str) -> str:
    """DB'deki ACU sayfalarında basit keyword arama yap, en alakalı içerikleri döndür."""
    words = query.lower().split()
    if not words:
        return ""
    filters = Q()
    for word in words:
        if len(word) >= 3:
            filters |= Q(content__icontains=word) | Q(title__icontains=word)
    if not filters:
        return ""
    pages = Page.objects.filter(filters)[:3]
    if not pages:
        return ""
    context_parts = []
    total = 0
    for p in pages:
        snippet = p.content[:800]
        if total + len(snippet) > RAG_MAX_CHARS:
            break
        context_parts.append(f"[{p.title}]\n{snippet}")
        total += len(snippet)
    return "\n\n".join(context_parts)


def _build_system_prompt(user_message: str) -> str:
    """Kullanıcı mesajına göre RAG context eklenmiş system prompt oluştur."""
    context = _search_pages(user_message)
    if context:
        return (
            f"{SYSTEM_BASE}\n\n"
            f"Answer the user's question using ONLY the following context. "
            f"Summarize the relevant parts in Turkish.\n\n"
            f"Context:\n{context}"
        )
    return SYSTEM_BASE


def _parse_client_id(raw: str | None):
    if not raw:
        return None
    try:
        return uuid.UUID(str(raw))
    except (ValueError, TypeError):
        return None


def _call_claude(messages: list) -> tuple[str | None, str | None]:
    system_text = ""
    api_messages = []
    for m in messages:
        if m["role"] == "system":
            system_text = m["content"]
        else:
            api_messages.append({"role": m["role"], "content": m["content"]})

    payload = json.dumps({
        "model": CLAUDE_MODEL,
        "max_tokens": 256,
        "system": system_text,
        "messages": api_messages,
    }).encode("utf-8")

    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "x-api-key": ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")
        return None, f"Claude API hatası: {detail}"
    except urllib.error.URLError as e:
        return None, f"Claude API bağlantı hatası: {e.reason}"

    content_blocks = data.get("content", [])
    reply = ""
    for block in content_blocks:
        if block.get("type") == "text":
            reply += block.get("text", "")
    reply = reply.strip()
    if not reply:
        return None, "Claude boş yanıt döndü"
    return reply, None


def _call_ollama(ollama_messages: list) -> tuple[str | None, str | None]:
    payload = json.dumps(
        {
            "model": OLLAMA_MODEL,
            "messages": ollama_messages,
            "stream": False,
            "options": {
                "num_predict": OLLAMA_NUM_PREDICT,
                "num_ctx": OLLAMA_NUM_CTX,
            },
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
            data = json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")
        return None, detail or e.reason
    except urllib.error.URLError as e:
        err = str(e.reason)
        if isinstance(e.reason, TimeoutError) or "timed out" in err.lower():
            return None, (
                "Ollama yanıtı zaman aşımına uğradı. Docker RAM artırın, "
                "OLLAMA_NUM_PREDICT azaltın veya modeli küçültün."
            )
        return None, err
    except socket.timeout:
        return None, "Ollama zaman aşımı (socket). Sunucu veya model çok yavaş."

    reply = (data.get("message") or {}).get("content", "").strip()
    if not reply:
        return None, "Empty model response"
    return reply, None


def _call_llm(messages: list) -> tuple[str | None, str | None]:
    if LLM_BACKEND == "claude" and ANTHROPIC_API_KEY:
        return _call_claude(messages)
    return _call_ollama(messages)


@csrf_exempt
@require_GET
def list_sessions(request):
    cid = _parse_client_id(
        request.GET.get("client_id") or request.headers.get("X-Client-Id")
    )
    if cid is None:
        return JsonResponse({"error": "client_id gerekli (UUID)"}, status=400)
    sessions = ChatSession.objects.filter(client_id=cid)[:100]
    return JsonResponse(
        {
            "sessions": [
                {
                    "id": str(s.id),
                    "title": s.title,
                    "updated_at": s.updated_at.isoformat(),
                }
                for s in sessions
            ]
        }
    )


@csrf_exempt
@require_http_methods(["GET", "DELETE"])
def session_detail(request, pk):
    cid = _parse_client_id(
        request.GET.get("client_id") or request.headers.get("X-Client-Id")
    )
    if cid is None:
        return JsonResponse({"error": "client_id gerekli (query, UUID)"}, status=400)

    session = get_object_or_404(ChatSession, pk=pk, client_id=cid)

    if request.method == "DELETE":
        session.delete()
        return JsonResponse({"ok": True})

    msgs = [
        {
            "id": str(m.id),
            "role": m.role,
            "content": m.content,
            "timestamp": m.created_at.isoformat(),
        }
        for m in session.messages.all()
    ]
    return JsonResponse({"session_id": str(session.id), "title": session.title, "messages": msgs})


@csrf_exempt
@require_http_methods(["POST"])
def chat_completion(request):
    try:
        body = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    client_uuid = _parse_client_id(body.get("client_id"))

    if client_uuid is not None:
        return _chat_with_db(request, body, client_uuid)

    raw_history = body.get("messages")
    user_msg = (body.get("message") or "").strip()
    system_text = _build_system_prompt(user_msg)
    ollama_messages = [{"role": "system", "content": system_text}]

    if isinstance(raw_history, list) and len(raw_history) > 0:
        for item in raw_history[-40:]:
            if not isinstance(item, dict):
                continue
            role = item.get("role")
            content = (item.get("content") or "").strip()
            if role in ("user", "assistant") and content:
                ollama_messages.append({"role": role, "content": content})
        if len(ollama_messages) < 2:
            return JsonResponse(
                {"error": "messages must include at least one user/assistant turn"},
                status=400,
            )
    else:
        message = (body.get("message") or "").strip()
        if not message:
            return JsonResponse({"error": "message or messages is required"}, status=400)
        ollama_messages.append({"role": "user", "content": message})

    reply_text, err = _call_llm(ollama_messages)
    if err:
        status = 504 if "zaman aşımı" in err.lower() or "timeout" in err.lower() else 502
        return JsonResponse({"error": err}, status=status)
    return JsonResponse({"reply": reply_text})


def _chat_with_db(request, body: dict, client_uuid: uuid.UUID) -> JsonResponse:
    message = (body.get("message") or "").strip()
    if not message:
        return JsonResponse({"error": "message is required"}, status=400)

    session_id_raw = body.get("session_id")
    if session_id_raw:
        try:
            sid = uuid.UUID(str(session_id_raw))
        except (ValueError, TypeError):
            return JsonResponse({"error": "session_id geçersiz UUID"}, status=400)
        session = get_object_or_404(ChatSession, pk=sid, client_id=client_uuid)
    else:
        session = ChatSession.objects.create(client_id=client_uuid, title="Yeni sohbet")

    prior = list(session.messages.all())
    system_text = _build_system_prompt(message)
    ollama_messages: list = [{"role": "system", "content": system_text}]
    for m in prior[-39:]:
        if m.role in ("user", "assistant") and m.content.strip():
            ollama_messages.append({"role": m.role, "content": m.content})
    ollama_messages.append({"role": "user", "content": message})

    user_row = ChatMessage.objects.create(
        session=session, role="user", content=message
    )
    title_changed = False
    if session.title == "Yeni sohbet" and len(prior) == 0:
        session.title = message[:197] + ("…" if len(message) > 200 else "")
        session.save(update_fields=["title"])
        title_changed = True

    reply_text, err = _call_llm(ollama_messages)
    if err:
        user_row.delete()
        if title_changed:
            session.title = "Yeni sohbet"
            session.save(update_fields=["title"])
        status = 504 if "zaman aşımı" in err.lower() or "timeout" in err.lower() else 502
        return JsonResponse({"error": err, "session_id": str(session.id)}, status=status)

    ChatMessage.objects.create(session=session, role="assistant", content=reply_text)
    session.save()

    return JsonResponse(
        {
            "reply": reply_text,
            "session_id": str(session.id),
            "title": session.title,
        }
    )
