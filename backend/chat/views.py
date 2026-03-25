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

from .models import ChatMessage, ChatSession

OLLAMA_BASE = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "phi3:mini")
OLLAMA_NUM_PREDICT = int(os.environ.get("OLLAMA_NUM_PREDICT", "256"))
OLLAMA_NUM_CTX = int(os.environ.get("OLLAMA_NUM_CTX", "2048"))
OLLAMA_HTTP_TIMEOUT = int(os.environ.get("OLLAMA_HTTP_TIMEOUT", "180"))

SYSTEM_TEXT = (
    "Yanıtları kısa ve net tut; gereksiz uzatma, birkaç cümle yeter."
)


def _parse_client_id(raw: str | None):
    if not raw:
        return None
    try:
        return uuid.UUID(str(raw))
    except (ValueError, TypeError):
        return None


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

    system_text = SYSTEM_TEXT
    raw_history = body.get("messages")
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

    reply_text, err = _call_ollama(ollama_messages)
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
    ollama_messages: list = [{"role": "system", "content": SYSTEM_TEXT}]
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

    reply_text, err = _call_ollama(ollama_messages)
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
