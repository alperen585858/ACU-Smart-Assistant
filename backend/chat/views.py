import json
import os
import urllib.error
import urllib.request

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST

OLLAMA_BASE = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "phi3:mini")


@csrf_exempt
@require_POST
def chat_completion(request):
    try:
        body = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    message = (body.get("message") or "").strip()
    if not message:
        return JsonResponse({"error": "message is required"}, status=400)

    payload = json.dumps(
        {
            "model": OLLAMA_MODEL,
            "messages": [{"role": "user", "content": message}],
            "stream": False,
        }
    ).encode("utf-8")

    req = urllib.request.Request(
        f"{OLLAMA_BASE}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            data = json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")
        return JsonResponse({"error": detail or e.reason}, status=502)
    except urllib.error.URLError as e:
        return JsonResponse({"error": str(e.reason)}, status=502)

    reply = (data.get("message") or {}).get("content", "").strip()
    if not reply:
        return JsonResponse({"error": "Empty model response"}, status=502)
    return JsonResponse({"reply": reply})
