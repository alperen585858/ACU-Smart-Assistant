import json
import os

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

from .chat_logic import run_chat_completion

MAX_BODY_SIZE = int(os.environ.get("MAX_REQUEST_BODY_BYTES", "32768"))  # 32 KB
MAX_MESSAGE_LENGTH = int(os.environ.get("MAX_MESSAGE_LENGTH", "5000"))


@csrf_exempt
@require_http_methods(["POST"])
def chat_completion(request):
    if len(request.body) > MAX_BODY_SIZE:
        return JsonResponse(
            {"error": f"Request body too large. Max {MAX_BODY_SIZE} bytes."},
            status=413,
        )

    try:
        body = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    if not isinstance(body, dict):
        return JsonResponse({"error": "Request body must be a JSON object."}, status=400)

    message = body.get("message", "")
    if isinstance(message, str) and len(message) > MAX_MESSAGE_LENGTH:
        return JsonResponse(
            {"error": f"Message too long. Max {MAX_MESSAGE_LENGTH} characters."},
            status=400,
        )

    messages = body.get("messages")
    if isinstance(messages, list):
        for item in messages:
            if isinstance(item, dict):
                content = item.get("content", "")
                if isinstance(content, str) and len(content) > MAX_MESSAGE_LENGTH:
                    return JsonResponse(
                        {"error": f"Message in history too long. Max {MAX_MESSAGE_LENGTH} characters."},
                        status=400,
                    )

    return run_chat_completion(body)
