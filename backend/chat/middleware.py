"""Simple in-memory rate limiter for chat API endpoints."""

import time
from collections import defaultdict

from django.http import JsonResponse


class RateLimitMiddleware:
    """
    Limits requests per IP to chat API endpoints.

    Configurable via Django settings:
        RATE_LIMIT_REQUESTS: max requests per window (default 30)
        RATE_LIMIT_WINDOW: window size in seconds (default 60)
    """

    def __init__(self, get_response):
        self.get_response = get_response
        self.requests = defaultdict(list)

    def __call__(self, request):
        if not request.path.startswith("/api/chat"):
            return self.get_response(request)

        from django.conf import settings

        max_requests = getattr(settings, "RATE_LIMIT_REQUESTS", 30)
        window = getattr(settings, "RATE_LIMIT_WINDOW", 60)

        ip = self._get_client_ip(request)
        now = time.time()

        # Clean old entries
        self.requests[ip] = [t for t in self.requests[ip] if now - t < window]

        if len(self.requests[ip]) >= max_requests:
            return JsonResponse(
                {"error": f"Rate limit exceeded. Max {max_requests} requests per {window}s."},
                status=429,
            )

        self.requests[ip].append(now)
        return self.get_response(request)

    def _get_client_ip(self, request):
        forwarded = request.META.get("HTTP_X_FORWARDED_FOR")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.META.get("REMOTE_ADDR", "unknown")
