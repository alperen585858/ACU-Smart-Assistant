"""Health check endpoint for monitoring and Docker health checks."""

import time

from django.db import connection
from django.http import JsonResponse


def health_check(request):
    """Returns service health status including DB connectivity."""
    checks = {}

    # Database check
    try:
        start = time.time()
        with connection.cursor() as cursor:
            cursor.execute("SELECT 1")
        checks["database"] = {"status": "ok", "latency_ms": round((time.time() - start) * 1000, 1)}
    except Exception as e:
        checks["database"] = {"status": "error", "detail": str(e)}

    all_ok = all(c["status"] == "ok" for c in checks.values())
    return JsonResponse(
        {"status": "healthy" if all_ok else "unhealthy", "checks": checks},
        status=200 if all_ok else 503,
    )
