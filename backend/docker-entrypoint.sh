#!/bin/sh
set -e
if [ "${RUN_MIGRATIONS:-}" = "1" ] || [ "${RUN_MIGRATIONS:-}" = "true" ]; then
  python manage.py migrate --noinput
fi
# Ilk canlı: volume'deki DB'de hic chunk yoksa, sunursuz (max-pages=0) HTTP crawl + embedding.
# AUTO_RAG_WHEN_EMPTY=0 ile kapat. Tam OBS/JS/Chrome: host venv (UBUNTU_MAKINE_KURULUM.md).
if [ "${AUTO_RAG_WHEN_EMPTY:-1}" = "1" ] || [ "${AUTO_RAG_WHEN_EMPTY}" = "true" ]; then
  python manage.py init_rag_if_empty
fi
exec "$@"
