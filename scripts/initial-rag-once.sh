#!/usr/bin/env sh
# Elle calistir: RAG i refresh_rag ile doldurur.
# Not: docker compose'da varsayilan AUTO_RAG_WHEN_EMPTY=1 iken, bos volume'da
# backend ayaga kalkarken zaten init_rag_if_empty (sınırsız HTTP) calisir; bu
# script coğu zaman gereksiz (AUTO kapatildiginda veya tekrar indekslemek icin).
#
# stack ayakta: docker compose -f docker-compose.prod.yml up -d
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
COMPOSE="docker compose -f docker-compose.prod.yml"
$COMPOSE exec -T backend python manage.py refresh_rag --max-pages 0 --depth -1 --without-obs --without-acibadem-js
