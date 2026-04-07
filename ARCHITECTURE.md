# Architecture Overview

This document describes the current high-level architecture of the ACU Smart Assistant project.

## System Boundaries

- `frontend/` (Next.js): user-facing chat UI, browser runtime
- `backend/` (Django): chat API, session/message persistence, RAG orchestration
- `PostgreSQL + pgvector`: relational data + vector search over crawled chunks
- Optional `Ollama` / `Claude`: LLM inference backend

## Backend Structure

### HTTP Layer (`backend/chat`)

- `completion_views.py`: accepts chat completion POST requests
- `session_views.py`: session list/detail/delete endpoints
- `views.py`: compatibility export layer for existing URL imports
- `urls.py`: endpoint routing

### Orchestration Layer

- `chat_logic.py`: coordinates request parsing, history handling, RAG prompt prep, LLM call, DB write flow

### Service Layer

- `rag_service.py`:
  - builds RAG query text from user/history
  - handles smalltalk bypass and prompt construction
  - returns `(system_prompt, user_prompt, rag_meta)`
- `llm_service.py`:
  - provider abstraction (`ollama` / `claude`)
  - provider-specific HTTP calls
  - output sanitization for leaked delimiters
- `message_utils.py`:
  - message trimming helpers
  - client UUID parsing

### Data Layer

- `chat/models.py`: `ChatSession`, `ChatMessage`
- `core/models.py`: `DocumentChunk` vectorized crawl fragments
- `core/rag_retrieval.py`: retrieval, rerank, thresholding, context assembly

## Frontend Structure

- `app/chat/`: chat screen and interaction flow
- `app/layout.tsx`, `app/page.tsx`: app shell and entry route
- API base URL is read from `NEXT_PUBLIC_API_URL`

Frontend sends user messages to backend chat endpoints and renders assistant replies with session support.

## Request/Data Flow

1. User submits message in Next.js chat UI.
2. Frontend calls Django chat endpoint.
3. Django parses body and session context (`client_id` / `session_id`).
4. `rag_service` prepares retrieval query and prompt strategy.
5. `core/rag_retrieval` fetches best chunks from pgvector-backed `DocumentChunk`.
6. Prompt + history are assembled in `chat_logic`.
7. `llm_service` calls selected backend (Ollama or Claude).
8. Reply is sanitized and returned.
9. If session-backed, user/assistant messages are persisted.

## RAG Flow (Core)

`core/rag_retrieval.py` applies a multi-step pipeline:

1. Build query variants (composed text + optional keyword line).
2. Embed variants via embedding model.
3. Retrieve vector candidates with cosine distance.
4. Rerank by lexical overlap/trigram signals.
5. Apply distance threshold (or relaxed fallback).
6. Fill context text under character budget.
7. Return context + source metadata to chat layer.

## Runtime Configuration

Main runtime behavior is controlled via `.env`:

- LLM backend and model selection (`LLM_BACKEND`, `OLLAMA_*`, `CLAUDE_*`)
- RAG retrieval parameters (`RAG_TOP_K`, `RAG_MAX_DISTANCE`, etc.)
- DB connection (`POSTGRES_*`)
- Frontend API URL (`NEXT_PUBLIC_API_URL`)

## Notes / Current Trade-offs

- `chat_logic.py` still contains both stateless and DB-backed chat paths; this is intentional for API compatibility.
- A dedicated test module for the new service boundaries is still recommended (`backend/chat/tests_*` split).
