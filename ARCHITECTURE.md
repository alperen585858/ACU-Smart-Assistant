# Architecture Overview

This document describes the high-level architecture of the ACU Smart Assistant project.

## System Diagram

```
┌──────────────┐    HTTP     ┌──────────────┐    SQL/pgvector    ┌──────────────────┐
│   Next.js    │ ──────────► │    Django     │ ◄───────────────► │  PostgreSQL 15   │
│  Frontend    │  :3000      │   Backend     │      :5432        │  + pgvector      │
│  (React 19)  │ ◄────────── │   (DRF)       │                   │  + pg_trgm       │
└──────────────┘    JSON     └──────┬───────┘                   └──────────────────┘
                                    │
                                    │ HTTP :11434
                                    ▼
                             ┌──────────────┐
                             │   Ollama /    │
                             │  Claude API   │
                             │   (LLM)       │
                             └──────────────┘
```

## System Boundaries

| Component | Technology | Port | Purpose |
|-----------|-----------|------|---------|
| Frontend | Next.js 16, React 19, TypeScript, Tailwind CSS | 3000 | User-facing chat UI |
| Backend | Django 6.0.3, Python | 8000 | Chat API, RAG orchestration, session management |
| Database | PostgreSQL 15 + pgvector + pg_trgm | 5432 | Relational data + vector search + trigram matching |
| LLM | Ollama (llama3.2:3b) or Claude API | 11434 | Language model inference |

## Database Schema

### Chat Models (`chat/models.py`)

```
ChatSession
├── id: UUID (PK, auto-generated)
├── client_id: UUID (indexed, browser-specific)
├── title: VARCHAR(500), default "New chat"
├── created_at: TIMESTAMP
└── updated_at: TIMESTAMP
    Order: -updated_at

ChatMessage
├── id: UUID (PK, auto-generated)
├── session: FK → ChatSession (CASCADE)
├── role: VARCHAR(20) [user | assistant]
├── content: TEXT
└── created_at: TIMESTAMP
    Order: created_at
```

### RAG Models (`core/models.py`)

```
Page
├── url: URL (unique, max 2000)
├── title: VARCHAR(500)
├── content: TEXT
├── source: VARCHAR(100, indexed)
├── created_at: TIMESTAMP
└── updated_at: TIMESTAMP

DocumentChunk
├── page: FK → Page (CASCADE)
├── chunk_index: INT
├── content: TEXT
├── embedding: VECTOR(384) [HNSW index, cosine ops]
├── source_url: URL(2000)
├── page_title: VARCHAR(500)
├── created_at: TIMESTAMP
└── updated_at: TIMESTAMP
    Unique: (page, chunk_index)
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/chat/` | Send message, receive AI response |
| GET | `/api/chat/sessions/?client_id=UUID` | List user's sessions (max 100) |
| GET | `/api/chat/sessions/{id}/?client_id=UUID` | Load session with messages |
| DELETE | `/api/chat/sessions/{id}/?client_id=UUID` | Delete session and messages |

### Chat Completion Request/Response

```json
// Request
{
  "message": "What faculties are available?",
  "client_id": "uuid-string",
  "session_id": "uuid-string (optional)"
}

// Success Response
{
  "reply": "...",
  "session_id": "uuid",
  "title": "...",
  "rag": {
    "embedding_ok": true,
    "chunks_used": 4,
    "sources": [{"url": "...", "title": "...", "cosine_distance": 0.42}],
    "context_chars_sent": 3200
  }
}

// Error Response (502/504)
{
  "error": "Ollama connection refused",
  "rag": {...}
}
```

## Backend Structure

### HTTP Layer (`chat/`)

- **`completion_views.py`** — Accepts chat POST requests, delegates to `chat_logic`
- **`session_views.py`** — Session list, detail, and delete endpoints
- **`views.py`** — Compatibility re-export for URL routing
- **`urls.py`** — Endpoint routing under `/api/chat/`

### Orchestration Layer

- **`chat_logic.py`** — Main coordinator:
  - `run_chat_completion()` dispatches to stateless or DB-backed paths
  - `_chat_with_db()` handles session creation, title auto-generation, message persistence
  - Rolls back on LLM errors (deletes user message, restores title)
  - History window: `CHAT_HISTORY_MAX_MESSAGES = 12`

### Service Layer

- **`rag_service.py`** — RAG prompt preparation:
  - `prepare_chat_prompts()` returns `(system_prompt, user_prompt, rag_meta)`
  - Smalltalk detection: skips RAG for greetings/thanks (regex-based, max 160 chars)
  - Query enrichment: adds university keywords, detects department/faculty/STEM intent
  - Context wrapping with `===CONTEXT===` / `===QUESTION===` delimiters

- **`llm_service.py`** — LLM provider abstraction:
  - `call_llm()` routes to Ollama or Claude based on `LLM_BACKEND` env var
  - Output sanitization: strips leaked delimiters from LLM replies
  - Timeout handling: 504 for timeouts, 502 for connection errors

- **`message_utils.py`** — Helpers:
  - UUID parsing and validation
  - Message trimming for LLM context windows

### RAG Pipeline (`core/`)

```
User Query
    │
    ▼
┌─────────────────────┐
│ 1. Query Enrichment  │  rag_service.py: compose_rag_search_query()
│    + keyword extract  │  rag_keywords.py: rag_keywords_from_query()
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│ 2. Multi-Embed       │  embeddings.py: embed_texts()
│    Query variants    │  (primary + raw + keyword line)
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│ 3. Vector Search     │  rag_retrieval.py: cosine distance
│    Top-K candidates  │  pgvector HNSW index
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│ 4. Rerank            │  Word overlap + trigram similarity
│    + Intent boost    │  + STEM/department boosting
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│ 5. Threshold Filter  │  RAG_MAX_DISTANCE (0.62)
│    + Relaxed fallback│  RAG_RELAX_ON_EMPTY
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│ 6. Context Assembly  │  RAG_MAX_CHARS (7KB budget)
│    Max 2 chunks/URL  │  RAG_SNIPPET_CHARS (1100/chunk)
└─────────┬───────────┘
          ▼
    Context + Sources
```

**Fallback chain**: If embeddings fail → lexical keyword search over chunks → full page content search.

### RAG Configuration (`core/rag_config.py`)

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `RAG_MAX_CHARS` | 7000 | Max context chars injected into prompt |
| `RAG_TOP_K` | 28 | Vector candidates to retrieve |
| `RAG_MAX_DISTANCE` | 0.62 | Cosine distance relevance threshold |
| `RAG_RELAX_ON_EMPTY` | true | Use below-threshold results if no strict hits |
| `RAG_MAX_CHUNKS_PER_URL` | 2 | Prevent single page monopolizing context |
| `RAG_SNIPPET_CHARS` | 1100 | Max chars per chunk in context |
| `RAG_MULTI_EMBED` | true | Embed multiple query variants |
| `RAG_KEYWORD_BOOST` | true | Boost keyword-matched chunks |
| `RAG_RERANK_OVERLAP_WEIGHT` | 0.06 | Word overlap rerank boost |
| `RAG_LEXICAL_WEIGHT` | 0.12 | Trigram similarity rerank boost |

### Embedding Model

- Model: `BAAI/bge-small-en-v1.5` (384 dimensions)
- Chunking: 700 chars per chunk, 120 chars overlap
- Cached with `@lru_cache` for efficiency
- Configurable via `EMBEDDING_MODEL` env var

## Frontend Structure

### Component Architecture

- **`app/chat/page.tsx`** — Main chat component (client-side rendered)
  - State management via React `useState` hooks
  - Session persistence via `localStorage` (client_id + last session)
  - Auto-scrolling, textarea auto-resize, loading indicators

- **`app/layout.tsx`** — App shell with fonts (Geist Sans/Mono, Syne)
- **`app/page.tsx`** — Entry route, redirects to `/chat`

### TypeScript Types

```typescript
Message { id, role, content, timestamp, rag?, latencySec? }
SessionMeta { id, title, updatedAt }
RagMeta { embedding_ok, chunks_used, sources, context_chars_sent, ... }
RagSource { url, title, cosine_distance }
```

### Features

- Multi-session chat with history sidebar
- RAG source preview (shows which pages were used)
- Latency tracking per message
- Suggested prompts for empty state
- Responsive design (mobile-aware)

## Docker Deployment

```yaml
Services (docker-compose.yml):

  ollama          → ollama/ollama:latest
                    Port: 11434, Volume: ollama_data
                    Auto-pulls model via entrypoint script

  db              → pgvector/pgvector:pg15
                    Port: 5433→5432, Volume: pgdata
                    Health check: pg_isready (5s interval)

  backend         → Python 3.12-slim (./backend/Dockerfile)
                    Port: 8000, depends_on: db (healthy), ollama
                    Command: migrate + runserver

  frontend        → Node.js 22 (./frontend/Dockerfile)
                    Port: 3000, depends_on: backend
                    Hot-reload via WATCHPACK_POLLING

Network: acu_net (bridge)
Volumes: pgdata, ollama_data, frontend_node_modules
```

## Request/Data Flow

```
1. User types message in Next.js chat UI
2. Frontend POSTs to /api/chat/ with client_id, message, session_id
3. completion_views.py → chat_logic.run_chat_completion()
4. chat_logic parses body, loads session history (if DB-backed)
5. rag_service checks for smalltalk bypass
6. rag_service enriches query + calls rag_retrieval
7. rag_retrieval embeds query → vector search → rerank → assemble context
8. chat_logic builds final prompt (system + history + RAG context + user msg)
9. llm_service calls Ollama or Claude API
10. Reply sanitized, persisted to DB, returned with RAG metadata
```

## Security Considerations

| Area | Status | Details |
|------|--------|---------|
| Authentication | Client UUID only | Browser-generated UUID in localStorage, no server auth |
| CORS | Dev-only config | Allows any localhost port via regex |
| CSRF | Disabled on chat API | `@csrf_exempt` on all endpoints |
| Input validation | Basic | UUID format check, message length clamping (900 chars) |
| Output sanitization | Yes | Strips leaked RAG delimiters from LLM output |
| Rate limiting | Not implemented | No request throttling |
| HTTPS | Not configured | Development-only HTTP setup |
| Data isolation | Per client_id | Sessions scoped by UUID, no cross-client access |

## Error Handling

| Scenario | HTTP Status | Response |
|----------|------------|----------|
| Invalid JSON body | 400 | `{error: "Invalid JSON"}` |
| Missing message field | 400 | `{error: "message or messages is required"}` |
| Invalid UUID | 400 | `{error: "session_id is not a valid UUID"}` |
| LLM timeout | 504 | `{error: "...", rag: {...}}` |
| LLM connection refused | 502 | `{error: "...", rag: {...}}` |
| Empty LLM response | 502 | `{error: "Empty model response"}` |
| Embedding model fails | 200 | Falls back to lexical search, `rag.embedding_ok: false` |

## Runtime Configuration

All runtime behavior is controlled via `.env`:

```bash
# LLM Backend
LLM_BACKEND=ollama              # "ollama" or "claude"
OLLAMA_BASE_URL=http://ollama:11434
OLLAMA_MODEL=llama3.2:3b
OLLAMA_NUM_PREDICT=384
OLLAMA_NUM_CTX=8192
OLLAMA_TEMPERATURE=0.15

# Claude (optional)
ANTHROPIC_API_KEY=sk-ant-...
CLAUDE_MODEL=claude-haiku-4-5-20251001

# Database
POSTGRES_USER=acu
POSTGRES_PASSWORD=acu_secret
POSTGRES_DB=acu_chatbot

# RAG Tuning
RAG_TOP_K=28
RAG_MAX_DISTANCE=0.62
RAG_MAX_CHARS=7000

# Frontend
NEXT_PUBLIC_API_URL=http://localhost:8000
```

## Management Commands

| Command | Purpose |
|---------|---------|
| `scrape_acibadem` | Crawl ACU website, store pages in DB |
| `build_page_embeddings` | Generate vector embeddings for pages |
| `refresh_rag` | End-to-end: scrape → embed |
| `rag_stats` | Report corpus metrics |
| `rag_diagnose_coverage` | Diagnose RAG coverage gaps |
| `rag_index_audit` | Audit vector index health |

## Current Trade-offs

- `chat_logic.py` contains both stateless and DB-backed chat paths for API compatibility.
- Client-side UUID authentication is not secure for production; suitable for demo/development.
- CORS configured for localhost only; production deployment needs proper domain configuration.
- Embedding model loaded lazily and cached; long-running processes may accumulate memory.
- HNSW vector index optimized for speed over perfect recall (m=16, ef_construction=64).
