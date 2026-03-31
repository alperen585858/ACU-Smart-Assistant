# ACU Smart Assistant

Django API, Next.js chat UI, and RAG over crawled university pages using **PostgreSQL + pgvector** and local embeddings.

## Requirements

- Python 3.12+
- Node.js 20+ (for Next.js)
- PostgreSQL **15+** with the **pgvector** extension (Docker image `pgvector/pgvector:pg15` or a local install with `CREATE EXTENSION vector`)
- Optional: [Ollama](https://ollama.com/) for local LLM inference

## Quick start

1. Clone the repo and enter the project directory.

2. Copy environment template and adjust values (never commit `.env`):

   ```bash
   cp .env.example .env
   ```

3. Install dependencies:

   ```bash
   cd backend && pip install -r requirements.txt
   cd ../frontend && npm install
   ```

4. Run migrations:

   ```bash
   cd ../backend && python manage.py migrate
   ```

5. Load corpus and build embeddings (one command):

   ```bash
   python manage.py refresh_rag --max-pages 60 --depth 2
   ```

   Or inspect index stats:

   ```bash
   python manage.py rag_stats
   ```

6. Start services:

   ```bash
   # Terminal 1 — API
   cd backend && python manage.py runserver

   # Terminal 2 — UI (set NEXT_PUBLIC_API_URL in frontend/.env.local if needed)
   cd frontend && npm run dev
   ```

## Database: two valid setups

### A) Docker Compose (full stack)

From the project root:

```bash
docker compose up -d
```

Use `.env` so that **host** tools (optional `manage.py` on the machine) point at the published DB port (often `5433` → set `POSTGRES_PORT=5433`, `POSTGRES_HOST=localhost`). Containers use `POSTGRES_HOST=db` and port `5432` internally.

### B) Local PostgreSQL + pgvector

Install Postgres and pgvector, create a database and role, then:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

Set `POSTGRES_HOST=localhost` and `POSTGRES_PORT=5432` (or your local port) in `.env`.

## Environment variables

See [`.env.example`](.env.example) for all options: DB credentials, Ollama/Claude, `EMBEDDING_MODEL`, `RAG_TOP_K`, `RAG_MAX_DISTANCE`, and frontend `NEXT_PUBLIC_*` URLs.

## What not to commit

- `.env` (secrets and machine-specific ports)
- Python `venv` / `.venv`
- `frontend/node_modules`, `frontend/.next`
- `backend/db.sqlite3` (only used if Postgres env vars are unset)

## License

Add your license here if you publish the repository publicly.
