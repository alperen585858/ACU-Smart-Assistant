# ACU Smart Assistant

ACU Smart Assistant; `Django` backend, `Next.js` frontend ve `PostgreSQL + pgvector` tabanli RAG (Retrieval-Augmented Generation) kullanan bir universite sohbet asistanidir.  
Kaynak icerik `acibadem.edu.tr` sayfalarindan cekilir, parcalanir, embedding uretilir ve sohbet yanitlari bu baglamla desteklenir.

## Icerik

- [Proje Ozeti](#proje-ozeti)
- [Teknoloji Yigini](#teknoloji-yigini)
- [Dizin Yapisi](#dizin-yapisi)
- [Hizli Baslangic Docker Compose](#hizli-baslangic-docker-compose)
- [Docker Olmadan Yerel Gelistirme](#docker-olmadan-yerel-gelistirme)
- [RAG Veri Akisi ve Komutlar](#rag-veri-akisi-ve-komutlar)
- [API Ozeti](#api-ozeti)
- [Ortam Degiskenleri](#ortam-degiskenleri)
- [CI/CD ve Deploy](#cicd-ve-deploy)
- [Sorun Giderme](#sorun-giderme)

## Proje Ozeti

- **Backend (`backend`)**: Django REST API, sohbet oturum yonetimi, RAG retrieval, LLM cagri katmani.
- **Frontend (`frontend`)**: Next.js tabanli chat arayuzu, session gecmisi, RAG meta gorunumu.
- **Veritabani**: PostgreSQL + `pgvector` (`VectorField(384)`).
- **LLM**: Varsayilan `Ollama`, opsiyonel `Claude` (Anthropic API).
- **Dokuman kaynagi**: `acibadem.edu.tr` alanindaki sayfalar (`/en` odakli crawl).

## Teknoloji Yigini

- Python 3.12+
- Django 6
- PostgreSQL 15 + pgvector
- Node.js 22 (frontend ve CI ile uyumlu)
- Next.js 16
- Docker + Docker Compose v2

## Dizin Yapisi

```text
Project/
  backend/                 Django API + RAG + management commands
  frontend/                Next.js chat UI
  nginx/                   Reverse proxy config
  scripts/                 Yardimci scriptler (ollama entrypoint vb.)
  deploy/                  Uretim/deploy rehber ve policy taslaklari
  docker-compose.yml       Gelistirme stack
  docker-compose.prod.yml  Sunucuda source'tan production build
  docker-compose.ec2.yml   ECR imajlari ile production calisma
  DEPLOYMENT.md            Ayrintili yayina alma rehberi
```

## Hizli Baslangic Docker Compose

En az kurulum maliyetiyle tum servisi ayaga kaldirmak icin:

1. Proje kokunde `.env` olustur:

```bash
cp .env.example .env
```

2. Servisleri baslat:

```bash
docker compose up -d --build
```

3. Ilk RAG indeksini olustur (`backend` konteynerinde):

```bash
docker compose exec backend python manage.py refresh_rag --max-pages 60 --depth 2
```

4. Erisim noktalarini ac:

- Uygulama (Nginx): `http://localhost:8080` (veya `NGINX_HOST_PORT`)
- Backend health: `http://localhost:8000/api/health/` (veya `BACKEND_HOST_PORT`)
- API dokumantasyonu: `http://localhost:8000/api/docs/`

Notlar:

- Varsayilan: Ollama Mac uygulamasinda calisir; `docker compose` Ollama konteyneri baslatmaz. Backend konteyneri `OLLAMA_DOCKER_URL` (varsayilan `http://host.docker.internal:11434`) uzerinden host Ollama’ya baglanir. Docker ici Ollama icin: `OLLAMA_DOCKER_URL=http://ollama:11434` ve `docker compose --profile docker-ollama up -d`.
- Host tarafinda `manage.py` kosacaksan `POSTGRES_HOST=localhost`, hosta acilan portu ve `OLLAMA_BASE_URL=http://127.0.0.1:11434` kullan.

## Docker Olmadan Yerel Gelistirme

Bu modda backend ve frontend hostta calisir; PostgreSQL+pgvector zorunludur.

1. Veritabani hazirla:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

2. Koke `.env` kopyala ve duzenle:

```bash
cp .env.example .env
```

3. Backend:

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python manage.py migrate
python manage.py runserver
```

4. Frontend:

```bash
cd frontend
npm install
echo "NEXT_PUBLIC_API_URL=http://localhost:8000" > .env.local
npm run dev
```

5. Ilk RAG verisini yukle:

```bash
cd backend
python manage.py refresh_rag --max-pages 60 --depth 2
```

## RAG Veri Akisi ve Komutlar

### Veri modeli

- `core.Page`: crawl edilen sayfa
- `core.DocumentChunk`: sayfa parcasi + embedding (`dim=384`)

### End-to-end komut (onerilen)

```bash
cd backend
python manage.py refresh_rag --max-pages 60 --depth 2
```

`refresh_rag` varsayilan davranisi:

- Once mevcut `Page` ve `DocumentChunk` kayitlarini temizler
- Ardindan `scrape_acibadem` calistirir
- Sonra `build_page_embeddings` ile embeddingleri yazar

Eski kayitlari korumak icin:

```bash
python manage.py refresh_rag --keep-existing
```

### Ayrik calisma (manuel kontrol)

Sadece crawl:

```bash
python manage.py scrape_acibadem --crawl --max-pages 40 --depth 2 --delay 1.5
```

Sadece embedding:

```bash
python manage.py build_page_embeddings --batch-size 16 --chunk-size 700 --chunk-overlap 120
```

### Diagnostik komutlar

- `python manage.py rag_stats` -> sayfa/chunk kapsami
- `python manage.py rag_index_audit` -> index kapsam ve defaultlarin ozeti
- `python manage.py rag_diagnose_coverage --top-n 120` -> ornek sorularda retrieval kapsami

## API Ozeti

Taban path: `/api`

### Health

- `GET /api/health/` -> DB baglantisi dahil servis sagligi

### Chat

- `POST /api/chat/`
  - Stateful mod: `client_id` (+ opsiyonel `session_id`) ile oturumlu konusma
  - Stateless mod: `messages` listesi ile tek istekte gecmis gonderimi
- `GET /api/chat/sessions/?client_id=<uuid>` -> kullanicinin oturum listesi
- `GET /api/chat/sessions/<session_uuid>/?client_id=<uuid>` -> oturum detaylari
- `DELETE /api/chat/sessions/<session_uuid>/?client_id=<uuid>` -> oturum silme

### API Dokumantasyon

- OpenAPI schema: `/api/schema/`
- Swagger UI: `/api/docs/`

## Ortam Degiskenleri

Tum degiskenler icin referans dosya: [`.env.example`](.env.example)

En kritikler:

- Veritabani: `POSTGRES_DB`, `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_HOST`, `POSTGRES_PORT`
- LLM secimi: `LLM_BACKEND=ollama|claude`
- Ollama: `OLLAMA_BASE_URL`, `OLLAMA_MODEL`, `OLLAMA_NUM_CTX`, `OLLAMA_HTTP_TIMEOUT`
- Claude: `ANTHROPIC_API_KEY`, `CLAUDE_MODEL`
- RAG ayarlari: `RAG_MAX_CHARS`, `RAG_TOP_K`, `RAG_MAX_DISTANCE`, `RAG_RELAX_ON_EMPTY`
- Frontend: `NEXT_PUBLIC_API_URL`
- Rate limit: `RATE_LIMIT_REQUESTS`, `RATE_LIMIT_WINDOW`

Onemli:

- Django ayarlari `.env` dosyasini proje kokunden (`Project/.env`) yukler.
- Uretimde `SECRET_KEY` zorunludur.

## CI/CD ve Deploy

### CI

`/.github/workflows/ci.yml`:

- Backend: `ruff check` + `python -m unittest core.tests -v`
- Frontend: `npm ci` + `npm run lint`

### Otomatik EC2 deploy (sunucuda build)

`/.github/workflows/deploy-ec2-simple.yml`:

- `main` push sonrasi CI basariliysa SSH ile EC2'ye baglanir
- Sunucuda `git fetch/reset` + `docker compose -f docker-compose.prod.yml up -d --build`

### AWS ECR deploy (manuel tetikleme)

`/.github/workflows/deploy-aws.yml`:

- Backend/frontend production imajlarini build edip ECR'a push eder
- Opsiyonel olarak EC2'de `docker-compose.ec2.yml` ile imajlari cekip restart eder
- Opsiyonel ECS rolling restart adimlari vardir

Daha ayrintili operasyon bilgisi icin: [`DEPLOYMENT.md`](DEPLOYMENT.md) ve [`deploy/README.md`](deploy/README.md)

## Sorun Giderme

- **`relation ... does not exist`**: `python manage.py migrate` calistir.
- **`vector extension` hatasi**: DB'de `CREATE EXTENSION vector;` eksik.
- **Frontend backend'e baglanamiyor**: `NEXT_PUBLIC_API_URL` degerini ve backend portunu kontrol et.
- **Chat cevaplari zayif/alakasiz**: `refresh_rag` calistir, sonra `rag_stats` ve `rag_diagnose_coverage` ile kapsami kontrol et.
- **Ollama ulasilamiyor**: `OLLAMA_BASE_URL` ve Ollama servisinin aktifligini dogrula.
- **413 Request body too large**: mesaj boyutu limitlerini (`MAX_REQUEST_BODY_BYTES`, `MAX_MESSAGE_LENGTH`) gozden gecir.

## Guvenlik ve Repo Hijyeni

- `.env` ve gizli anahtarlari repoya commit etme.
- Yerel artifactleri commit etme: `.venv`, `node_modules`, `.next`, loglar, cache klasorleri.
- Docker build context'ini kucuk tutmak icin `backend/.dockerignore` ve `frontend/.dockerignore` dosyalari kullanilir.
