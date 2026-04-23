# İlk deploy, volume ve `docker compose up`

Hedef: **Veri** (PostgreSQL + RAG) **`pgdata_prod`** volume’ünde; **Ollama** **`ollama_data_prod`**. Aynı volume kaldıkça indeks kalır.

## `docker compose up` ne yapıyor?

1. **Migrasyon (isteğe bağlı):** `RUN_MIGRATIONS=1` iken [docker-entrypoint.sh](../backend/docker-entrypoint.sh) migration çalıştırır.
2. **Boş RAG → otomatik tarama (varsayılan açık):** `DocumentChunk` yokken [`init_rag_if_empty`](../backend/core/management/commands/init_rag_if_empty.py) → `refresh_rag --max-pages 0 --depth -1` (sınırsız sayfa, sınırsız derinlik) + **headless (OBS/JS yok)**, sadece HTTP crawl + embedding. `AUTO_RAG_WHEN_EMPTY=0` ile kapatılır.
3. Gunicorn başlar.

**Neden OBS/JS yok?** [Dockerfile.prod](../backend/Dockerfile.prod) imajında **Chrome yok**; sınırsız tarama sadece `requests` ile yapılabilir. Tam kapsam (OBS Bologna, academic staff JS) için host’da venv: `README` / `UBUNTU_MAKINE_KURULUM.md`.

**Sonraki açılış / deploy:** `DocumentChunk` zaten vardır → `init_rag_if_empty` **anında çıkar**; sadece kod/imaj + volume’daki veri (yeniden tarama yok). GitHub [deploy-vmware-ssh.yml](../.github/workflows/deploy-vmware-ssh.yml) da aynı mantık: compose **yeniden sadece tarama koşturmaz**; volume durduğu sürece indeks kalır.

**İndeksi zorla yenilemek:** Sunucuda manuel:
`docker compose -f docker-compose.prod.yml exec backend python manage.py refresh_rag ...`

| Ortam değişkeni | Anlam |
|-----------------|--------|
| `RUN_MIGRATIONS` | `1` ise migrate (ilk kurulumda gerek) |
| `AUTO_RAG_WHEN_EMPTY` | `1` (varsayılan) indeks yokken otomatik RAG; `0` sadece elle |

## Veri silinmesin diye

- `docker compose down` **-v yok** → volume’ler durur.
- `docker compose down -v` veya volume’ü silmek **tüm DB’yi siler**; tekrar `up` açıldığında indeks boş olur, **otomatik RAG yine (AUTO_RAG_WHEN_EMPTY=1) uzun süre çalışabilir**.

## İlk sunucu komutu (özet)

```bash
cd /path/to/Project
RUN_MIGRATIONS=1 docker compose -f docker-compose.prod.yml up -d --build
```

Gerekirse: `docker compose -f docker-compose.prod.yml logs -f backend` (ilk indeks inşa süresi uzun olabilir; healthcheck gunicorn ayağa kalkana kadar bekler).

`scripts/initial-rag-once.sh` yalnızca **AUTO_RAG_WHEN_EMPTY=0** yaptıysan veya aynı `refresh_rag`’i elle tekrar istiyorsan anlamlı.
