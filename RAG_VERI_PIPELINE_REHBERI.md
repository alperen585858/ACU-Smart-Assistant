# RAG veri pipeline’ı: crawl, embedding ve doğrulama

Bu rehber, sohbet asistanının **hangi veriyi** kullandığını ve komutları **hangi sırada** çalıştırmanız gerektiğini açıklar. Amaç: sadece siteyi taramak yetmez; RAG **vektör indeksinin** güncel olması gerekir.

## 1. İki ayrı katman

| Katman | Django modeli | Ne işe yarar |
|--------|---------------|--------------|
| Ham metin | `Page` | Crawl ile çekilen URL, başlık, sayfa metni. Sohbet doğrudan bunu aramaz. |
| Arama indeksi | `DocumentChunk` | `Page` metninin parçaları + **embedding** vektörü. `search_document_chunks` buraya bakar. |

**Kural:** `scrape_acibadem` yalnızca `Page`’i günceller. RAG’in cevabı değişsin istiyorsanız, uygun zamanda **`build_page_embeddings`** (veya aşağıdaki `refresh_rag`) çalışmalıdır.

## 2. Önerilen yol: tek komut (uçtan uca)

Proje kökünde `.env` ayarlı, `backend` dizininde sanal ortam açıkken:

```bash
cd backend
python manage.py refresh_rag --max-pages 60 --depth 2
```

Bu komut sırayla:

1. İsteğe bağlı **mevcut RAG satırlarını temizler** (varsayılan davranış — aşağıya bakın).
2. `scrape_acibadem` ile crawl yapar.
3. `build_page_embeddings` ile tüm `Page` satırları için chunk + embedding üretir.

### Önemli: `refresh_rag` varsayılanı veri siler

**Varsayılan:** `refresh_rag`, çalışmadan önce tüm `DocumentChunk` ve **tüm `Page`** kayıtlarını siler. Yalnızca seed/crawl ile yeniden doldurulabilen bir korpusu tasarladıysanız uygundur.

Mevcut `Page` / chunk’ları koruyup sadece pipeline’ı çalıştırmak için:

```bash
python manage.py refresh_rag --keep-existing --max-pages 60 --depth 2
```

`--keep-existing` kullanırken crawl yine yeni/update sayfaları `update_or_create` ile yazar; ardından embedding tüm sayfalar için yeniden hesaplanır.

## 3. Yeni manuel link (seed) ekleyince — doğru RAG için komut sırası

`scrape_acibadem` yalnızca **`DEFAULT_SEEDS`** listesindeki URL’leri (ve isteğe bağlı crawl ile bulduğu linkleri) çeker. Yeni bir sayfayı korpusa almak için önce bu listeye eklemeniz gerekir; dosya: `backend/core/management/commands/scrape_acibadem.py` içindeki `DEFAULT_SEEDS`.

### 3.1. Kod değişikliği

1. Tam URL’yi `DEFAULT_SEEDS` tuple’ına ekleyin (tercihen `https://www.acibadem.edu.tr/en/...` gibi sitedeki gerçek adres).
2. **İngilizce içerik** kullanıyorsanız yolun `/en` veya `/en/...` ile başlaması gerekir; aksi halde komut varsayılan olarak sayfayı **atlar** (normalize boş döner).
3. **Yalnızca Türkçe** bir sayfa ekliyorsanız, aşağıdaki scrape komutuna `--allow-non-english` ekleyin.

### 3.2. Çalıştırılacak komutlar (sıra bu şekilde olmalı)

```bash
cd backend

# A) Sadece seed listesini çekmek (crawl kapalı — yeni link + diğer seed’ler indirilir)
python manage.py scrape_acibadem --delay 1.5

# B) Seed’den çıkan linkleri de takip etmek isterseniz
# python manage.py scrape_acibadem --crawl --max-pages 60 --depth 2 --delay 1.5

# C) RAG için zorunlu: Page → DocumentChunk + embedding
python manage.py build_page_embeddings --batch-size 16

# D) İsteğe bağlı doğrulama
python manage.py rag_stats
```

**Özet sıra:** seed’e URL ekle → **`scrape_acibadem`** → **`build_page_embeddings`** → (isteğe bağlı) **`rag_stats`**.

### 3.3. Tek alternatif: mevcut veriyi silmeden uçtan uca

Seed’i dosyaya ekledikten sonra, elle iki komut yerine:

```bash
cd backend
python manage.py refresh_rag --keep-existing --max-pages 60 --depth 2
```

`--keep-existing` olduğu için mevcut `Page` / chunk’lar silinmez; crawl yeni URL’yi `update_or_create` ile yazar, ardından tüm sayfalar için embedding yenilenir.

### 3.4. Dikkat edilecekler

- **Sadece seed’e yazıp komut çalıştırmamak:** Link DB’ye hiç inmez; RAG görmez.
- **Scrape yapıp embedding atlamak:** `Page` dolu kalır, sohbet yine eski indekse bakar — **`build_page_embeddings` şart**.
- **`--max-pages`:** `--crawl` kullanırken varsayılan **40**; erişemediğiniz sayfalar varsa değeri artırın (`--max-pages 80` gibi). Sadece seed çekiyorsanız kuyruk seed sırasıyla işlenir; çok fazla seed’iniz varsa ve limit düşükse bazı seed’ler sıraya göre **alınmayabilir** — gerekirse `max-pages` yükseltin veya yeni linki listeye **daha üst sıraya** koyun.
- Chat’in bağlı olduğu veritabanı ile bu komutların kullandığı **aynı `.env` / DB** olmalıdır.

## 4. Manuel akış (adım adım)

Crawl ve embedding’i ayrı kontrol etmek istediğinizde:

```bash
cd backend

# 1) Sayfaları çek (örnek: crawl açık, sınır ve gecikme ile)
python manage.py scrape_acibadem --crawl --max-pages 60 --depth 2 --delay 1.5

# 2) Mutlaka: indeksi güncelle
python manage.py build_page_embeddings --batch-size 16
```

**Tek bir sayfa** güncellendiyse:

```bash
python manage.py build_page_embeddings --page-id 182
```

(`182` yerine Django admin veya DB’den ilgili `Page.id` kullanın.)

### Ne zaman `build_page_embeddings` şart?

- `scrape_acibadem` çalıştıktan sonra (yeni veya değişmiş içerik).
- Seed listesine yeni URL ekledikten ve crawl aldıktan sonra.
- Elle `Page` düzenlediyseniz.

`Page`’de metin görünüp sohbetin “bilmiyorum” demesi çoğunlukla **son embedding adımının eksik veya eski** olmasından kaynaklanır.

## 5. Doğrulama (sıkıntı yaşamamak için)

```bash
cd backend

python manage.py rag_stats
python manage.py rag_index_audit
```

- **Pages total** ile **Pages embedded** farkı: embed edilmemiş sayfa sayısıdır. İdealde crawl sonrası tüm sayfalar için chunk vardır; fark varsa `build_page_embeddings` eksik kalmış olabilir.

İsteğe bağlı örnek kontrol (belirli URL `Page`’de var mı, metin var mı):

```bash
python manage.py shell -c "from core.models import Page; u='https://acibadem.edu.tr/en/...'; p=Page.objects.filter(url=u).first(); print(p.id if p else None, len(p.content) if p else 0)"
```

## 6. Sık hatalar

| Yapılan | Sonuç |
|---------|--------|
| Sadece `scrape_acibadem`, embedding yok | `Page` dolu; RAG zayıf veya yanlış — vektör indeksi güncel değil. |
| Seed’e URL eklemek ama crawl + embed yapmamak | URL hiç `Page`’e yazılmamış veya embed yoktur. |
| Farklı makinede crawl, chat başka DB’de | API’nin bağlandığı veritabanında `rag_stats` aynı görünmeli. |
| `refresh_rag` ** `--keep-existing` olmadan** yanlışlıkla çalıştırmak | Tüm `Page` / chunk’lar silinir; yalnızca crawl limiti kadar sayfa geri gelir (`--max-pages`). |
| Embedding modeli ilk seferde inmemiş / proxy | `embed_texts` hata verebilir; log ve ortam değişkenlerine (`EMBEDDING_MODEL`) bakın. |

## 7. Ortam hatırlatmaları

- Django genelde **`Project/.env`** dosyasını kullanır (`README.md` ile uyumlu).
- PostgreSQL’de **pgvector** extension gerekir.
- Sohbet API’si hangi veritabanına bağlıysa, bütün `manage.py` komutları **aynı** `DATABASES` ayarıyla çalışmalıdır.

## 8. İlgili dosyalar

- `core/management/commands/scrape_acibadem.py` — crawl, seed URL’ler.
- `core/management/commands/build_page_embeddings.py` — chunk + embedding.
- `core/management/commands/refresh_rag.py` — ikisini birlikte; varsayılan tam silme.
- `core/rag_retrieval.py` — `DocumentChunk` üzerinden arama.

---

**Kısa özet:** Crawl → `Page`. RAG → `DocumentChunk` + embedding. Güncel cevap için crawl sonrası **mutlaka** embedding adımını çalıştırın; mümkünse `refresh_rag` veya `build_page_embeddings` ile `rag_stats` ile doğrulayın.
