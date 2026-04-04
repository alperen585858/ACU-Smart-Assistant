# RAG (Retrieval-Augmented Generation) analizi — ACU chatbot

Bu belge, projedeki RAG hattını uçtan uca inceleyerek **neden “düzgün çalışmıyor” gibi görünebileceğini** ve **somut riskleri** özetler. Kod referansları `Project/backend` ve `Project/frontend` altındaki mevcut yapıya göredir.

---

## 1. Mimari özeti

| Aşama | Bileşen | Açıklama |
|--------|---------|----------|
| Toplama | `scrape_acibadem` | `Page` tablosuna HTML’den metin yazar. |
| Parçalama + gömme | `build_page_embeddings` | Metni karakter pencereleriyle böler, `sentence-transformers` ile vektörler üretir, `DocumentChunk` kaydeder. |
| Sorgu | `chat/views.py` | `embed_query` ile sorgu vektörü; `pgvector` üzerinden kosinüs mesafesi; isteğe bağlı anahtar kelime takviyesi. |
| Üretim | Ollama / Claude | Sistem + (geçmiş) + son kullanıcı mesajında `===CONTEXT===` bloğu. |

**Kritik bağımlılık:** `DocumentChunk.embedding` alanı `pgvector` gerektirir; üretim yolu PostgreSQL + `vector` uzantısı olmadan migration veya sorgular sorun çıkarır (`settings.py`: `POSTGRES_USER` yoksa SQLite kullanılır — aşağıya bakın).

---

## 2. Yüksek öncelikli problemler

### 2.1 Docker’da çok düşük bağlam penceresi (`OLLAMA_NUM_CTX`)

`docker-compose.yml` içinde backend için varsayılan:

- `OLLAMA_NUM_CTX: ${OLLAMA_NUM_CTX:-1024}`

Oysa RAG tarafında:

- Sistem mesajı uzun (`SYSTEM_RAG_USER_WRAPPER`).
- Son kullanıcı balonu `RAG_USER_BUBBLE_MAX_CHARS` ile **binlerce karaktere** çıkabilir (varsayılan 4500+).
- Üstüne `CHAT_HISTORY_MAX_MESSAGES` (varsayılan 12) ile kısaltılmış geçmiş eklenir.

**Sonuç:** 1024 token bağlamda model, promptun başını (sistem + erken geçmiş) veya `===CONTEXT===` bloğunu **keserek** atar. Kod yorumlarında bu durumun “Microsoft / 2023” tarzı genel yanıtlara yol açtığı açıkça belirtilmiş; Docker varsayılanı bu riski **artırıyor**. `views.py` içindeki varsayılan `OLLAMA_NUM_CTX` (4096) Docker ile **override** ediliyor.

**Öneri:** Docker veya `.env` içinde en az **4096** (tercihen donanıma göre 8192) ve geçmiş mesaj sayısını buna göre sınırlamak.

### 2.2 SQLite ile yerel geliştirme: pgvector uyumsuzluğu

`config/settings.py`: `POSTGRES_USER` tanımlı değilse veritabanı **SQLite** olur. Migration’lar `CREATE EXTENSION vector` ve `VectorField` kullanır.

**Sonuç:** Yerel ortamda PostgreSQL kullanılmadan migration hatası veya vektör sorgularının çalışmaması; RAG fiilen devre dışı veya kurulum takılı kalır.

**Öneri:** RAG geliştirirken her zaman `pgvector/pgvector` imajlı Postgres’e bağlanın; `.env` ile `POSTGRES_*` doldurun.

### 2.3 İngilizce öncelikli gömme modeli + Türkçe sorular

`core/embeddings.py` varsayılanı: `BAAI/bge-small-en-v1.5` (İngilizce ağırlıklı). `views.py` içinde bunun için `_compose_rag_search_query` ile Türkçe sorulara İngilizce anahtar kelimeler eklenmesi denenmiş; bu **kısmi** bir iyileştirme.

**Sonuç:** Türkçe veya karışık sorularda semantik eşleşme zayıf kalır; `RAG_RELAX_ON_EMPTY` açıkken “en yakın K chunk” alınır ve **ilgisiz bağlam** modele gidebilir.

**Öneri:** Çok dilli veya Türkçe destekli bir gömme modeli (`EMBEDDING_MODEL`) ve aynı modelin hem indeks hem sorguda kullanıldığından emin olun; vektör boyutunun `VectorField(dimensions=384)` ile uyumlu olması gerekir (model değişiminde migration / yeniden indeks şart).

### 2.4 Gevşek geri çekme (`RAG_RELAX_ON_EMPTY`)

Eşik (`RAG_MAX_DISTANCE`, varsayılan 0.62) altında sonuç yokken, sistem yine de **mesafeye göre en yakın K** chunk’ı bağlama koyuyor.

**Sonuç:** Soru ile alakası düşük sayfalar modele “ground truth” gibi sunulur; küçük modeller kurallara uymayıp uydurma yapabilir veya yanlış sayfadan cevap üretebilir.

**Öneri:** Debug için `false` deneyin veya eşiği/ `RAG_TOP_K` değerini gözlemle ayarlayın.

---

## 3. Veri hattı (crawl → chunk → DB)

### 3.1 `scrape_acibadem` tek başına: çoğu zaman tek sayfa

`--crawl` verilmezse kuyruk yalnızca tohum URL’lerinden oluşur; varsayılan tohum `https://www.acibadem.edu.tr/en` — pratikte **tek sayfa** indekslenebilir.

**Sonuç:** Kullanıcı `refresh_rag` çalıştırmadan veya manuel scrape’te `--crawl` kullanmadan beklediği kapsam oluşmaz.

**Not:** `refresh_rag` yönetim komutu `crawl=True` ile çağırır; tam pipeline için bu komut doğru yol.

### 3.2 Sayfa başına üst sınır (`MAX_CONTENT_CHARS = 5000`)

Uzun sayfalarda metin kesilir; adres / iletişim alt kısımda kalıyorsa chunk’lara hiç girmeyebilir.

### 3.3 Yalnızca `/en/` yolları (varsayılan)

`normalize_url` ile İngilizce olmayan path’ler filtrelenir. Türkçe içerik isteyen sorular için corpus daralır (bilinçli tasarım; farkında olunmalı).

### 3.4 İlk gömme çalıştırması

Backend (özellikle Docker) ilk `SentenceTransformer` yüklemesinde model indirir; ağ yoksa, disk doluysa veya bellek yetersizse `embed_query` boş dönebilir → `embedding_ok: false`, RAG devre dışı.

---

## 4. Retrieval mantığı

### 4.1 Anahtar kelime takviyesi (`RAG_KEYWORD_BOOST`)

`DocumentChunk.objects.filter(content__icontains=term)[:3]` — sıralama vektör benzerliğine göre değil; sabit “mesafe” 0.75 ile listeye ekleniyor. Büyük tabloda **maliyet** ve **gürültülü** sonuç riski var.

### 4.2 Boş indeks

`DocumentChunk` yoksa bağlam metni boş kalır; `meta["reason"]` = `no_matching_chunks` benzeri durumlar ve kullanıcıya genel fallback cümlesi gider.

### 4.3 URL başına tekrar

Aynı URL’den çok chunk gelince, belirli bir karakter eşiğinden sonra aynı URL atlanıyor; bu iyi bir çeşitlilik önlemi, ancak doğru cevap tek URL’nin farklı chunk’larındaysa bazen bağlamı kısaltabilir.

---

## 5. LLM tarafı ve prompt

### 5.1 Sistem mesajı: “yalnızca İngilizce”

Kullanıcı Türkçe yazdığında bile talimatlar İngilizce cevap istiyor (Türkçe sadece kullanıcı yapıştırdıysa istisna). Ürün beklentisi Türkçe yanıtsa bu **ürün/UX uyumsuzluğu** olarak algılanır; RAG “çalışmıyor” sanılabilir.

### 5.2 `OLLAMA_NUM_PREDICT` (Docker’da 96)

Kısa üst sınır; detaylı, çok cümleli grounded cevaplar kesilir.

### 5.3 Claude yolu

`_call_claude` içinde `max_tokens: 256` sabit; uzun bağlam + kısa çıktı kombinasyonu yine kısıtlıdır.

---

## 6. Frontend ve API uyumu

Sohbet sayfası oturumlu modda yalnızca `client_id`, `message`, `session_id` gönderir; RAG sorgusu sunucuda önceki **kayıtlı kullanıcı mesajlarından** türetilir — bu tutarlı.

Stateless `messages[]` ile RAG birleşimi başka istemciler için `views.py` içinde ayrı işleniyor; tek tip test etmek için her iki yolu da göz önünde bulundurun.

---

## 7. Önerilen kontrol listesi (hızlı teşhis)

1. **Postgres + pgvector:** `DocumentChunk.objects.count()` > 0 mu? (`rag_stats` veya Django shell)
2. **Pipeline:** `python manage.py refresh_rag` (veya eşdeğer) en az bir kez başarıyla bitti mi?
3. **Ollama bağlamı:** `OLLAMA_NUM_CTX` Docker’da 1024 mü? RAG açıkken **artırın**.
4. **Embedding:** İlk istekte model indirme / RAM hatası loglarda var mı?
5. **Dil:** Soru dili ile gömme modeli ve crawl dili (`/en/`) uyumlu mu?

---

## 8. Sonuç

RAG zinciri **tasarım olarak bir bütün**; “çalışmıyor” hissi genelde şu üçlüden birinden gelir: **(A)** indeks boş veya çok dar crawl, **(B)** PostgreSQL/pgvector olmadan çalıştırma, **(C)** Özellikle Docker’daki **düşük `OLLAMA_NUM_CTX`** ile promptun bağlam penceresine sığmaması ve modelin bağlamı görmeden genel yanıt vermesi. Önce ortam ve bağlam boyutunu düzeltmek, sonra gömme modeli ve eşikleri veriyle ayarlamak en yüksek getirili adımlardır.

---

*Belge tarihi: 5 Nisan 2026 — kod tabanına göre statik analiz; üretim logları olmadan runtime kesinliği için log ve örnek sorgularla doğrulanmalıdır.*
