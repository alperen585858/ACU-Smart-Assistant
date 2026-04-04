# Sohbet neden “saçmalıyor” gibi görünür?

Bu belge, ACU chatbot’ta **anlamsız, alakasız, genel sohbet botu cevapları, uydurma bilgi veya sürekli red / İngilizce-Türkçe karışık** yanıtlar gibi sorunların **olası teknik ve ürün nedenlerini** özetler. Kod ve Docker ayarları `Project/backend/chat/views.py` ile `Project/docker-compose.yml` üzerinden bağlanır.

---

## 1. Bağlam penceresi çok küçük (en sık neden)

**Belirti:** Model “Microsoft / OpenAI / 2023 bilgim” der, sitedeki metni yok sayar, soruya yaklaşmaz veya önceki turdaki talimatları unutur.

**Neden:** Ollama’ya gönderilen toplam metin (sistem + sohbet geçmişi + son kullanıcı mesajındaki `===CONTEXT===` bloğu) **tek seferde işlenebilecek token sayısını** aşarsa, çoğu model **baştan veya ortadan kesme** yapar. Kod yorumlarında açıkça: uzun thread’lerde sistem + RAG’ın düşmesi ve “generic disclaimer”e kayma.

**Projede somut çatışma:**

- `views.py` içinde `OLLAMA_NUM_CTX` varsayılanı **4096**.
- `docker-compose.yml` backend servisinde varsayılan **`OLLAMA_NUM_CTX: 1024`** — bu, yerel dosyadaki 4096’yı pratikte **eziyor**.

1024 token; uzun sistem metni + birkaç mesaj geçmişi + 2000 karaktere kadar RAG özeti için **yetersiz** kalır. Model gerçek “grounding” metnini görmeden eğitimden gelen kalıplarla konuşur → **saçma veya halüsinasyon**.

**Ne yapmalı:** `.env` / Compose’ta `OLLAMA_NUM_CTX` değerini **en az 4096** (mümkünse daha yüksek) yapın; gerekirse `CHAT_HISTORY_MAX_MESSAGES` değerini düşürün.

---

## 2. Çıktı uzunluğu sınırı çok kısa (Docker)

**Belirti:** Cümle yarım kesilir, tek kelime / garip bitiş, “anlamsız” hissi.

**Neden:** Compose’ta `OLLAMA_NUM_PREDICT` varsayılanı **96** token. Kısa ve keskin cevap hedeflenmiş olsa da, model bazen düşünceyi tamamlayamadan kesilir.

**Ne yapmalı:** 192–256 gibi bir değer deneyin (`views.py` varsayılanı 256).

---

## 3. Küçük yerel model (`phi3:mini`)

**Belirti:** Talimatlara uymama, CONTEXT’i yanlış yorumlama, bazen alakasız dolgu.

**Neden:** Mini modeller **uzun kurallı sistem prompt** + yapılandırılmış `===CONTEXT===` / `===QUESTION===` formatını tutarlı işlemekte zorlanır; bağlam biraz kaybolunca hemen “genel asistan” moduna kayar.

**Ne yapmalı:** Mümkünse daha güçlü bir model; ayrıca bağlam penceresini büyütmek ve geçmişi kısaltmak aynı modelde bile belirgin iyileştirir.

---

## 4. RAG yanlış veya zayıf bağlam getiriyor

**Belirti:** Cevap “resmi sitesinden” gibi görünür ama **yanlış sayfa / yanlış konu**; veya soru Türkçe iken İngilizce snippet’lerle zor eşleşme.

**Nedenler (özet):**

- Gömme modeli **İngilizce ağırlıklı** (`BAAI/bge-small-en-v1.5`); Türkçe sorguda vektör araması şaşar.
- `RAG_RELAX_ON_EMPTY=true` iken eşik altında sonuç yoksa yine de **en yakın K chunk** gönderilir → soru ile **alakasız** metin modele “doğru kaynak” gibi gider; küçük model uydurmaya meyillidir.
- İndeks az sayfa / eski crawl → hiç bağlam yok veya dar.

**Ne yapmalı:** `refresh_rag` ile güncel ve yeterli sayfa; gerekirse `RAG_RELAX_ON_EMPTY=false` ile test; çok dilli gömme modeli düşünmek. Ayrıntı için `RAG_ANALIZ.md`.

---

## 5. RAG fiilen kapalı (embedding / DB)

**Belirti:** Sürekli aynı fallback cümlesi veya genel bilgi dolgusu; arayüzde `embedding_ok: false` veya chunk sayısı 0.

**Neden:** `embed_query` boş döner (model indirilemedi, bellek, ilk çalıştırma) veya `DocumentChunk` tablosu boş / Postgres yok.

**Ne yapmalı:** Backend logları, veritabanında chunk sayısı, `SentenceTransformer` ilk yükleme.

---

## 6. Dil talimatı ile kullanıcı beklentisi çakışıyor

**Belirti:** Kullanıcı Türkçe yazar, sistem **İngilizce cevap** ister; model karışık veya kullanıcıya “saçma” gelen ton üretir.

**Neden:** `SYSTEM_BASE` ve `SYSTEM_RAG_USER_WRAPPER` açıkça **English only** (RAG modunda istisna: kullanıcı Türkçe yapıştırdıysa). Bu bilinçli bir kısıt; kullanıcı Türkçe sohbet bekliyorsa **ürün olarak** “bot saçmalıyor” algısı oluşur.

**Ne yapmalı:** Beklenen dil politikasını prompt ile uyumlu hale getirmek (ayrı bir ürün kararı).

---

## 7. Uzun sohbet + geçmiş kırpma

**Belirti:** İlk mesajlar iyiyken **sonradan** bozulma.

**Neden:** `CHAT_HISTORY_MAX_MESSAGES` (varsayılan 12) ve mesaj başına `CHAT_MESSAGE_MAX_CHARS` (900) ile geçmiş kısaltılıyor; üstüne token limiti düşükse eski taraftaki sistem bağlamı tamamen gidebilir.

---

## 8. Hızlı kontrol listesi

| Kontrol | Olumsuzda ne olur? |
|--------|---------------------|
| `OLLAMA_NUM_CTX` ≥ 4096 mü? (Docker’da özellikle) | Bağlam kesilir, genel saçma yanıt |
| `OLLAMA_NUM_PREDICT` çok düşük mü? | Yarım cevap |
| Chunk sayısı > 0 ve `embedding_ok`? | RAG yok, boş veya genel cevap |
| `relaxed_retrieval` sık açık mı? | Yanlış sayfadan uydurma riski |
| Model boyutu / RAM yeterli mi? | Bozuk veya tekrarlayan çıktı |

---

## 9. Kısa özet

Çoğu “saçmalama” vakası **tek başına prompt hatası değil**; **bağlam penceresinin (özellikle Docker’daki 1024) RAG + geçmiş + uzun sistem mesajına yetmemesi** ve **küçük model + zayıf/yanlış retrieval** kombinasyonundan çıkar. Önce `OLLAMA_NUM_CTX` ve `OLLAMA_NUM_PREDICT` değerlerini makul seviyeye çekmek, sonra indeks ve retrieval kalitesini doğrulamak en hızlı iyileştirme yoludur.

---

*Belge tarihi: 5 Nisan 2026.*
