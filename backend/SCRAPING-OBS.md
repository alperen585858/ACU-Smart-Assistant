# OBS (Bologna) scraping komutları

Bu dosya `obs.acibadem.edu.tr` Bologna sayfalarını `core.Page` tablosuna almak için kullanılan **iki** Django management komutunu özetler.

Çalışma dizini: `Project/backend` (burada `manage.py` vardır).

```bash
cd Project/backend
```

---

## Gerçek kullanım sırası (bizim senaryo)

İlk denemede yaklaşık **140 sayfa** ile sınırlı çekim yaptık (`--max-programs 140` gibi); keşif listesinin tamamı gelmedi, **`progCourses`** ve diğer `prog*.aspx` sekmelerinde eksikler kaldı. Sonrasında eksikleri kapatmak için:

1. **Tekrar scrape:** İlk turdaki **`--max-programs 140`** sınırı olmadan (veya çok daha yüksek bir tavanla) **`scrape_obs_bologna`**’yı yeniden koştuk; böylece kalan Bologna URL’leri ve 2. geçiş **`prog*.aspx`** HTTP indirmeleri de işlendi.  
2. **Backfill:** Hâlâ boş veya eksik **`progCourses` / prog\*.aspx** satırları için DB’deki showPac kabuklarından **`backfill_obs_prog_tabs`** çalıştırdık.

**Güncelleme:** `scrape_obs_bologna` bir **dry-run değilse** ve `--skip-prog-backfill` verilmediyse, koşunun sonunda **aynı `backfill_obs_prog_tabs` mantığı** otomatik çalışır (showPac satırlarından sentetik `prog*.aspx` HTTP indirme). Böylece kritik sekmeler için ayrı ikinci komutu unutma ihtiyacı kalmaz.

Hâlâ elle backfill (sadece HTTP, Selenium yok) veya scrape’e ek güvence için iki komut alt alta:

```bash
python manage.py scrape_obs_bologna --fetch-workers 6 --progress
python manage.py backfill_obs_prog_tabs
```

İlk prototip için sadece 140 çekilmiş örnek (eksik beklenir; sonra yukarıdaki ikili veya ikinci scrape satırında `--max-programs` **kaldırılır**):

```bash
python manage.py scrape_obs_bologna --max-programs 140 --progress
```

### Tek komut yeter mi, ayrıca `backfill` şart mı?

**Genelde hayır.** `scrape_obs_bologna`:

1. Ana tarama + 2. geçiş ile `prog*.aspx` indirir,  
2. Ardından (**varsayılan**) veritabanındaki tüm showPac kabukları için **HTTP backfill** yapar (`backfill_obs_prog_tabs` ile aynı kod).

`--skip-prog-backfill` verirsen bu son adım çalışmaz.

**Sadece `backfill_obs_prog_tabs` ne zaman?**

- OBS’i **hiç Selenium çalıştırmadan**, sadece mevcut showPac satırlarından sekmeleri tazelemek / doldurmak.  
- Çok nadir kalan gedikler için **bir kez daha** aynı HTTP geçişini koşturmak (`--force` ile).

Özet: günlük tam koşu → `python manage.py scrape_obs_bologna` ( **`--max-programs`** koymadan; gerekirse `OBS_SECOND_PASS_MAX` yüksek). Backfill komutu artık çoğu ekip için **yedek**.

---

## 1. `scrape_obs_bologna` — Tam tarama (Selenium + HTTP)

**Ne yapar:** Bologna URL’lerini keşfeder, **headless Chrome** ile ağır sayfaları (özellikle `showPac`) çeker, `prog*.aspx` gibi sekmeler için **ikinci geçişte HTTP** kullanır; **dry-run değilse** ve `--skip-prog-backfill` yoksa en sonda **`backfill_obs_prog_tabs` ile aynı HTTP backfill’i** bir kez daha çalıştırır (ortak eksikleri kapatır). Sonuçlar `core.Page` içinde `source=obs.acibadem.edu.tr` ile saklanır.

```bash
python manage.py scrape_obs_bologna
```

### Sık kullanılan bayraklar

| Bayrak | Açıklama |
|--------|----------|
| `--dry-run` | Veritabanına yazmaz; URL, başlık özeti, içerik uzunluğu basar. |
| `--lang en` | Dil tercihi (varsayılan `en`). |
| `--delay 0.5` | Navigasyon / tıklama sonrası bekleme (saniye). |
| `--fetch-workers 4` | Paralel Chrome sayısı (1–12; env: `OBS_FETCH_WORKERS`). |
| `--max-programs N` | En fazla N sayfa (keşif listesinin başından). |
| `--verbose` | Keşif, özet, hata ve scraper WARNING logları. |
| `--progress` | Her URL öncesi/sonrası ilerleme satırları (tam `--verbose` değil). |
| `--stall-warn-interval 60` | `--fetch-workers 1` iken tek sayfa uzun süre beklerken periyodik uyarı (`0` kapalı). |
| `--clear-existing` | **Sadece** OBS kaynaklı Page satırlarını siler; ilişkili `DocumentChunk` da silinir. |
| `--fast` | Bazı `OBS_*` env varsayılanlarını “hızlı” preset ile doldurur (kalın içerik pahasına daha kısa süre). |
| `--skip-section alt1,alt2` | Virgülle ayrılmış alt dize; ilgili bölüm genişletme / linklerini atlar. |
| `--http-pass2-timeout 180` | 2. geçiş HTTP istekleri için URL başına süre sınırı (saniye); env: `OBS_HTTP_PASS2_PER_URL_TIMEOUT_S`. |
| `--skip-prog-backfill` | Scrape bittikten sonra otomatik showPac-tabanlı `prog*.aspx` HTTP backfill çalıştırma (**varsayılan**: çalışır). |

### Örnekler

Tam tarama (üretim benzeri, biraz daha hızlı worker):

```bash
python manage.py scrape_obs_bologna --fetch-workers 6 --progress
```

Önce DB’yi temizleyip yeniden çekmek (dikkat: sadece OBS sayfaları silinir):

```bash
python manage.py scrape_obs_bologna --clear-existing --verbose
```

Hızlı deneme preset’i (~10 dk hedef değildir; yük/sunucuya bağlıdır):

```bash
python manage.py scrape_obs_bologna --fast --delay 0.15 --fetch-workers 8 --progress
```

### İlgili ortam değişkenleri (özet)

- `OBS_SECOND_PASS_MAX` — 2. geçişte işlenecek ek URL üst sınırı (varsayılan mantık komutta; üst banda kadar çıkabilir).
- `OBS_FETCH_WORKERS` — `--fetch-workers` için CLI varsayılanı.
- `OBS_HTTP_PASS2_PER_URL_TIMEOUT_S`, `OBS_HTTP_PASS2_WORKERS` — 2. geçiş HTTP zaman aşımı ve worker sayısı.
- `OBS_SHOWPAC_MAX_DYNCON`, `OBS_SHOWPAC_SIDEBAR_CLICKS`, `OBS_DYNCON_HTTP_TIMEOUT` — showPac / dynCon davranışı (`--fast` bazılarını `setdefault` ile doldurur; siz export ettiyseniz sizinki geçerli).

---

## 2. `backfill_obs_prog_tabs` — Eksik `prog*.aspx` (yalnız HTTP)

**Ne yapar:** Selenium ile yeniden tarama **çalıştırmaz**. Veritabanındaki mevcut **showPac** `Page` satırlarını tarar; her biri için `synthetic_prog_followups_from_showpac_url` ile üretilen standart **`prog*.aspx`** URL’lerini **HTTP ile** çeker ve `Page` olarak `update_or_create` eder.

Kullanım: İlk scrape’te kaçan veya sonra eklenen programme sekmesi URL’leri için **hızlı tamamlama**.

```bash
python manage.py backfill_obs_prog_tabs
```

### Bayraklar

| Bayrak | Açıklama |
|--------|----------|
| `--dry-run` | HTTP yapmaz; kaç showPac tarandığı ve örnek sentetik URL sayısı/örnek liste basar. |
| `--limit N` | En fazla N showPac kaydı işle (`0` = hepsi). |
| `--force` | İçeriği dolu olan `prog*` sayfasını bile yeniden indirip günceller. |

### Örnekler

Dry-run ile kapsam kontrolü:

```bash
python manage.py backfill_obs_prog_tabs --dry-run
```

Tüm kabukları doldurma:

```bash
python manage.py backfill_obs_prog_tabs
```

---

## RAG / embedding notu

Sadece `Page` güncellenir; vektör araması **`DocumentChunk`** kullanıyorsa scrape veya backfill sonrası ilgili sayfalar için embedding üretimi gerekir (ör. projedeki `build_page_embeddings`, `refresh_rag` vb. akışına göre).

---

## Özet akış

Yukarıdaki **“Gerçek kullanım sırası”** bölümüne bakın: **`scrape_obs_bologna`** (gerekirse 140’dan sonra limitsiz tekrar) + **`backfill_obs_prog_tabs`**.
