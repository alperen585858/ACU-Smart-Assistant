# RAG Veri Pipeline Rehberi

Bu rehber, `refresh_rag` sonrasi indeksin bos kalmamasini ve OBS-Bologna iceriginin korunmasini hedefleyen operasyon standardini tanimlar.

## Refresh Akisi (Varsayilan)

`python manage.py refresh_rag` komutu varsayilan olarak su sirayi calistirir:

1. `clear` (yalnizca `--keep-existing` verilmediginde)
2. `scrape_acibadem`
3. `scrape_obs_bologna` (varsayilan acik)
4. `build_page_embeddings`

Kritik notlar:

- `--keep-existing` verilmezse mevcut `Page` ve `DocumentChunk` satirlari temizlenir.
- OBS adimi varsayilan olarak aciktir; sadece bilincli durumlarda `--without-obs` kullanin.
- OBS scrape dili varsayilan `en` olarak korunur (`--obs-lang en`).

## Standart Dogrulama Akisi

Refresh sonrasi su komutlari standarttir:

```bash
python manage.py rag_stats
python manage.py rag_index_audit
python manage.py rag_diagnose_coverage --top-n 120
```

Tek komutla ayni akisi calistirmak icin:

```bash
python manage.py rag_verify_refresh --top-n 120
```

Opsiyonel olarak coverage raporu disariya alinabilir:

```bash
python manage.py rag_verify_refresh --top-n 120 --json-out reports/rag_coverage.json
```

## Basari Kriterleri

- `Page.source` dagiliminda `obs.acibadem.edu.tr` gorulmeli.
- `DocumentChunk` sayisi refresh sonrasi sifira dusup kalmamali.
- `rag_diagnose_coverage` raporunda OBS odakli sorularda top-N gorunurlugu iyilesmeli.

## Retrieval Tuning Notu

Varsayilan retrieval ayarlari, OBS kapsamini iyilestirmek icin su sekilde guncellendi:

- `RAG_VECTOR_CANDIDATE_POOL=80` (onceki 40)
- `RAG_MAX_DISTANCE=0.68` (onceki 0.62)
- `RAG_MAX_CHUNKS_PER_URL=3` (onceki 2)

Geri donus testleri icin gecici env override ile karsilastirma yapin:

```bash
RAG_VECTOR_CANDIDATE_POOL=40 RAG_MAX_DISTANCE=0.62 RAG_MAX_CHUNKS_PER_URL=2 python manage.py rag_diagnose_coverage --top-n 120 --json-out reports/coverage_baseline.json
python manage.py rag_diagnose_coverage --top-n 120 --json-out reports/coverage_tuned.json
```
