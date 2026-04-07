# Dosya Yapisi Analizi

Bu dokuman, projenin mevcut klasor yapisini hizlica anlamak ve gelistirme surecinde yol gosterici bir referans sunmak icin hazirlanmistir.

## Genel Mimari

Proje temel olarak iki ana katmana ayriliyor:

- `backend/`: Django tabanli API ve is kurallari
- `frontend/`: Next.js tabanli kullanici arayuzu

Ek olarak kok seviyede ortak yapilandirma ve calistirma dosyalari bulunuyor:

- `.env` ve `.env.example`: Ortam degiskenleri
- `docker-compose.yml`: Servis orkestrasyonu
- `README.md`: Kurulum ve kullanim dokumani
- `scripts/`: Yardimci scriptler

## Klasor Bazli Inceleme

### `backend/`

Django projesi ve uygulama modulleri burada toplaniyor.

- `manage.py`: Django yonetim giris noktasi
- `config/`: Proje ayarlari (`settings.py`, `urls.py`, `asgi.py`, `wsgi.py`)
- `chat/`: Sohbet akislarina odakli uygulama modulu
  - `views.py`: HTTP endpoint mantigi (buyuk dosya, ana is akisi burada)
  - `models.py`: Veri modelleri
  - `urls.py`: Uygulama route tanimlari
- `core/`: RAG ve ortak domain mantigi
  - `rag_retrieval.py`, `rag_keywords.py`, `rag_config.py`: RAG altyapisi
  - `embeddings.py`: Embedding islemleri
  - `management/`: Ozel Django management komutlari
- `requirements.txt`: Python bagimliliklari

Not:
- `backend/` icinde hem `.venv/` hem `venv/` bulunuyor. Tek bir sanal ortam standardina gecmek bakim maliyetini azaltir.

### `frontend/`

Next.js (App Router) tabanli istemci tarafi kodlari burada.

- `app/`: Uygulama route ve sayfa yapisi
  - `layout.tsx`: Ana yerlesim
  - `page.tsx`: Giris sayfasi
  - `chat/`: Sohbet ekrani
- `public/`: Statik varliklar
- `package.json`: NPM script ve bagimliliklari
- `next.config.ts`, `tsconfig.json`, `eslint.config.mjs`: Framework ve kalite ayarlari
- `.next/` ve `node_modules/`: Derleme/calisma artefaktlari

## Guclu Yonler

- Backend/Frontend ayrimi net ve anlasilir.
- `core/` altinda RAG odakli ayri bir katman olusturulmus.
- Docker ve `.env.example` kullanimi dagitim ve onboarding acisindan olumlu.

## Eksiklikler

1. `chat/views.py` tek dosyada cok fazla sorumluluk tasiyor.
2. Python sanal ortam yapisi ikili (`.venv` ve `venv`) gorunuyor.
3. Build/gelistirme artefaktlarinin yonetimi icin net bir kontrol listesi dokumanda yer almiyor.
4. Kod tabani icin katmanlar arasi sorumluluk sinirlari (controller/service/repository) yazili degil.
5. Mimari kararlar ve teknik borc kalemleri tek bir merkezi dosyada toplanmamis.

## Iyilestirme Onerileri

1. `chat` modulu icinde endpoint mantigini daha kucuk dosyalara ayirin (`views/`, `services/`, `serializers/` gibi).
2. Tek sanal ortam standardi belirleyin ve ekipte ayni yapiyi kullanin.
3. `.gitignore` ve calistirma dokumanlarini artefakt yonetimi acisindan netlestirin.
4. Teknik dokumantasyonu `ARCHITECTURE.md` ve gerekirse `DECISIONS.md` ile guclendirin.
5. Her ana modulu test kapsamiyla birlikte takip edin (unit + integration).

## Yapilacaklar (Aksiyon Plani)

### Kisa Vade (1-3 gun)

- [x] `chat/views.py` icindeki endpointleri konu bazli gruplandirip bolunme plani cikar.
  - Plan: `list_sessions` ve `session_detail` -> `views/sessions.py`, `chat_completion` ve `_chat_with_db` -> `views/chat.py`, LLM/RAG helperlari -> `services/llm_service.py` ve `services/rag_service.py`, ortak text/validation yardimcilari -> `utils/message_utils.py`.
- [x] `.venv` veya `venv` icinden birini standart sec ve README'de tek akisi yaz.
  - Secim: standart Python ortami olarak `backend/.venv` kullanilacak.
- [x] `.gitignore` dosyasini kontrol et; `.next/`, `node_modules/`, `__pycache__/` gibi artefaktlarin disarida kaldigini dogrula.
  - Durum: `__pycache__/` zaten disarida; `frontend/.next/` ve `frontend/node_modules/` acikca eklendi.
- [x] `README.md` icine "Gelistirme Ortami Kurulumu" bolumunu net adimlarla ekle/guncelle.
  - Durum: yeni bolumde `.venv` olusturma/aktif etme ve backend/frontend kurulum akisi netlestirildi.

### Orta Vade (1-2 hafta)

- [x] `backend/chat` icin modul yapisini uygula (`views`, `services`, `schemas/serializers` ayirimi).
  - Durum: endpointler `session_views.py` ve `completion_views.py` olarak ayrildi; is mantigi `chat_logic.py` altinda toplandi; ikinci adimda `rag_service.py`, `llm_service.py`, `message_utils.py` ile servis katmani ayrildi.
- [x] `core/` altindaki RAG akisina dair kisa bir teknik akis diyagrami veya aciklama ekle.
  - Durum: `ARCHITECTURE.md` icine `RAG Flow (Core)` bolumu eklendi (query variants -> embedding -> vector retrieval -> rerank -> threshold/fallback -> context assembly).
- [x] `ARCHITECTURE.md` olustur; backend/frontend sinirlari ve veri akisini yaz.
  - Durum: `ARCHITECTURE.md` olusturuldu; katmanlar, is sinirlari, request/data flow ve konfigurasyon basliklari eklendi.
- [x] Kritik endpointler icin temel test senaryolari ekle.
  - Durum: `backend/chat/tests.py` eklendi; `sessions` ve `chat completion` endpointleri icin temel basari/hata senaryolari yazildi.
  - Not: Test calistirma ortami `core` migration'larindaki vector extension/index gereksinimi nedeniyle yerel DB yetkisine bagli hata verdi.

### Uzun Vade (2-4 hafta)

- [ ] Kod sahipligi ve klasor bazli sorumluluklari dokumante et.
- [ ] Teknik borc listesi olustur ve onceliklendirme matrisi belirle.
- [ ] CI adimlarina (lint/test/build) kalite kapilari ekleyip zorunlu hale getir.

## Ozet

Mevcut yapi, tam yiginli (Django + Next.js) bir sohbet/RAG uygulamasi icin dogru bir temel sunuyor. En yuksek etkiyi saglayacak adimlar: `chat/views.py` modullestirme, ortam standardizasyonu ve mimari dokumantasyonun guclendirilmesi.
