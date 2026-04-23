# Ubuntu sunucusunda kurulum ve RAG (scraping + embedding) komutları

Bu dosya, projeyi **sıfır bir Ubuntu** makinesine alıp çalıştırırken ve **tüm veri indeksini** (crawl, embedding) yenilerken kullanacağın komutların özeti.

## Nereye kuruyorsun?

| Ortam | Rol |
|--------|-----|
| **VMware içindeki Ubuntu** | Asıl hedef: burada kurulumu bu rehberdeki komutlarla yap. |
| **AWS’deki production** | Ayrı ortam: **silinme zorunluluğu yok**; yedek / ikinci canlı gibi bırakabilirsin. |

İkisi **birbirini otomatik silmez**. Aynı repoyu her iki tarafta da kullanabilirsin; sadece her makinede **kendi `.env` / `.env.production`** değerleri (farklı `ALLOWED_HOSTS`, `NEXT_PUBLIC_API_URL`, veritabanı) olmalı. AWS’e yanlışlıkla `terraform destroy` / EBS silme / konteyner volume silme gibi işlemler yapılmadıkça yedek production ayakta kalır.

Detaylar için: [README.md](README.md), [DEPLOYMENT.md](DEPLOYMENT.md), [RAG_VERI_PIPELINE_REHBERI.md](RAG_VERI_PIPELINE_REHBERI.md).

---

## 0. GitHub `main` → VMware Ubuntu otomatik deploy

Repository’de workflow: [`.github/workflows/deploy-vmware-ssh.yml`](.github/workflows/deploy-vmware-ssh.yml).

- **Tetikleyici:** `main` branch’e **push** sonrası [CI](.github/workflows/ci.yml) **başarılı** olunca çalışır. İstersen manuel: GitHub **Actions** → **Deploy Ubuntu (VMware / SSH)** → **Run workflow**.

**Repository ayarları (Settings → Secrets and variables → Actions):**

| Tip | Ad | Örnek |
|-----|-----|--------|
| Secret | `DEPLOY_SSH_PRIVATE_KEY` | SSH private key (tam metin) |
| Secret | `DEPLOY_HOST` | VM’nin dışarıdan erişilen IP veya DNS |
| Secret | `DEPLOY_USER` | `ubuntu` vb. |
| Variable | `DEPLOY_PATH` | `/home/ubuntu/Project` (git kökü) |

**`production` environment** kullanıyorsan (varsayılan), aynı isimli secret/variable’ları o environment için de tanımlaman gerekir.

**VM’de bir kez:** Aynı repoyu klonla; `~/.ssh/authorized_keys` içine bu private key’e ait **public key**; repoya `git pull` için **Deploy key** (veya erişim belirteci). `docker` ve aşağıdaki bölüm 4 ile servis ayağa kalkmış olsun.

**Ağ (önemli):** GitHub’ın sunucuları (internet) bu makineye **SSH açabilmeli**. Sadece **VMware NAT** ve 192.168.x iç IP varsa, dışarıdan erişim olmaz; bu durumda ağda **Bridget + dış port / statik yönlendirme**, public IP, VPN veya **self-hosted GitHub runner** (VM üzerinde çalışır, inbound SSH gerekmez) gibi alternatiflere ihtiyaç vardır.

Eski [EC2 SSH deploy](.github/workflows/deploy-ec2-simple.yml) workflow’u kullanmıyorsan depoda etkisiz bırakabilir veya yorum satırıyla kapatabilirsin; VMware için yeni dosya yeterlidir.

---

## 1. Sistem ve Docker

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y git curl ca-certificates
```

Docker kurulumu:

```bash
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker "$USER"
```

Oturumu kapatıp aç veya `newgrp docker`, sonra:

```bash
docker compose version
```

---

## 2. Projeyi indir

```bash
cd ~
git clone <REPO_URL> Project
cd Project
```

`<REPO_URL>` yerine kendi repon (HTTPS veya SSH).

---

## 3. Ortam dosyaları

```bash
cd ~/Project
cp .env.example .env
```

`.env` içinde en az: `POSTGRES_*`, gerekirse `SECRET_KEY`, `ALLOWED_HOSTS`, `LLM` ayarları. **Üretim (Docker prod)** için:

```bash
# Örnek şablon varsa doldur:
# cp deploy/env.production.example .env.production
```

`docker-compose.prod.yml` **`.env.production`** ve kökteki **`.env`** (özellikle `NEXT_PUBLIC_API_URL`) okur. Tarayıcının API’ye gittiği tam adres `NEXT_PUBLIC_API_URL` ile aynı olmalı (ör. `https://alanadiniz.com`).

---

## 4. Servisleri ayağa kaldır (üretim compose)

Proje kökünden:

```bash
cd ~/Project
RUN_MIGRATIONS=1 docker compose -f docker-compose.prod.yml up -d --build
```

İlk migrasyon sonrası tekrar deploy ederken `RUN_MIGRATIONS` vermeyebilir veya `0` bırakabilirsin.

Konteynerlerin durumu:

```bash
docker compose -f docker-compose.prod.yml ps
```

---

## 5. Scraping + embedding (tek sefer, önerilen)

Tüm RAG hattı: mevcut indeksi temizle (varsayılan) → Acıbadem crawl → JS sayfalar (Chrome) → OBS Bologna (Chrome) → embedding.

### 5.1 Headless Chrome gerektirmeyen hafif senaryo (Docker `backend` içinde)

`Dockerfile.prod` imajında Chrome yok; sadece HTTP crawl + embedding için:

```bash
cd ~/Project
docker compose -f docker-compose.prod.yml exec backend \
  python manage.py refresh_rag --without-obs --without-acibadem-js --max-pages 60 --depth 2
```

### 5.2 Tam kapsam (Selenium: OBS + academic staff JS)

Bunun için **Ubuntu üzerinde** Chrome/Chromium + sanal ortam ile `manage.py` çalıştırman gerekir (bölüm 6). Tam crawl (sayfa/derinlik sınırı yok, sadece `/en/...` ve robots kuralları):

```bash
cd ~/Project/backend
source .venv/bin/activate
python manage.py refresh_rag --max-pages 0 --depth -1
```

- `--max-pages 0` → sayfa üst sınırı yok (kuyruk bitene kadar).
- `--depth -1` → link derinliği sınırı yok.
- Mevcut veriyi silmeden: `--keep-existing` ekle.

---

## 6. Ubuntu’da venv + Chromium (tam pipeline için)

```bash
sudo apt install -y python3.12-venv python3-pip
sudo apt install -y chromium-browser chromium-chromedriver
# veya: Google Chrome resmi .deb
```

```bash
cd ~/Project
cp .env.example .env
# Veritabanı: Docker db kullanıyorsan, hosttan bağlanırken port haritalamasına göre POSTGRES_PORT ayarla
# (ör. compose 5433 açıyorsa POSTGRES_HOST=127.0.0.1, POSTGRES_PORT=5433)

cd backend
python3.12 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
python manage.py migrate
```

İlk embedding model indirmesi internet ve disk kullanır (`sentence-transformers`).

Sonra:

```bash
python manage.py refresh_rag --max-pages 0 --depth -1
```

---

## 7. Komutlar (ayrı ayrı)

Sadece Acıbadem (requests) crawl:

```bash
cd ~/Project/backend && source .venv/bin/activate
python manage.py scrape_acibadem --crawl --max-pages 0 --depth -1 --delay 1.5
```

İngilizce olmayan yolları da taramak (dikkat: çok büyür):

```bash
python manage.py scrape_acibadem --crawl --max-pages 0 --depth -1 --allow-non-english
```

Sadece embedding (DB’deki `Page` satırlarından):

```bash
python manage.py build_page_embeddings --batch-size 16 --chunk-size 700 --chunk-overlap 120
```

Doğrulama:

```bash
python manage.py rag_stats
python manage.py rag_index_audit
python manage.py rag_verify_refresh --top-n 120
```

---

## 8. Docker içinden kısa yol (migrate / tek komut)

```bash
cd ~/Project
docker compose -f docker-compose.prod.yml exec backend python manage.py migrate
docker compose -f docker-compose.prod.yml exec backend \
  python manage.py refresh_rag --without-obs --without-acibadem-js --max-pages 0 --depth -1
```

(Chrome adımları atlanır; indeks sınırsız HTTP crawl + embedding olur.)

---

## 9. VMware (Ubuntu sanal makine) notları

- Ağ: **Bridged** kullanıyorsan VM, LAN’dan bir IP alır; **NAT** ise VMware’in port yönlendirmesi (ör. 8080 → VM:80) gerekebilir.
- UFW veya `iptables` açıksa: Nginx/Docker’ın dinlediği portlara (80, 443, 8080 vb.) izin ver.
- SSH: genelde 22; host üzerinden `ssh user@<VM-IP>`.
- Kalıcı veri: Docker volume’ler (`pgdata_prod` vb.) silinmedikçe DB uçmaz; `docker compose down -v` **volume’ü de siler** — dikkat.

## 10. AWS production (yedeğe bırakılan ortam)

AWS tarafı **ayrı bir sunucu / ayrı stack**; bu Markdown’taki VMware adımlarını uygulamak onu silmez. Canlı bırakmak için:

- Aynı GitHub’dan **ayrı deploy path** veya ayrı sunucu kullanımı normal; **AWS’te sadece güncellemek** isteğe bağlı.
- Eski production’a dokunmamak için: yanlışlıkla aynı DNS’i iki yere aynı anda verme (kafa karışıklığı) veya hangi sunucunun “asıl” olduğunu not et.
- Ayrıntı: [DEPLOYMENT.md](DEPLOYMENT.md) (EC2, güvenlik grubu, Elastic IP, GitHub `EC2_DEPLOY_PATH` / `EC2_HOST` vb.).

---

## 11. Hızlı referans tablosu

| Amaç | Komut (backend dizininde, venv açık) |
|--------|----------------------------------------|
| Tam pipeline (sınırsız en crawl) | `python manage.py refresh_rag --max-pages 0 --depth -1` |
| Veriyi silmeden yenile | Aynı komuta `--keep-existing` ekle |
| OBS + JS yok (Docker) | `refresh_rag --without-obs --without-acibadem-js ...` |
| Sadece istatistik | `python manage.py rag_stats` |

---
