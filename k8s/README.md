# ACU Smart Assistant — Kubernetes (kind) runbook

Bu klasördeki manifestler **yerel kind** kümesinde `acu-smart-assistant` namespace’inde uygulanmak üzere hazırlanmıştır. Servis sırası: **Postgres (pgvector) → Ollama → Django backend → Next.js frontend → Nginx**.

## Önkoşullar

- **Docker** (Desktop veya benzeri; kind node’ları container olarak çalışır)
- **kubectl**
- **kind** ([kurulum](https://kind.sigs.k8s.io/docs/user/quick-start/#installation)): `kind` PATH’te olmalı (ör. `./.k8s-kind/kind` proje içine indirilip kullanılabilir).

Küme adı bu dokümanda **`acu-kind`** kabul edilir; farklı isim kullanıyorsan `--name` ve bağlamı buna göre değiştir.

## 1. kind kümesini oluşturma

```bash
kind create cluster --name acu-kind
kubectl config use-context kind-acu-kind
kubectl cluster-info
```

İstersen tek düğümlü varsayılan profil yeterlidir; ek `kind` YAML’i zorunlu değildir.

## 2. Uygulama imajlarını derleme

Depo kökünden backend ve frontend imajlarını üretin (imaj adları `docker compose` proje adıyla eşleşir; tipik olarak `acu-smart-assistant-backend:latest` ve `acu-smart-assistant-frontend:latest`):

```bash
cd /path/to/ACU-Smart-Assistant
docker compose build backend frontend
docker images | grep acu-smart-assistant
```

## 3. İmajları kind düğümüne yükleme

Yerel Docker’daki imajları kind içindeki containerd’a aktarın:

```bash
kind load docker-image acu-smart-assistant-backend:latest --name acu-kind
kind load docker-image acu-smart-assistant-frontend:latest --name acu-kind
kind load docker-image pgvector/pgvector:pg15 --name acu-kind
kind load docker-image nginx:1.27-alpine --name acu-kind
```

### Ollama imajı (`ollama/ollama:latest`)

Bazı ortamlarda `kind load docker-image ollama/ollama:latest` **containerd içe aktarma hatası** verebilir (özellikle çok katmanlı imajlarda). Bu durumda host’ta imajın çekilmiş olduğundan emin olun, sonra node’a doğrudan içe aktarın:

```bash
docker pull ollama/ollama:latest   # host’ta
docker save ollama/ollama:latest | docker exec -i acu-kind-control-plane ctr -n k8s.io images import -
```

Manifestte `imagePullPolicy: IfNotPresent` tanımlıdır; böylece düğümde imaj varken registry’ye gitmez (kind içi DNS sorunlarında kritik).

## 4. Gizli bilgiler (teslim öncesi)

`02-secrets.yaml` içinde **örnek** `POSTGRES_PASSWORD` ve `SECRET_KEY` vardır. Gerçek bir teslimatta bunları **değiştirip** `kubectl apply` ile güncelleyin veya `kubectl create secret` / sealed secrets kullanın; düz metin sırları repoya commit etmeyin.

## 5. Manifestleri uygulama

Sıra genelde otomatik çözülür; tek seferde:

```bash
kubectl apply -f k8s/
```

Namespace ve tüm kayıtlar `k8s/` altındaki dosyalarla oluşturulur.

## 6. Doğrulama

```bash
kubectl get pods -n acu-smart-assistant
kubectl get svc -n acu-smart-assistant
```

Tüm workload pod’larının **Running** ve mümkünse **READY 1/1** olması beklenir.

Sağlık kontrolleri:

```bash
kubectl exec -n acu-smart-assistant deploy/acu-backend -- \
  python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health/')"
```

### Ollama ve LLM modeli

Varsayılan manifestte `OLLAMA_PULL_MODEL` boştur (uzun süren model indirmesini başlangıçta engellemek için). Sohbet/LLM için küme içinde modeli bir kez çekin:

```bash
kubectl exec -n acu-smart-assistant deploy/acu-ollama -- ollama pull llama3.2:3b
```

`acu-config` içindeki `OLLAMA_MODEL` ile uyumlu olmalıdır.

## 7. Erişim (port-forward)

### Tek giriş: Nginx (önerilen)

Tarayıcı ve API aynı origin üzerinden:

```bash
kubectl port-forward svc/acu-nginx 8080:80 -n acu-smart-assistant
```

- Arayüz: `http://localhost:8080/`
- API: `http://localhost:8080/api/...`

**Not:** Frontend deployment’ı `NEXT_PUBLIC_API_URL=http://localhost:8000` ile gelir; bu, **tarayıcının doğrudan backend’e gitmesi** anlamına gelir ve Nginx üzerinden 8080 kullanırken CORS/origin uyumsuzluğu verebilir. Tamamen Nginx üzerinden tek port kullanacaksanız, imajı `NEXT_PUBLIC_API_URL=http://localhost:8080` (veya kullandığınız host/port) ile yeniden build edip `kind load` etmeyi düşünün.

### Ayrı portlar (mevcut imajla uyumlu)

Backend ve frontend’e ayrı port-forward:

```bash
kubectl port-forward svc/acu-backend 8000:8000 -n acu-smart-assistant &
kubectl port-forward svc/acu-frontend 3000:3000 -n acu-smart-assistant &
```

Bu durumda `NEXT_PUBLIC_API_URL=http://localhost:8000` tarayıcıdan backend’e gider; UI `http://localhost:3000` üzerinden açılır.

## 8. Sık sorunlar

| Belirti | Olası neden | Ne yapmalı |
|--------|-------------|------------|
| `ImagePullBackOff` / registry DNS | kind düğümü dış registry’ye erişemiyor | İmajı host’ta çekip `kind load` veya `docker save \| ctr import`; `imagePullPolicy: IfNotPresent` |
| İki Ollama/DB pod’u, PVC takılı | RWO volume + RollingUpdate | Ollama deployment’ta `strategy: Recreate` kullanın (manifestte tanımlı) |
| `kubectl` pod içinde yok | Normal | `kubectl` **her zaman host’ta** çalışır; pod içinde değil |
| `kubectl` localhost:80 reddediyor | kind kapalı / yanlış context | `kind get clusters`; `kubectl config get-contexts` |
| Uzun süren `docker compose build` | Torch vb. bağımlılıklar | İlk build sürebilir; beklenen davranış |

---

## Teslim için ekran görüntüsü yönergeleri

Aşağıdakiler ödev/rapor tesliminde tutarlı bir kanıt seti için yeterlidir. Her ekran görüntüsünde mümkünse **tam pencere** veya **terminal + tarih/saat** görünsün; gerekiyorsa aynı oturumda ardışık alın.

### Zorunlu (minimum)

1. **Küme bağlamı**  
   `kubectl config current-context` çıktısı — `kind-acu-kind` (veya kullandığınız kind context adı) görünmeli.

2. **Namespace ve pod durumu**  
   ```bash
   kubectl get pods -n acu-smart-assistant -o wide
   ```  
   Tüm ilgili pod’ların `Running` ve `Ready` olduğu satırlar görünmeli (en azından db, backend, frontend; ollama ve nginx dağıtımınıza göre).

3. **Servisler**  
   ```bash
   kubectl get svc -n acu-smart-assistant
   ```  
   `ClusterIP` ve portların listelendiği görüntü.

4. **Uygulama çalışıyor**  
   - Port-forward ile (Nginx: `8080→80` veya yukarıdaki ayrı senaryo) tarayıcıda açılmış **ana sayfa** veya uygulama ekranı.  
   - Adres çubuğunda `localhost` ve kullandığınız port net olsun.

### İsteğe bağlı (kalite / ek puan)

5. **Sağlık endpoint’i**  
   Tarayıcı veya `curl` ile `http://localhost:<port>/api/health/` (port-forward’u backend veya nginx’e göre seçin) — JSON veya 200 yanıtı.

6. **Veritabanı pod içi** (sadece göstermek için)  
   ```bash
   kubectl exec -it deploy/acu-db -n acu-smart-assistant -- psql -U acu -d acu_chatbot -c "\conninfo"
   ```  
   Bağlantı bilgisi çıktısı (şifre görünmeyecek şekilde kırpılabilir).

7. **Ollama model listesi** (LLM kanıtı)  
   ```bash
   kubectl exec -n acu-smart-assistant deploy/acu-ollama -- ollama list
   ```  
   İndirilen model satırı görünsün.

### Güvenlik notu (teslim dosyalarında)

- `02-secrets.yaml` veya ekran görüntülerinde **gerçek üretim şifreleri** paylaşmayın; örnek değerler veya maskeleme kullanın.

---

## Dosya özeti (`k8s/`)

| Dosya | İçerik |
|-------|--------|
| `00-namespace.yaml` | `acu-smart-assistant` namespace |
| `01-configmap.yaml` | Uygulama ortam değişkenleri (DB host, Ollama URL, CSRF, vb.) |
| `02-secrets.yaml` | Örnek Secret (üretimde değiştirin) |
| `10–12-*` | Postgres PVC, Deployment, Service |
| `20–22-*` | Ollama PVC, Deployment, Service |
| `30–31-*` | Backend Deployment, Service |
| `40–41-*` | Frontend Deployment, Service |
| `50–52-*` | Nginx ConfigMap, Deployment, Service |

Sorular veya kurumsal bir pipeline (`kustomize` / Helm) ihtiyacı için depo sahipleriyle iletişime geçin.
