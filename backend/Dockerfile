FROM python:3.12-slim

WORKDIR /app

# Selenium (scrape_acibadem_js, scrape_obs_bologna): Chromium + uyumlu chromedriver (--no-sandbox ile headless).
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        chromium \
        chromium-driver \
        fonts-liberation \
    && rm -rf /var/lib/apt/lists/*

ENV CHROMIUM_BIN=/usr/bin/chromium

COPY requirements.txt .
RUN pip install --no-cache-dir --timeout=300 torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir --timeout=300 -r requirements.txt

COPY . .

EXPOSE 8000
