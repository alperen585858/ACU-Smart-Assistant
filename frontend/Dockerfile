# bookworm-slim: Next.js yerel (SWC) ikilileri Alpine/musl’da bazen SIGSEGV veriyor; glibc daha stabil.
FROM node:22-bookworm-slim

WORKDIR /app

ENV NEXT_TELEMETRY_DISABLED=1

COPY package.json package-lock.json ./
RUN npm ci

COPY . .

EXPOSE 3000

# Docker Desktop + bind mount için dosya izleme
ENV WATCHPACK_POLLING=true

CMD ["npm", "run", "dev", "--", "--hostname", "0.0.0.0", "--port", "3000"]
