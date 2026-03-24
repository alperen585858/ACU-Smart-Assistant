FROM node:22-alpine

WORKDIR /app

ENV NEXT_TELEMETRY_DISABLED=1

COPY package.json package-lock.json ./
RUN npm ci

COPY . .

EXPOSE 3000

# Listen on all interfaces so the host can reach the dev server
CMD ["npm", "run", "dev", "--", "--hostname", "0.0.0.0", "--port", "3000"]
