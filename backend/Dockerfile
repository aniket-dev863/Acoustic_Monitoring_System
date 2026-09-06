FROM node:20-bookworm-slim

WORKDIR /app

# Install build dependencies for better-sqlite3 native compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    make \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY backend/package*.json ./
RUN npm install --omit=dev

COPY backend/ ./backend/
COPY data/sample/ ./data/sample/
COPY experiments/results/ ./experiments/results/
COPY experiments/plots/ ./experiments/plots/

ENV PORT=5001
ENV NODE_ENV=production
ENV ML_SERVICE_URL=http://ml-service:8000

EXPOSE 5001

CMD ["node", "backend/src/server.js"]
