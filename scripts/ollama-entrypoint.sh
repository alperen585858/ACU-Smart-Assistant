#!/bin/sh
# Ollama sunucusunu başlat, model yoksa otomatik indir.

ollama serve &
SERVER_PID=$!

echo "Waiting for Ollama server to start..."
sleep 5

if [ -n "$OLLAMA_PULL_MODEL" ]; then
    echo "Pulling model: $OLLAMA_PULL_MODEL"
    ollama pull "$OLLAMA_PULL_MODEL"
fi

wait $SERVER_PID
