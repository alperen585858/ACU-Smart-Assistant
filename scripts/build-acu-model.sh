#!/bin/bash
# Build the ACU custom Ollama model from Modelfile
# Usage: ./scripts/build-acu-model.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
MODELFILE="$PROJECT_DIR/backend/ollama/Modelfile"

if [ ! -f "$MODELFILE" ]; then
    echo "ERROR: Modelfile not found at $MODELFILE"
    exit 1
fi

echo "Building acu-assistant model from $MODELFILE ..."
ollama create acu-assistant -f "$MODELFILE"
echo "Done. Run with: ollama run acu-assistant"
