#!/usr/bin/env bash
# Download BAAI/Emu3.5-VisionTokenizer checkpoint from Hugging Face into pretrained/

set -e

REPO_ID="BAAI/Emu3.5-VisionTokenizer"
OUTPUT_DIR="pretrained"

mkdir -p "$OUTPUT_DIR"
echo "Downloading ${REPO_ID} to ${OUTPUT_DIR}/ ..."

if command -v huggingface-cli &> /dev/null; then
  huggingface-cli download "$REPO_ID" --local-dir "$OUTPUT_DIR"
elif command -v hf &> /dev/null; then
  hf download "$REPO_ID" --local-dir "$OUTPUT_DIR"
else
  echo "Error: Neither 'huggingface-cli' nor 'hf' found. Install with: pip install -U huggingface_hub"
  exit 1
fi

echo "Done. Checkpoint files are in ${OUTPUT_DIR}/"
