#!/usr/bin/env bash
set -euo pipefail

: "${QWEN3_BASE_MODEL_PATH:?set QWEN3_BASE_MODEL_PATH}"
: "${QWEN3_LORA_OUTPUT_DIR:?set QWEN3_LORA_OUTPUT_DIR}"
: "${QWEN3_MERGED_MODEL_DIR:?set QWEN3_MERGED_MODEL_DIR}"

export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"
python -m qwen3_agent_sft.training.merge_lora \
  --base-model-path "$QWEN3_BASE_MODEL_PATH" \
  --adapter-dir "$QWEN3_LORA_OUTPUT_DIR" \
  --output-dir "$QWEN3_MERGED_MODEL_DIR"
