#!/usr/bin/env bash
set -euo pipefail

: "${QWEN3_MERGED_MODEL_DIR:?set QWEN3_MERGED_MODEL_DIR}"

python -m sglang.launch_server \
  --model-path "$QWEN3_MERGED_MODEL_DIR" \
  --host 0.0.0.0 \
  --port "${LLM_PORT:-8001}" \
  --served-model-name qwen3-agent-lora
