#!/usr/bin/env bash
set -euo pipefail

: "${QWEN3_MERGED_MODEL_DIR:?set QWEN3_MERGED_MODEL_DIR}"

vllm serve "$QWEN3_MERGED_MODEL_DIR" \
  --host 0.0.0.0 \
  --port "${LLM_PORT:-8001}" \
  --served-model-name qwen3-agent-lora \
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE:-1}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.85}" \
  --max-model-len "${MAX_MODEL_LEN:-32768}"
