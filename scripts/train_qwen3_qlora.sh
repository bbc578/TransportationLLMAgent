#!/usr/bin/env bash
set -euo pipefail

: "${QWEN3_BASE_MODEL_PATH:?set QWEN3_BASE_MODEL_PATH}"
: "${QWEN3_LORA_OUTPUT_DIR:?set QWEN3_LORA_OUTPUT_DIR}"

python - <<'PY'
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("bitsandbytes") else "bitsandbytes is required for QLoRA")
PY

export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"
python scripts/prepare_sft_data.py
python -m qwen3_agent_sft.training.train_lora --config configs/train_lora.yaml --qlora
echo "QLoRA adapter saved to: ${QWEN3_LORA_OUTPUT_DIR}"
