#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"
uvicorn qwen3_agent_sft.api.main:app --host 0.0.0.0 --port "${API_PORT:-8080}"
