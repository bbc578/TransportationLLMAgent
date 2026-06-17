#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"
streamlit run src/qwen3_agent_sft/app/streamlit_app.py --server.port "${STREAMLIT_PORT:-8501}"
