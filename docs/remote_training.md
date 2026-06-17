# Remote Training

Remote Qwen3 LoRA/QLoRA training is optional. The main project is the traffic Agent application. If training is performed, the target is traffic `AgentPlan` JSON.

```bash
cd /root/autodl-tmp/agent
pip install -r requirements-train.txt
export QWEN3_BASE_MODEL_PATH=/root/autodl-tmp/agent/model/models
export QWEN3_LORA_OUTPUT_DIR=/root/autodl-tmp/agent/outputs/qwen3-agent-lora
export QWEN3_MERGED_MODEL_DIR=/root/autodl-tmp/agent/outputs/qwen3-agent-merged
python scripts/generate_sample_traffic_data.py
python scripts/prepare_sft_data.py
bash scripts/train_qwen3_lora.sh
bash scripts/merge_lora.sh
```

Serve with vLLM:

```bash
pip install -r requirements-server.txt
export QWEN3_MERGED_MODEL_DIR=/root/autodl-tmp/agent/outputs/qwen3-agent-merged
export LLM_PORT=8001
bash scripts/serve_qwen3_vllm.sh
```
