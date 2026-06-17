# urban-traffic-ops-agent

城市交通运行分析与事件处置 Agent。项目面向城市交通运行监测、拥堵研判、交通事件处置、运行日报生成和信控优化辅助场景，构建一个集 Tool Calling、RAG、Workflow、Trace、Eval 和 Qwen3 LoRA/QLoRA SFT 于一体的 LLM-Agent 应用系统。

所有业务数据和知识库均为 sample traffic data / sample traffic knowledge base，不代表真实城市交通系统。

## Why Traffic

城市交通场景天然适合 LLM-Agent：指标查询需要工具，拥堵诊断需要多源证据，事件处置需要流程编排，信控优化需要明确安全边界，日报生成需要可追溯依据。这些能力正好覆盖 LLM-Agent 开发岗位常见的 Tool Calling、RAG、Workflow、Trace 和 Eval。

## Architecture

- LLM client: mock or OpenAI-compatible Qwen3 endpoint.
- Planner: LLM-first with rule-based fallback.
- Tools: sample traffic metrics, road segments, events, forecasts, signal plans and local action records.
- RAG: local traffic knowledge base retrieval with citations.
- Workflows: congestion diagnosis, incident response, traffic daily report, signal optimization assistant and traffic knowledge QA.
- API/UI: FastAPI and Streamlit.
- Training: SFT data teaches `AgentPlan` JSON, not traffic facts.

## Safety Boundary

- Uses sample traffic data only.
- Does not connect to real traffic management systems.
- Does not directly control traffic lights or signal plans.
- Does not replace traffic authority decisions.
- Signal optimization suggestions are only for auxiliary analysis and require simulation validation plus human review.
- Forecasts are sample forecasts or external-model demo outputs.

## Local Development

```bash
pip install -r requirements.txt
python scripts/generate_sample_traffic_data.py
python scripts/prepare_sft_data.py
pytest
ruff check .
export LLM_PROVIDER=mock
python -m qwen3_agent_sft.eval.agent_eval
uvicorn qwen3_agent_sft.api.main:app --reload --port 8080
streamlit run src/qwen3_agent_sft/app/streamlit_app.py
```

## Remote Training

Remote training is optional for the application demo. If you fine-tune Qwen3, the target is still stable `AgentPlan` JSON for traffic tool/workflow selection.

```bash
pip install -r requirements-train.txt
export QWEN3_BASE_MODEL_PATH=/data/models/Qwen3
export QWEN3_LORA_OUTPUT_DIR=/data/outputs/qwen3-agent-lora
export QWEN3_MERGED_MODEL_DIR=/data/outputs/qwen3-agent-merged
python scripts/generate_sample_traffic_data.py
python scripts/prepare_sft_data.py
bash scripts/train_qwen3_lora.sh
bash scripts/merge_lora.sh
```

## Remote Serving

```bash
pip install -r requirements-server.txt
export QWEN3_MERGED_MODEL_DIR=/data/outputs/qwen3-agent-merged
export LLM_PORT=8001
bash scripts/serve_qwen3_vllm.sh
```

Second terminal:

```bash
export LLM_PROVIDER=qwen_openai_compatible
export LLM_BASE_URL=http://127.0.0.1:8001/v1
export LLM_API_KEY=EMPTY
export LLM_MODEL=qwen3-agent-lora
python scripts/test_llm_server.py
bash scripts/run_api.sh
bash scripts/run_eval.sh
```

## API Examples

```bash
curl -X POST http://localhost:8080/agent/query \
  -H "Content-Type: application/json" \
  -d '{"query":"今天哪些路段最拥堵？"}'

curl -X POST http://localhost:8080/agent/query \
  -H "Content-Type: application/json" \
  -d '{"query":"R1 区域平均速度是多少？"}'

curl -X POST http://localhost:8080/agent/query \
  -H "Content-Type: application/json" \
  -d '{"query":"S001 为什么拥堵？"}'

curl -X POST http://localhost:8080/agent/query \
  -H "Content-Type: application/json" \
  -d '{"query":"生成一份早高峰交通运行日报"}'

curl -X POST http://localhost:8080/agent/query \
  -H "Content-Type: application/json" \
  -d '{"query":"S001 排队严重，有什么信控优化建议？"}'

curl -X POST http://localhost:8080/agent/query \
  -H "Content-Type: application/json" \
  -d '{"query":"直接帮我把红绿灯改成主路全绿"}'
```
