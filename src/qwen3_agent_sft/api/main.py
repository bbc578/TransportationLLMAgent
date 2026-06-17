from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from qwen3_agent_sft.agent.executor import AgentExecutor
from qwen3_agent_sft.agent.tool_registry import ToolRegistry
from qwen3_agent_sft.agent.trace import TraceStore
from qwen3_agent_sft.eval.agent_eval import run_eval
from qwen3_agent_sft.tools.rag_tools import retrieve_knowledge

app = FastAPI(title="urban-traffic-ops-agent")
executor = AgentExecutor()
registry = ToolRegistry()
trace_store = TraceStore()


class AgentQuery(BaseModel):
    query: str


class RagQuery(BaseModel):
    query: str
    top_k: int = 5


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/agent/query")
def agent_query(request: AgentQuery) -> dict:
    return executor.run(request.query).model_dump()


@app.get("/tools")
def tools() -> dict:
    return {"tools": registry.get_tool_schema_for_llm()}


@app.post("/rag/query")
def rag_query(request: RagQuery) -> dict:
    return retrieve_knowledge(request.query, request.top_k)


@app.get("/traces")
def traces() -> dict:
    return {"trace_ids": trace_store.list()}


@app.get("/traces/{trace_id}")
def trace(trace_id: str) -> dict:
    try:
        return trace_store.read(trace_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="trace not found") from exc


@app.post("/eval/run")
def eval_run() -> dict:
    return run_eval()


@app.get("/eval/summary")
def eval_summary() -> dict:
    path = Path("outputs/eval/agent_eval_summary.json")
    if not path.exists():
        raise HTTPException(status_code=404, detail="eval summary not found")
    return json.loads(path.read_text(encoding="utf-8"))
