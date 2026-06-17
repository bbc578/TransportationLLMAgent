from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from qwen3_agent_sft.agent.executor import AgentExecutor
from qwen3_agent_sft.agent.tool_registry import ToolRegistry
from qwen3_agent_sft.eval.agent_eval import run_eval
from qwen3_agent_sft.tools.rag_tools import retrieve_knowledge

st.set_page_config(page_title="Urban Traffic Ops Agent", layout="wide")
st.title("城市交通运行分析与事件处置 Agent")

page = st.sidebar.radio(
    "Page",
    [
        "Overview",
        "Agent Chat",
        "Tool Calling Demo",
        "RAG Demo",
        "Workflow Demo",
        "Trace Viewer",
        "Evaluation",
        "Training Status",
    ],
)

executor = AgentExecutor()
registry = ToolRegistry()

if page == "Overview":
    st.write(
        "sample-traffic-data LLM-Agent platform with Tool Calling, RAG, Workflow, Trace, Eval and Qwen3 LoRA SFT."
    )
elif page == "Agent Chat":
    query = st.text_input("Query", "S001 为什么拥堵？")
    if st.button("Run"):
        st.json(executor.run(query).model_dump())
elif page == "Tool Calling Demo":
    tool = st.selectbox("Tool", registry.list_tools())
    args = st.text_area("Arguments JSON", "{}")
    if st.button("Execute"):
        st.json(registry.execute_tool(tool, json.loads(args)))
elif page == "RAG Demo":
    query = st.text_input("RAG Query", "信控优化建议需要注意什么？")
    if st.button("Retrieve"):
        st.json(retrieve_knowledge(query))
elif page == "Workflow Demo":
    query = st.text_input("Workflow Query", "生成一份早高峰交通运行日报")
    if st.button("Run Workflow"):
        st.json(executor.run(query).model_dump())
elif page == "Trace Viewer":
    trace_dir = Path("outputs/traces")
    traces = sorted(trace_dir.glob("*.json"))
    selected = st.selectbox("Trace", [p.name for p in traces])
    if selected:
        st.json(json.loads((trace_dir / selected).read_text(encoding="utf-8")))
elif page == "Evaluation":
    if st.button("Run Eval"):
        st.json(run_eval())
elif page == "Training Status":
    st.code(
        "\n".join(
            [
                "SFT target: AgentPlan JSON for traffic tool/workflow selection.",
                "QWEN3_BASE_MODEL_PATH=/data/models/Qwen3",
                "QWEN3_LORA_OUTPUT_DIR=/data/outputs/qwen3-agent-lora",
                "QWEN3_MERGED_MODEL_DIR=/data/outputs/qwen3-agent-merged",
                "bash scripts/train_qwen3_lora.sh",
                "bash scripts/merge_lora.sh",
                "bash scripts/serve_qwen3_vllm.sh",
            ]
        )
    )
