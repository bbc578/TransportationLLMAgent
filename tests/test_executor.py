from pathlib import Path

from qwen3_agent_sft.agent.executor import AgentExecutor


def test_executor_saves_trace():
    response = AgentExecutor().run("S001 为什么拥堵？")
    assert response.trace_id
    assert Path(f"outputs/traces/{response.trace_id}.json").exists()
