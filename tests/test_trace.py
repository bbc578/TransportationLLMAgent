from pathlib import Path

from qwen3_agent_sft.agent.trace import TraceStore


def test_trace_store_round_trip():
    out = Path("outputs/test_artifacts/traces")
    store = TraceStore(out)
    store.save({"trace_id": "abc", "final_answer": "ok"})
    assert store.read("abc")["final_answer"] == "ok"
