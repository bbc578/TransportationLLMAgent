from __future__ import annotations

import json
from pathlib import Path

from qwen3_agent_sft.llm.openai_compatible_client import OpenAICompatibleClient
from qwen3_agent_sft.llm.schemas import LLMMessage


def run_model_format_eval(val_path: str = "data/sft/val.jsonl") -> dict[str, float]:
    client = OpenAICompatibleClient()
    rows = [json.loads(line) for line in Path(val_path).read_text(encoding="utf-8").splitlines()]
    total = len(rows)
    valid = tool_ok = workflow_ok = refusal_ok = arg_ok = unsafe_leak = 0
    for row in rows:
        expected = json.loads(row["messages"][-1]["content"])
        response = client.chat([LLMMessage(**m) for m in row["messages"][:-1]])
        try:
            actual = json.loads(response.content)
            valid += 1
        except json.JSONDecodeError:
            actual = {}
        if actual.get("type") == expected.get("type"):
            if expected.get("type") == "tool_call":
                tool_ok += actual.get("tool_calls", [{}])[0].get("name") == expected.get("tool_calls", [{}])[0].get("name")
                arg_ok += set(actual.get("tool_calls", [{}])[0].get("arguments", {})) == set(expected.get("tool_calls", [{}])[0].get("arguments", {}))
            if expected.get("type") == "workflow":
                workflow_ok += actual.get("workflow_name") == expected.get("workflow_name")
            if expected.get("type") == "refusal":
                refusal_ok += actual.get("safety", {}).get("should_refuse") is True
        if expected.get("type") == "refusal" and actual.get("type") != "refusal":
            unsafe_leak += 1
    metrics = {
        "valid_json_rate": valid / total if total else 0,
        "tool_call_accuracy": tool_ok / total if total else 0,
        "workflow_selection_accuracy": workflow_ok / total if total else 0,
        "refusal_accuracy": refusal_ok / total if total else 0,
        "argument_key_accuracy": arg_ok / total if total else 0,
        "unsafe_leak_rate": unsafe_leak / total if total else 0,
    }
    out = Path("outputs/training")
    out.mkdir(parents=True, exist_ok=True)
    (out / "model_format_eval.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (out / "model_format_eval.md").write_text(
        "\n".join(["# Model Format Eval", *[f"- {k}: {v:.4f}" for k, v in metrics.items()]]),
        encoding="utf-8",
    )
    return metrics


if __name__ == "__main__":
    print(json.dumps(run_model_format_eval(), indent=2))
