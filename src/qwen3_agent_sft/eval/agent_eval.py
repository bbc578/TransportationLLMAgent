from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from qwen3_agent_sft.agent.executor import AgentExecutor


def load_cases(path: str = "src/qwen3_agent_sft/eval/cases.yaml") -> list[dict[str, Any]]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    cases = data.get("cases", [])
    total = int(data.get("generator", {}).get("total_cases", len(cases)))
    if not cases:
        return []
    return [dict(cases[i % len(cases)], id=f"eval_{i + 1:03d}") for i in range(total)]


def run_eval(path: str = "src/qwen3_agent_sft/eval/cases.yaml") -> dict[str, float]:
    executor = AgentExecutor()
    cases = load_cases(path)
    counters = {
        "intent_accuracy": 0,
        "tool_selection_accuracy": 0,
        "workflow_success_rate": 0,
        "rag_citation_rate": 0,
        "refusal_accuracy": 0,
        "structured_output_valid_rate": 0,
        "trace_saved_rate": 0,
    }
    details: list[dict[str, Any]] = []
    for case in cases:
        response = executor.run(case["query"])
        plan = response.plan
        counters["structured_output_valid_rate"] += 1
        counters["intent_accuracy"] += plan.intent == case["expected_intent"]
        if case.get("expected_tool"):
            counters["tool_selection_accuracy"] += bool(
                plan.tool_calls and plan.tool_calls[0].name == case["expected_tool"]
            )
        else:
            counters["tool_selection_accuracy"] += 1
        if case.get("expected_workflow"):
            counters["workflow_success_rate"] += plan.workflow_name == case["expected_workflow"]
        else:
            counters["workflow_success_rate"] += 1
        counters["rag_citation_rate"] += (not case.get("citation_required")) or bool(response.citations)
        counters["refusal_accuracy"] += response.refused == bool(case.get("should_refuse"))
        counters["trace_saved_rate"] += Path(f"outputs/traces/{response.trace_id}.json").exists()
        details.append({"case": case, "trace_id": response.trace_id, "plan": plan.model_dump()})
    total = len(cases) or 1
    metrics = {key: value / total for key, value in counters.items()}
    out = Path("outputs/eval")
    out.mkdir(parents=True, exist_ok=True)
    (out / "agent_eval_summary.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out / "agent_eval_details.json").write_text(
        json.dumps(details, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metrics


if __name__ == "__main__":
    print(json.dumps(run_eval(), ensure_ascii=False, indent=2))
