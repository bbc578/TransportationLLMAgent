from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from qwen3_agent_sft.agent.prompt_templates import SYSTEM_PROMPT
from qwen3_agent_sft.training.schemas import RawCase


def load_raw_cases(path: str | Path) -> list[RawCase]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if "cases" in data:
        return [RawCase.model_validate(item) for item in data["cases"]]
    return _expand_templates(data)


def _expand_templates(data: dict[str, Any]) -> list[RawCase]:
    total = int(data.get("generator", {}).get("total_cases", 260))
    templates = data.get("templates", {})
    cases: list[RawCase] = []
    names = list(templates)
    for i in range(total):
        name = names[i % len(names)]
        tmpl = templates[name]
        query = tmpl["queries"][i % len(tmpl["queries"])]
        item = {
            "id": f"{name}_{i + 1:04d}",
            "user_query": query,
            "expected_intent": tmpl["expected_intent"],
            "expected_tool_calls": _patch_arguments(tmpl.get("expected_tool_calls", []), query),
            "expected_workflow": tmpl.get("expected_workflow"),
            "expected_answer_style": tmpl["expected_answer_style"],
            "should_refuse": tmpl.get("should_refuse", False),
            "refusal_reason": tmpl.get("refusal_reason"),
            "citation_required": tmpl.get("citation_required", False),
        }
        cases.append(RawCase.model_validate(item))
    return cases


def _patch_arguments(tool_calls: list[dict[str, Any]], query: str) -> list[dict[str, Any]]:
    patched = json.loads(json.dumps(tool_calls, ensure_ascii=False))
    for call in patched:
        args = call.setdefault("arguments", {})
        if call["name"] == "retrieve_knowledge":
            args["query"] = query
        region = _find_region(query)
        segment = _find_segment(query)
        event = _find_event(query)
        if region and "region_id" in args:
            args["region_id"] = region
        if segment and "road_segment_id" in args:
            args["road_segment_id"] = segment
        if event and "event_id" in args:
            args["event_id"] = event
        if "30" in query and "horizon_minutes" in args:
            args["horizon_minutes"] = 30
    return patched


def build_assistant_target(case: RawCase) -> dict[str, Any]:
    if case.should_refuse:
        return {
            "type": "refusal",
            "intent": "unsafe_operation",
            "tool_calls": [],
            "safety": {"should_refuse": True, "reason": case.refusal_reason},
        }
    if case.expected_workflow:
        args = _workflow_arguments(case.expected_workflow, case.user_query)
        return {
            "type": "workflow",
            "intent": case.expected_intent,
            "workflow_name": case.expected_workflow,
            "arguments": args,
            "safety": {"should_refuse": False, "reason": None},
        }
    return {
        "type": "tool_call",
        "intent": case.expected_intent,
        "tool_calls": case.expected_tool_calls,
        "safety": {"should_refuse": False, "reason": None},
    }


def build_records(cases: list[RawCase]) -> list[dict[str, Any]]:
    return [
        {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": case.user_query},
                {
                    "role": "assistant",
                    "content": json.dumps(build_assistant_target(case), ensure_ascii=False),
                },
            ],
            "metadata": case.model_dump(),
        }
        for case in cases
    ]


def write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_dataset(
    raw_cases_path: str | Path = "data/sft/raw_cases.yaml",
    train_path: str | Path = "data/sft/train.jsonl",
    val_path: str | Path = "data/sft/val.jsonl",
    stats_path: str | Path = "data/sft/dataset_stats.json",
) -> dict[str, Any]:
    cases = load_raw_cases(raw_cases_path)
    records = build_records(cases)
    split = int(len(records) * 0.9)
    write_jsonl(records[:split], Path(train_path))
    write_jsonl(records[split:], Path(val_path))
    stats = {
        "total": len(records),
        "train": split,
        "val": len(records) - split,
        "refusal": sum(1 for case in cases if case.should_refuse),
        "workflow": sum(1 for case in cases if case.expected_workflow),
        "tool_call": sum(1 for case in cases if not case.expected_workflow and not case.should_refuse),
    }
    Path(stats_path).write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    return stats


def _workflow_arguments(workflow_name: str, query: str) -> dict[str, Any]:
    segment = _find_segment(query) or "S001"
    region = _find_region(query)
    event = _find_event(query) or "E001"
    if workflow_name == "congestion_diagnosis_workflow":
        return {"road_segment_id": segment, "region_id": region}
    if workflow_name == "incident_response_workflow":
        return {"event_id": event, "road_segment_id": _find_segment(query)}
    if workflow_name == "traffic_daily_report_workflow":
        return {"region_id": region, "time_range": "sample_day"}
    if workflow_name == "signal_optimization_assistant_workflow":
        return {"road_segment_id": segment, "intersection_id": None}
    if workflow_name == "traffic_knowledge_qa_workflow":
        return {"query": query}
    return {}


def _find_region(text: str) -> str | None:
    for region in ["R1", "R2", "R3", "R4"]:
        if region in text:
            return region
    return None


def _find_segment(text: str) -> str | None:
    for idx in range(1, 31):
        segment = f"S{idx:03d}"
        if segment in text:
            return segment
    return None


def _find_event(text: str) -> str | None:
    for idx in range(1, 41):
        event = f"E{idx:03d}"
        if event in text:
            return event
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-cases-path", default="data/sft/raw_cases.yaml")
    parser.add_argument("--train-path", default="data/sft/train.jsonl")
    parser.add_argument("--val-path", default="data/sft/val.jsonl")
    args = parser.parse_args()
    stats = build_dataset(args.raw_cases_path, args.train_path, args.val_path)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
