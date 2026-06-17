from __future__ import annotations

from typing import Any

from qwen3_agent_sft.agent.tool_registry import ToolRegistry


def _run_steps(
    registry: ToolRegistry, calls: list[tuple[str, dict[str, Any]]]
) -> tuple[list[dict[str, Any]], list[str], list[dict[str, Any]]]:
    outputs: list[dict[str, Any]] = []
    steps: list[dict[str, Any]] = []
    citations: list[str] = []
    for name, args in calls:
        clean_args = {key: value for key, value in args.items() if value is not None}
        output = registry.execute_tool(name, clean_args)
        outputs.append(output)
        steps.append({"tool": name, "arguments": clean_args, "output": output})
        citations.extend(output.get("citations", []))
    return outputs, citations, steps


def congestion_diagnosis_workflow(
    registry: ToolRegistry, road_segment_id: str, region_id: str | None = None
) -> tuple[list[dict[str, Any]], list[str], list[dict[str, Any]]]:
    return _run_steps(
        registry,
        [
            ("query_road_segment_info", {"road_segment_id": road_segment_id}),
            ("query_traffic_metrics", {"road_segment_id": road_segment_id, "region_id": region_id}),
            ("query_traffic_events", {"road_segment_id": road_segment_id, "region_id": region_id}),
            ("retrieve_knowledge", {"query": "congestion diagnosis guide bottleneck incident", "top_k": 5}),
        ],
    )


def incident_response_workflow(
    registry: ToolRegistry, event_id: str, road_segment_id: str | None = None
) -> tuple[list[dict[str, Any]], list[str], list[dict[str, Any]]]:
    return _run_steps(
        registry,
        [
            ("query_traffic_events", {"road_segment_id": road_segment_id}),
            ("query_traffic_metrics", {"road_segment_id": road_segment_id}),
            ("retrieve_knowledge", {"query": "traffic incident response process", "top_k": 5}),
        ],
    )


def traffic_daily_report_workflow(
    registry: ToolRegistry, region_id: str | None = None, time_range: str | None = None
) -> tuple[list[dict[str, Any]], list[str], list[dict[str, Any]]]:
    return _run_steps(
        registry,
        [
            ("query_traffic_metrics", {"region_id": region_id}),
            ("get_top_congested_segments", {"k": 5, "region_id": region_id}),
            ("query_traffic_events", {"region_id": region_id}),
            ("retrieve_knowledge", {"query": "traffic operation report template", "top_k": 5}),
        ],
    )


def signal_optimization_assistant_workflow(
    registry: ToolRegistry,
    road_segment_id: str | None = None,
    intersection_id: str | None = None,
) -> tuple[list[dict[str, Any]], list[str], list[dict[str, Any]]]:
    return _run_steps(
        registry,
        [
            ("query_traffic_metrics", {"road_segment_id": road_segment_id}),
            ("query_signal_plan", {"road_segment_id": road_segment_id, "intersection_id": intersection_id}),
            ("query_traffic_events", {"road_segment_id": road_segment_id}),
            ("retrieve_knowledge", {"query": "signal control principles simulation manual review", "top_k": 5}),
        ],
    )


def traffic_knowledge_qa_workflow(
    registry: ToolRegistry, query: str
) -> tuple[list[dict[str, Any]], list[str], list[dict[str, Any]]]:
    return _run_steps(registry, [("retrieve_knowledge", {"query": query, "top_k": 5})])
