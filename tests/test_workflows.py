from qwen3_agent_sft.agent.tool_registry import ToolRegistry
from qwen3_agent_sft.workflows.engine import WorkflowEngine


def test_policy_workflow():
    outputs, citations, steps = WorkflowEngine(ToolRegistry()).run(
        "traffic_knowledge_qa_workflow", {"query": "信控优化建议需要注意什么？"}
    )
    assert outputs
    assert citations
    assert steps


def test_congestion_diagnosis_workflow():
    outputs, citations, steps = WorkflowEngine(ToolRegistry()).run(
        "congestion_diagnosis_workflow", {"road_segment_id": "S001", "region_id": None}
    )
    assert outputs
    assert citations
    assert steps


def test_daily_report_and_signal_workflows():
    engine = WorkflowEngine(ToolRegistry())
    daily_outputs, daily_citations, _ = engine.run(
        "traffic_daily_report_workflow", {"region_id": "R1", "time_range": "sample_day"}
    )
    signal_outputs, signal_citations, _ = engine.run(
        "signal_optimization_assistant_workflow",
        {"road_segment_id": "S001", "intersection_id": None},
    )
    assert daily_outputs and daily_citations
    assert signal_outputs and signal_citations
