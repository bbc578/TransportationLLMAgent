from __future__ import annotations

from typing import Any

from qwen3_agent_sft.agent.tool_registry import ToolRegistry
from qwen3_agent_sft.workflows.traffic_workflows import (
    congestion_diagnosis_workflow,
    incident_response_workflow,
    signal_optimization_assistant_workflow,
    traffic_daily_report_workflow,
    traffic_knowledge_qa_workflow,
)


class WorkflowEngine:
    def __init__(self, registry: ToolRegistry | None = None) -> None:
        self.registry = registry or ToolRegistry()

    def run(
        self, workflow_name: str, arguments: dict[str, Any]
    ) -> tuple[list[dict[str, Any]], list[str], list[dict[str, Any]]]:
        if workflow_name == "congestion_diagnosis_workflow":
            return congestion_diagnosis_workflow(self.registry, **arguments)
        if workflow_name == "incident_response_workflow":
            return incident_response_workflow(self.registry, **arguments)
        if workflow_name == "traffic_daily_report_workflow":
            return traffic_daily_report_workflow(self.registry, **arguments)
        if workflow_name == "signal_optimization_assistant_workflow":
            return signal_optimization_assistant_workflow(self.registry, **arguments)
        if workflow_name == "traffic_knowledge_qa_workflow":
            return traffic_knowledge_qa_workflow(self.registry, **arguments)
        return [{"message": f"Unknown workflow: {workflow_name}", "error": "unknown_workflow"}], [], []
