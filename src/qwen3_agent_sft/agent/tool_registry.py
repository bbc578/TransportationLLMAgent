from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ValidationError

from qwen3_agent_sft.agent.tool_schemas import (
    CreateIncidentActionRecordInput,
    CreateIncidentActionRecordOutput,
    GetTopCongestedSegmentsInput,
    GetTopCongestedSegmentsOutput,
    QueryRoadSegmentInfoInput,
    QueryRoadSegmentInfoOutput,
    QuerySignalPlanInput,
    QuerySignalPlanOutput,
    QueryTrafficEventsInput,
    QueryTrafficEventsOutput,
    QueryTrafficForecastInput,
    QueryTrafficForecastOutput,
    QueryTrafficMetricsInput,
    QueryTrafficMetricsOutput,
    RetrieveKnowledgeInput,
    RetrieveKnowledgeOutput,
)
from qwen3_agent_sft.tools import (
    create_incident_action_record,
    get_top_congested_segments,
    query_road_segment_info,
    query_signal_plan,
    query_traffic_events,
    query_traffic_forecast,
    query_traffic_metrics,
    retrieve_knowledge,
)


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_model: type[BaseModel]
    output_model: type[BaseModel]
    function: Callable[..., dict[str, Any]]
    read_only: bool
    category: str
    safety_notes: str


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}
        self.register(
            "query_traffic_metrics",
            "Query sample traffic metrics by region, road segment and time range.",
            QueryTrafficMetricsInput,
            QueryTrafficMetricsOutput,
            query_traffic_metrics,
            read_only=True,
            category="traffic_data",
            safety_notes="Read-only sample traffic metrics. Not connected to a real traffic system.",
        )
        self.register(
            "get_top_congested_segments",
            "Return the most congested sample road segments by congestion index.",
            GetTopCongestedSegmentsInput,
            GetTopCongestedSegmentsOutput,
            get_top_congested_segments,
            read_only=True,
            category="traffic_analysis",
            safety_notes="Ranking is based on sample traffic data only.",
        )
        self.register(
            "query_road_segment_info",
            "Query sample road segment topology and signalization metadata.",
            QueryRoadSegmentInfoInput,
            QueryRoadSegmentInfoOutput,
            query_road_segment_info,
            read_only=True,
            category="traffic_data",
            safety_notes="Read-only sample road network metadata.",
        )
        self.register(
            "query_traffic_events",
            "Query sample traffic events by region, road segment or status.",
            QueryTrafficEventsInput,
            QueryTrafficEventsOutput,
            query_traffic_events,
            read_only=True,
            category="traffic_event",
            safety_notes="Read-only sample events. Does not modify real incident records.",
        )
        self.register(
            "query_traffic_forecast",
            "Query sample traffic forecasts for congestion risk analysis.",
            QueryTrafficForecastInput,
            QueryTrafficForecastOutput,
            query_traffic_forecast,
            read_only=True,
            category="traffic_forecast",
            safety_notes="Forecasts are sample or external-model demo outputs.",
        )
        self.register(
            "query_signal_plan",
            "Query read-only sample signal timing plans.",
            QuerySignalPlanInput,
            QuerySignalPlanOutput,
            query_signal_plan,
            read_only=True,
            category="signal_control",
            safety_notes="Read-only. The Agent must not issue real signal control commands.",
        )
        self.register(
            "create_incident_action_record",
            "Create a sample traffic incident action record under outputs/action_records.",
            CreateIncidentActionRecordInput,
            CreateIncidentActionRecordOutput,
            create_incident_action_record,
            read_only=False,
            category="traffic_event",
            safety_notes="Writes only a local sample action record; does not modify source CSV or real systems.",
        )
        self.register(
            "retrieve_knowledge",
            "Retrieve chunks from the local sample traffic knowledge base.",
            RetrieveKnowledgeInput,
            RetrieveKnowledgeOutput,
            retrieve_knowledge,
            read_only=True,
            category="rag",
            safety_notes="Knowledge retrieval only. No external network access.",
        )

    def register(
        self,
        name: str,
        description: str,
        input_model: type[BaseModel],
        output_model: type[BaseModel],
        function: Callable[..., dict[str, Any]],
        read_only: bool,
        category: str,
        safety_notes: str,
    ) -> None:
        self._tools[name] = ToolSpec(
            name=name,
            description=description,
            input_model=input_model,
            output_model=output_model,
            function=function,
            read_only=read_only,
            category=category,
            safety_notes=safety_notes,
        )

    def list_tools(self) -> list[str]:
        return sorted(self._tools)

    def get_tool_schema_for_llm(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": spec.name,
                    "description": spec.description,
                    "parameters": spec.input_model.model_json_schema(),
                },
                "metadata": {
                    "read_only": spec.read_only,
                    "category": spec.category,
                    "safety_notes": spec.safety_notes,
                },
            }
            for spec in self._tools.values()
        ]

    def validate_tool_call(self, name: str, arguments: dict[str, Any]) -> tuple[bool, str | None]:
        spec = self._tools.get(name)
        if spec is None:
            return False, f"unknown_tool: {name}"
        try:
            spec.input_model.model_validate(arguments)
        except ValidationError as exc:
            return False, f"input_validation_error: {exc.errors()}"
        return True, None

    def execute_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        spec = self._tools.get(name)
        if spec is None:
            return {"message": f"Unknown tool: {name}", "error": f"unknown_tool: {name}"}
        try:
            validated_input = spec.input_model.model_validate(arguments)
        except ValidationError as exc:
            return {
                "message": "Tool input validation failed.",
                "error": "input_validation_error",
                "details": exc.errors(),
            }
        raw_output = spec.function(**validated_input.model_dump())
        try:
            return spec.output_model.model_validate(raw_output).model_dump()
        except ValidationError as exc:
            return {
                "message": "Tool output validation failed.",
                "error": "output_validation_error",
                "details": exc.errors(),
                "raw_output": raw_output,
            }
