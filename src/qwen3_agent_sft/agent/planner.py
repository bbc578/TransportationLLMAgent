from __future__ import annotations

import json
import re

from pydantic import ValidationError

from qwen3_agent_sft.agent.prompt_templates import SYSTEM_PROMPT
from qwen3_agent_sft.agent.safety import check_safety
from qwen3_agent_sft.agent.schemas import AgentPlan
from qwen3_agent_sft.agent.tool_registry import ToolRegistry
from qwen3_agent_sft.llm.base import LLMClientBase
from qwen3_agent_sft.llm.schemas import LLMMessage

WORKFLOWS = {
    "congestion_diagnosis_workflow",
    "incident_response_workflow",
    "traffic_daily_report_workflow",
    "signal_optimization_assistant_workflow",
    "traffic_knowledge_qa_workflow",
}


class RuleBasedPlanner:
    def plan(self, user_query: str) -> AgentPlan:
        safety = check_safety(user_query)
        if safety.should_refuse:
            return AgentPlan(type="refusal", intent="unsafe_operation", safety=safety)

        region_id = _extract_region(user_query)
        segment_id = _extract_segment(user_query)
        event_id = _extract_event(user_query)
        horizon = _extract_horizon(user_query)
        model_name = _extract_model_name(user_query)

        if any(word in user_query for word in ["日报", "运行报告", "早高峰"]):
            return AgentPlan(
                type="workflow",
                intent="traffic_daily_report",
                workflow_name="traffic_daily_report_workflow",
                arguments={"region_id": region_id, "time_range": "sample_day"},
            )
        if any(word in user_query for word in ["信控优化", "配时优化", "排队严重", "绿信比建议"]):
            return AgentPlan(
                type="workflow",
                intent="signal_optimization_assistant",
                workflow_name="signal_optimization_assistant_workflow",
                arguments={"road_segment_id": segment_id, "intersection_id": None},
            )
        if any(word in user_query for word in ["为什么", "拥堵原因", "拥堵诊断"]):
            return AgentPlan(
                type="workflow",
                intent="congestion_diagnosis",
                workflow_name="congestion_diagnosis_workflow",
                arguments={"road_segment_id": segment_id or "S001", "region_id": region_id},
            )
        if any(word in user_query for word in ["处置", "响应", "事故", "事件处理"]):
            return AgentPlan(
                type="workflow",
                intent="incident_response_assistant",
                workflow_name="incident_response_workflow",
                arguments={"event_id": event_id or "E001", "road_segment_id": segment_id},
            )
        if any(word in user_query for word in ["处置记录", "创建记录", "行动记录"]):
            return AgentPlan(
                type="tool_call",
                intent="incident_response_assistant",
                tool_calls=[
                    {
                        "name": "create_incident_action_record",
                        "arguments": {
                            "event_id": event_id or "E001",
                            "action_type": "demo_response",
                            "description": user_query,
                            "operator": "demo_user",
                        },
                    }
                ],
            )
        if any(word in user_query for word in ["最拥堵", "拥堵排名", "top"]):
            return AgentPlan(
                type="tool_call",
                intent="congestion_ranking",
                tool_calls=[
                    {
                        "name": "get_top_congested_segments",
                        "arguments": {"k": 5, "region_id": region_id},
                    }
                ],
            )
        if any(word in user_query for word in ["路段信息", "基础信息", "上下游", "车道"]):
            return AgentPlan(
                type="tool_call",
                intent="road_segment_info",
                tool_calls=[
                    {
                        "name": "query_road_segment_info",
                        "arguments": {"road_segment_id": segment_id or "S001"},
                    }
                ],
            )
        if any(word in user_query for word in ["事件", "事故", "施工", "故障"]):
            return AgentPlan(
                type="tool_call",
                intent="traffic_event_query",
                tool_calls=[
                    {
                        "name": "query_traffic_events",
                        "arguments": {
                            "region_id": region_id,
                            "road_segment_id": segment_id,
                            "status": None,
                        },
                    }
                ],
            )
        if any(word in user_query for word in ["预测", "未来", "风险"]):
            return AgentPlan(
                type="tool_call",
                intent="traffic_forecast_query",
                tool_calls=[
                    {
                        "name": "query_traffic_forecast",
                        "arguments": {
                            "road_segment_id": segment_id,
                            "horizon_minutes": horizon,
                            "model_name": model_name,
                        },
                    }
                ],
            )
        if any(word in user_query for word in ["信号", "配时", "绿信比", "周期"]):
            return AgentPlan(
                type="tool_call",
                intent="signal_plan_query",
                tool_calls=[
                    {
                        "name": "query_signal_plan",
                        "arguments": {"road_segment_id": segment_id, "intersection_id": None},
                    }
                ],
            )
        if any(word in user_query for word in ["原则", "模板", "流程", "数字孪生", "知识"]):
            return AgentPlan(
                type="workflow",
                intent="knowledge_qa",
                workflow_name="traffic_knowledge_qa_workflow",
                arguments={"query": user_query},
            )
        return AgentPlan(
            type="tool_call",
            intent="traffic_metrics_query",
            tool_calls=[
                {
                    "name": "query_traffic_metrics",
                    "arguments": {
                        "region_id": region_id,
                        "road_segment_id": segment_id,
                        "start_time": None,
                        "end_time": None,
                    },
                }
            ],
        )


class LLMPlanner:
    def __init__(self, llm: LLMClientBase, registry: ToolRegistry | None = None) -> None:
        self.llm = llm
        self.registry = registry or ToolRegistry()
        self.fallback = RuleBasedPlanner()

    def plan(self, user_query: str) -> AgentPlan:
        safety = check_safety(user_query)
        if safety.should_refuse:
            return AgentPlan(type="refusal", intent="unsafe_operation", safety=safety)
        try:
            response = self.llm.chat(
                [
                    LLMMessage(role="system", content=SYSTEM_PROMPT),
                    LLMMessage(role="user", content=user_query),
                ],
                tools=self.registry.get_tool_schema_for_llm(),
            )
            payload = json.loads(_strip_json(response.content))
            plan = AgentPlan.model_validate(payload)
            self._validate_plan(plan)
            return plan
        except (json.JSONDecodeError, ValidationError, ValueError):
            return self.fallback.plan(user_query)

    def _validate_plan(self, plan: AgentPlan) -> None:
        if plan.type == "tool_call":
            for call in plan.tool_calls:
                ok, error = self.registry.validate_tool_call(call.name, call.arguments)
                if not ok:
                    raise ValueError(error)
        if plan.type == "workflow" and plan.workflow_name not in WORKFLOWS:
            raise ValueError(f"unknown_workflow: {plan.workflow_name}")


def _strip_json(content: str) -> str:
    content = content.strip()
    if content.startswith("```"):
        content = content.strip("`")
        content = content.removeprefix("json").strip()
    return content


def _extract_region(text: str) -> str | None:
    match = re.search(r"\bR[1-4]\b", text)
    return match.group(0) if match else None


def _extract_segment(text: str) -> str | None:
    match = re.search(r"\bS\d{3}\b", text)
    return match.group(0) if match else None


def _extract_event(text: str) -> str | None:
    match = re.search(r"\bE\d{3}\b", text)
    return match.group(0) if match else None


def _extract_horizon(text: str) -> int | None:
    match = re.search(r"\b(15|30|60)\b", text)
    return int(match.group(1)) if match else None


def _extract_model_name(text: str) -> str | None:
    for name in ["LastValue", "GraphWaveNetFull", "STGCNFull"]:
        if name in text:
            return name
    return None
