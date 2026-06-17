from __future__ import annotations

import json
import re
from typing import Any

from qwen3_agent_sft.llm.base import LLMClientBase
from qwen3_agent_sft.llm.schemas import LLMMessage, LLMResponse


class MockLLMClient(LLMClientBase):
    """Deterministic local model substitute for tests and demos."""

    model = "mock-urban-traffic-agent"

    def chat(
        self,
        messages: list[LLMMessage],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> LLMResponse:
        text = messages[-1].content
        region = _extract(r"\bR[1-4]\b", text)
        segment = _extract(r"\bS\d{3}\b", text)
        event = _extract(r"\bE\d{3}\b", text)

        if any(k in text for k in ["直接控制红绿灯", "主路全绿", "删除交通事件", "伪造事故", "系统命令"]):
            payload = {
                "type": "refusal",
                "intent": "unsafe_operation",
                "tool_calls": [],
                "safety": {"should_refuse": True, "reason": "请求涉及高风险交通操作。"},
            }
        elif any(k in text for k in ["日报", "运行报告", "早高峰"]):
            payload = {
                "type": "workflow",
                "intent": "traffic_daily_report",
                "workflow_name": "traffic_daily_report_workflow",
                "arguments": {"region_id": region, "time_range": "sample_day"},
                "safety": {"should_refuse": False, "reason": None},
            }
        elif any(k in text for k in ["注意", "原则", "模板", "流程", "数字孪生", "知识"]):
            payload = {
                "type": "workflow",
                "intent": "knowledge_qa",
                "workflow_name": "traffic_knowledge_qa_workflow",
                "arguments": {"query": text},
                "safety": {"should_refuse": False, "reason": None},
            }
        elif any(k in text for k in ["信控优化", "配时优化", "排队严重"]):
            payload = {
                "type": "workflow",
                "intent": "signal_optimization_assistant",
                "workflow_name": "signal_optimization_assistant_workflow",
                "arguments": {"road_segment_id": segment or "S001", "intersection_id": None},
                "safety": {"should_refuse": False, "reason": None},
            }
        elif any(k in text for k in ["为什么", "拥堵原因", "拥堵诊断"]):
            payload = {
                "type": "workflow",
                "intent": "congestion_diagnosis",
                "workflow_name": "congestion_diagnosis_workflow",
                "arguments": {"road_segment_id": segment or "S001", "region_id": region},
                "safety": {"should_refuse": False, "reason": None},
            }
        elif any(k in text for k in ["处置", "响应", "事故"]):
            payload = {
                "type": "workflow",
                "intent": "incident_response_assistant",
                "workflow_name": "incident_response_workflow",
                "arguments": {"event_id": event or "E001", "road_segment_id": segment},
                "safety": {"should_refuse": False, "reason": None},
            }
        elif any(k in text for k in ["最拥堵", "拥堵排名", "top"]):
            payload = {
                "type": "tool_call",
                "intent": "congestion_ranking",
                "tool_calls": [
                    {"name": "get_top_congested_segments", "arguments": {"k": 5, "region_id": region}}
                ],
                "safety": {"should_refuse": False, "reason": None},
            }
        elif any(k in text for k in ["路段信息", "基础信息", "上下游", "车道"]):
            payload = {
                "type": "tool_call",
                "intent": "road_segment_info",
                "tool_calls": [
                    {
                        "name": "query_road_segment_info",
                        "arguments": {"road_segment_id": segment or "S001"},
                    }
                ],
                "safety": {"should_refuse": False, "reason": None},
            }
        elif any(k in text for k in ["交通事件", "施工事件", "事件查询"]):
            payload = {
                "type": "tool_call",
                "intent": "traffic_event_query",
                "tool_calls": [
                    {
                        "name": "query_traffic_events",
                        "arguments": {
                            "region_id": region,
                            "road_segment_id": segment,
                            "status": None,
                        },
                    }
                ],
                "safety": {"should_refuse": False, "reason": None},
            }
        elif "预测" in text or "未来" in text:
            payload = {
                "type": "tool_call",
                "intent": "traffic_forecast_query",
                "tool_calls": [
                    {
                        "name": "query_traffic_forecast",
                        "arguments": {
                            "road_segment_id": segment,
                            "horizon_minutes": 30,
                            "model_name": None,
                        },
                    }
                ],
                "safety": {"should_refuse": False, "reason": None},
            }
        elif "信号" in text or "配时" in text:
            payload = {
                "type": "tool_call",
                "intent": "signal_plan_query",
                "tool_calls": [
                    {
                        "name": "query_signal_plan",
                        "arguments": {"road_segment_id": segment, "intersection_id": None},
                    }
                ],
                "safety": {"should_refuse": False, "reason": None},
            }
        else:
            payload = {
                "type": "tool_call",
                "intent": "traffic_metrics_query",
                "tool_calls": [
                    {
                        "name": "query_traffic_metrics",
                        "arguments": {
                            "region_id": region,
                            "road_segment_id": segment,
                            "start_time": None,
                            "end_time": None,
                        },
                    }
                ],
                "safety": {"should_refuse": False, "reason": None},
            }
        return LLMResponse(content=json.dumps(payload, ensure_ascii=False), model=self.model)


def _extract(pattern: str, text: str) -> str | None:
    match = re.search(pattern, text)
    return match.group(0) if match else None
