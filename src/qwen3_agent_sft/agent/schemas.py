from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ToolCall(BaseModel):
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class SafetyDecision(BaseModel):
    should_refuse: bool = False
    reason: str | None = None


class AgentPlan(BaseModel):
    type: Literal["tool_call", "workflow", "refusal"]
    intent: str
    tool_calls: list[ToolCall] = Field(default_factory=list)
    workflow_name: str | None = None
    arguments: dict[str, Any] = Field(default_factory=dict)
    safety: SafetyDecision = Field(default_factory=SafetyDecision)


class AgentResponse(BaseModel):
    trace_id: str
    answer: str
    plan: AgentPlan
    tool_outputs: list[dict[str, Any]] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)
    refused: bool = False
