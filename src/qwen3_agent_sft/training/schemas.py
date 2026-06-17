from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class RawCase(BaseModel):
    id: str
    user_query: str
    expected_intent: str
    expected_tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    expected_workflow: str | None = None
    expected_answer_style: str
    should_refuse: bool = False
    refusal_reason: str | None = None
    citation_required: bool = False
