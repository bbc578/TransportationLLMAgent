from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class LLMMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str


class LLMToolCall(BaseModel):
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class LLMResponse(BaseModel):
    content: str
    model: str = "mock"
    raw: dict[str, Any] = Field(default_factory=dict)
