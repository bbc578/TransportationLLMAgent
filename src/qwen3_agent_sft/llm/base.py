from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from qwen3_agent_sft.llm.schemas import LLMMessage, LLMResponse


class LLMClientBase(ABC):
    @abstractmethod
    def chat(
        self,
        messages: list[LLMMessage],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> LLMResponse:
        raise NotImplementedError
