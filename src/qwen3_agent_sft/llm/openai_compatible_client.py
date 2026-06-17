from __future__ import annotations

from typing import Any

import httpx

from qwen3_agent_sft.config import LLMSettings
from qwen3_agent_sft.llm.base import LLMClientBase
from qwen3_agent_sft.llm.schemas import LLMMessage, LLMResponse


class OpenAICompatibleClient(LLMClientBase):
    def __init__(self, settings: LLMSettings | None = None) -> None:
        self.settings = settings or LLMSettings.from_env()

    def chat(
        self,
        messages: list[LLMMessage],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> LLMResponse:
        url = self.settings.base_url.rstrip("/") + "/chat/completions"
        payload: dict[str, Any] = {
            "model": self.settings.model,
            "messages": [m.model_dump() for m in messages],
            "temperature": self.settings.temperature,
            "max_tokens": self.settings.max_tokens,
        }
        if tools is not None:
            payload["tools"] = tools
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice
        headers = {"Authorization": f"Bearer {self.settings.api_key}"}
        with httpx.Client(timeout=self.settings.timeout_seconds) as client:
            response = client.post(url, json=payload, headers=headers)
            response.raise_for_status()
        raw = response.json()
        content = raw["choices"][0]["message"].get("content") or ""
        return LLMResponse(content=content, model=self.settings.model, raw=raw)
