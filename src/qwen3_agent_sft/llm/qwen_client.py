from __future__ import annotations

from qwen3_agent_sft.config import LLMSettings
from qwen3_agent_sft.llm.base import LLMClientBase
from qwen3_agent_sft.llm.mock_client import MockLLMClient
from qwen3_agent_sft.llm.openai_compatible_client import OpenAICompatibleClient


def build_llm_client(settings: LLMSettings | None = None) -> LLMClientBase:
    settings = settings or LLMSettings.from_env()
    if settings.provider == "mock":
        return MockLLMClient()
    if settings.provider in {"qwen_openai_compatible", "openai_compatible", "vllm", "sglang"}:
        return OpenAICompatibleClient(settings)
    raise ValueError(f"Unsupported LLM_PROVIDER: {settings.provider}")
