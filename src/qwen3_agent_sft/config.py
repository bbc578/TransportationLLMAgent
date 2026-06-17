from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[2]


def env(name: str, default: str | None = None) -> str | None:
    load_dotenv(ROOT_DIR / ".env")
    return os.getenv(name, default)


def expand_env(value: Any) -> Any:
    if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
        return env(value[2:-1], value)
    if isinstance(value, list):
        return [expand_env(item) for item in value]
    if isinstance(value, dict):
        return {key: expand_env(item) for key, item in value.items()}
    return value


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return expand_env(yaml.safe_load(f) or {})


@dataclass(frozen=True)
class LLMSettings:
    provider: str = "mock"
    base_url: str = "http://127.0.0.1:8001/v1"
    api_key: str = "EMPTY"
    model: str = "qwen3-agent-lora"
    temperature: float = 0.2
    max_tokens: int = 2048
    timeout_seconds: float = 120

    @classmethod
    def from_env(cls) -> LLMSettings:
        return cls(
            provider=env("LLM_PROVIDER", "mock") or "mock",
            base_url=env("LLM_BASE_URL", "http://127.0.0.1:8001/v1") or "",
            api_key=env("LLM_API_KEY", "EMPTY") or "EMPTY",
            model=env("LLM_MODEL", "qwen3-agent-lora") or "qwen3-agent-lora",
            temperature=float(env("LLM_TEMPERATURE", "0.2") or "0.2"),
            max_tokens=int(env("LLM_MAX_TOKENS", "2048") or "2048"),
            timeout_seconds=float(env("LLM_TIMEOUT_SECONDS", "120") or "120"),
        )
