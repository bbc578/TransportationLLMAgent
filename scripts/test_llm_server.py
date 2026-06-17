from __future__ import annotations

import os
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main() -> None:
    base_url = os.getenv("LLM_BASE_URL", "http://127.0.0.1:8001/v1").rstrip("/")
    model = os.getenv("LLM_MODEL", "qwen3-agent-lora")
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "今天哪些路段最拥堵？请输出 AgentPlan JSON。"}],
        "temperature": 0.1,
        "max_tokens": 256,
    }
    response = httpx.post(
        f"{base_url}/chat/completions",
        headers={"Authorization": f"Bearer {os.getenv('LLM_API_KEY', 'EMPTY')}"},
        json=payload,
        timeout=120,
    )
    response.raise_for_status()
    print(response.text)


if __name__ == "__main__":
    main()
