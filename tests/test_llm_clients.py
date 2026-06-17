import json

from qwen3_agent_sft.llm.mock_client import MockLLMClient
from qwen3_agent_sft.llm.schemas import LLMMessage


def test_mock_client_returns_plan_json():
    response = MockLLMClient().chat([LLMMessage(role="user", content="今天哪些路段最拥堵？")])
    assert json.loads(response.content)["type"] == "tool_call"
