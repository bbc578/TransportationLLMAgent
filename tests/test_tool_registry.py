from qwen3_agent_sft.agent.tool_registry import ToolRegistry


def test_registry_lists_and_validates_tools():
    registry = ToolRegistry()
    assert "query_traffic_metrics" in registry.list_tools()
    ok, error = registry.validate_tool_call("query_traffic_metrics", {"region_id": "R1"})
    assert ok, error
    assert registry.execute_tool("missing", {})["error"] == "unknown_tool: missing"


def test_registry_rejects_invalid_input():
    registry = ToolRegistry()
    ok, error = registry.validate_tool_call("query_road_segment_info", {})
    assert not ok
    assert error and "input_validation_error" in error

    result = registry.execute_tool("query_traffic_metrics", {"unknown": "value"})
    assert result["error"] == "input_validation_error"


def test_registry_exposes_pydantic_input_schema_and_metadata():
    registry = ToolRegistry()
    tools = registry.get_tool_schema_for_llm()
    metrics_tool = next(t for t in tools if t["function"]["name"] == "query_traffic_metrics")
    parameters = metrics_tool["function"]["parameters"]
    assert "region_id" in parameters["properties"]
    assert metrics_tool["metadata"]["read_only"] is True


def test_registry_validates_output_model():
    registry = ToolRegistry()
    result = registry.execute_tool("get_top_congested_segments", {"k": 3})
    assert len(result["items"]) == 3
    assert result["error"] is None
