# Tool Calling Design

Traffic tools are registered through `ToolRegistry`. Each tool has independent Pydantic input/output schemas. The model selects tools, but the backend validates inputs, executes functions and validates outputs.

Registered tools:

- `query_traffic_metrics`
- `get_top_congested_segments`
- `query_road_segment_info`
- `query_traffic_events`
- `query_traffic_forecast`
- `query_signal_plan`
- `create_incident_action_record`
- `retrieve_knowledge`

Tools operate on sample traffic data only. Signal plans are read-only.
