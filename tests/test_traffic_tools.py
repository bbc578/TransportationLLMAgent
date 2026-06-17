from pathlib import Path

from qwen3_agent_sft.tools.traffic_data_tools import (
    create_incident_action_record,
    get_top_congested_segments,
    query_road_segment_info,
    query_signal_plan,
    query_traffic_events,
    query_traffic_forecast,
    query_traffic_metrics,
)


def test_query_traffic_metrics():
    result = query_traffic_metrics(region_id="R1")
    assert result["record_count"] > 0
    assert result["avg_speed_mean"] is not None


def test_get_top_congested_segments():
    result = get_top_congested_segments(k=3)
    assert len(result["items"]) == 3
    assert result["items"][0]["congestion_index"] >= result["items"][-1]["congestion_index"]


def test_query_road_segment_info():
    result = query_road_segment_info("S001")
    assert result["segment"]["road_segment_id"] == "S001"


def test_query_events_forecasts_and_signal_plan():
    assert query_traffic_events(road_segment_id="S001")["items"] is not None
    assert query_traffic_forecast(road_segment_id="S001", horizon_minutes=30)["items"]
    assert query_signal_plan(road_segment_id="S001")["items"]


def test_create_incident_action_record():
    result = create_incident_action_record("E001", "demo_response", "发布样例诱导提示")
    assert result["record"]["sample_action_record"] is True
    assert Path(result["path"]).exists()
