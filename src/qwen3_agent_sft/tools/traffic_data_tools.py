from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

TRAFFIC_DATA_DIR = Path("data/sample_traffic")
ACTION_RECORD_DIR = Path("outputs/action_records")


def _read_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(TRAFFIC_DATA_DIR / name)


def _filter_time(df: pd.DataFrame, start_time: str | None, end_time: str | None) -> pd.DataFrame:
    if start_time:
        df = df[df["timestamp"] >= start_time]
    if end_time:
        df = df[df["timestamp"] <= end_time]
    return df


def query_traffic_metrics(
    region_id: str | None = None,
    road_segment_id: str | None = None,
    start_time: str | None = None,
    end_time: str | None = None,
) -> dict[str, Any]:
    df = _read_csv("traffic_metrics.csv")
    if region_id:
        df = df[df["region_id"] == region_id]
    if road_segment_id:
        df = df[df["road_segment_id"] == road_segment_id]
    df = _filter_time(df, start_time, end_time)
    if df.empty:
        return {
            "message": "No matching sample traffic metrics found.",
            "record_count": 0,
            "avg_speed_mean": None,
            "flow_mean": None,
            "occupancy_mean": None,
            "congestion_index_mean": None,
            "travel_time_index_mean": None,
            "sample_rows": [],
        }
    return {
        "message": "Sample traffic metrics summary.",
        "record_count": int(len(df)),
        "avg_speed_mean": round(float(df["avg_speed"].mean()), 3),
        "flow_mean": round(float(df["flow"].mean()), 3),
        "occupancy_mean": round(float(df["occupancy"].mean()), 3),
        "congestion_index_mean": round(float(df["congestion_index"].mean()), 3),
        "travel_time_index_mean": round(float(df["travel_time_index"].mean()), 3),
        "sample_rows": df.head(10).to_dict(orient="records"),
    }


def get_top_congested_segments(k: int = 5, region_id: str | None = None) -> dict[str, Any]:
    df = _read_csv("traffic_metrics.csv")
    if region_id:
        df = df[df["region_id"] == region_id]
    if df.empty:
        return {"message": "No matching sample congestion records found.", "items": []}
    top = df.sort_values("congestion_index", ascending=False).head(k)
    return {
        "message": "Top congested sample road segments.",
        "items": top[
            [
                "timestamp",
                "region_id",
                "road_segment_id",
                "avg_speed",
                "flow",
                "occupancy",
                "congestion_index",
                "travel_time_index",
            ]
        ].to_dict(orient="records"),
    }


def query_road_segment_info(road_segment_id: str) -> dict[str, Any]:
    df = _read_csv("road_segments.csv")
    hit = df[df["road_segment_id"] == road_segment_id]
    if hit.empty:
        return {"message": f"No sample road segment found for {road_segment_id}.", "segment": None}
    return {"message": "Sample road segment info.", "segment": hit.iloc[0].to_dict()}


def query_traffic_events(
    region_id: str | None = None,
    road_segment_id: str | None = None,
    status: str | None = None,
) -> dict[str, Any]:
    df = _read_csv("traffic_events.csv")
    if region_id:
        df = df[df["region_id"] == region_id]
    if road_segment_id:
        df = df[df["road_segment_id"] == road_segment_id]
    if status:
        df = df[df["status"] == status]
    if df.empty:
        return {"message": "No matching sample traffic events found.", "items": []}
    return {"message": "Sample traffic events.", "items": df.head(20).to_dict(orient="records")}


def query_traffic_forecast(
    road_segment_id: str | None = None,
    horizon_minutes: int | None = None,
    model_name: str | None = None,
) -> dict[str, Any]:
    df = _read_csv("traffic_forecasts.csv")
    if road_segment_id:
        df = df[df["road_segment_id"] == road_segment_id]
    if horizon_minutes:
        df = df[df["horizon_minutes"] == horizon_minutes]
    if model_name:
        df = df[df["model_name"] == model_name]
    if df.empty:
        return {"message": "No matching sample traffic forecasts found.", "items": []}
    return {"message": "Sample traffic forecasts.", "items": df.head(20).to_dict(orient="records")}


def query_signal_plan(
    road_segment_id: str | None = None,
    intersection_id: str | None = None,
) -> dict[str, Any]:
    df = _read_csv("signal_plans.csv")
    if road_segment_id:
        df = df[df["road_segment_id"] == road_segment_id]
    if intersection_id:
        df = df[df["intersection_id"] == intersection_id]
    if df.empty:
        return {"message": "No matching sample signal plan found.", "items": []}
    return {"message": "Read-only sample signal plans.", "items": df.to_dict(orient="records")}


def create_incident_action_record(
    event_id: str,
    action_type: str,
    description: str,
    operator: str = "demo_user",
) -> dict[str, Any]:
    ACTION_RECORD_DIR.mkdir(parents=True, exist_ok=True)
    record = {
        "record_id": f"AR{datetime.now().strftime('%Y%m%d%H%M%S%f')}",
        "event_id": event_id,
        "action_type": action_type,
        "description": description,
        "operator": operator,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "sample_action_record": True,
    }
    path = ACTION_RECORD_DIR / f"{record['record_id']}.json"
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "message": "Created a sample traffic incident action record.",
        "record": record,
        "path": str(path),
    }
