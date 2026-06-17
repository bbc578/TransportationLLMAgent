from __future__ import annotations

import csv
from datetime import datetime, timedelta
from pathlib import Path

OUT = Path("data/sample_traffic")


def write_csv(name: str, rows: list[dict[str, object]]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / name).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    regions = ["R1", "R2", "R3", "R4"]
    segments = [f"S{i:03d}" for i in range(1, 31)]
    base = datetime(2026, 6, 16, 7, 0)

    road_rows = []
    road_types = ["arterial", "expressway", "collector", "ramp"]
    for i, segment in enumerate(segments, start=1):
        road_rows.append(
            {
                "road_segment_id": segment,
                "name": f"Sample Road {i}",
                "region_id": regions[(i - 1) % len(regions)],
                "road_type": road_types[(i - 1) % len(road_types)],
                "upstream_segments": "" if i == 1 else f"S{i - 1:03d}",
                "downstream_segments": "" if i == 30 else f"S{i + 1:03d}",
                "signalized_intersection": "true" if i % 2 == 0 else "false",
                "lanes": 2 + (i % 4),
            }
        )
    write_csv("road_segments.csv", road_rows)

    metric_rows = []
    for step in range(12):
        ts = base + timedelta(minutes=15 * step)
        peak_factor = 1.4 if ts.hour in {8, 9, 17, 18} else 1.0
        for i, segment in enumerate(segments, start=1):
            incident_factor = 1.7 if i in {1, 7, 13, 19, 25} and step in {4, 5, 6} else 1.0
            flow = int((620 + (i % 9) * 55 + step * 18) * peak_factor)
            occupancy = min(0.96, 0.28 + (i % 10) * 0.045 + (peak_factor - 1) * 0.12)
            congestion = min(1.0, round(0.22 + occupancy * 0.62 + (incident_factor - 1) * 0.22, 3))
            speed = max(8.0, round(62 - congestion * 48 - (i % 4) * 2.2, 1))
            metric_rows.append(
                {
                    "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
                    "region_id": regions[(i - 1) % len(regions)],
                    "road_segment_id": segment,
                    "avg_speed": speed,
                    "flow": flow,
                    "occupancy": round(occupancy, 3),
                    "congestion_index": congestion,
                    "travel_time_index": round(1 + congestion * 1.9, 3),
                }
            )
    write_csv("traffic_metrics.csv", metric_rows)

    event_types = ["accident", "roadwork", "weather", "illegal_parking", "signal_fault", "high_demand"]
    severities = ["low", "medium", "high"]
    statuses = ["open", "processing", "resolved"]
    event_rows = []
    for i in range(1, 41):
        segment = segments[(i * 3 - 1) % len(segments)]
        event_type = event_types[(i - 1) % len(event_types)]
        event_rows.append(
            {
                "event_id": f"E{i:03d}",
                "timestamp": (base + timedelta(minutes=10 * i)).strftime("%Y-%m-%d %H:%M:%S"),
                "region_id": regions[(i - 1) % len(regions)],
                "road_segment_id": segment,
                "event_type": event_type,
                "severity": severities[(i - 1) % len(severities)],
                "description": f"Sample {event_type} event on {segment}",
                "status": statuses[(i - 1) % len(statuses)],
            }
        )
    write_csv("traffic_events.csv", event_rows)

    signal_rows = []
    for i, segment in enumerate(segments[:15], start=1):
        signal_rows.append(
            {
                "intersection_id": f"I{i:03d}",
                "road_segment_id": segment,
                "cycle_length": 90 + (i % 4) * 10,
                "green_ratio_main": round(0.45 + (i % 5) * 0.04, 2),
                "green_ratio_side": round(0.25 + (i % 4) * 0.03, 2),
                "coordination_group": f"G{1 + (i % 3)}",
                "last_updated": "2026-06-01 00:00:00",
            }
        )
    write_csv("signal_plans.csv", signal_rows)

    forecast_rows = []
    model_names = ["LastValue", "GraphWaveNetFull", "STGCNFull"]
    for step in range(10):
        ts = base + timedelta(minutes=15 * step)
        for i, segment in enumerate(segments, start=1):
            for horizon in [15, 30, 60]:
                model_name = model_names[(i + horizon + step) % len(model_names)]
                congestion = min(1.0, round(0.24 + (i % 10) * 0.055 + horizon / 300, 3))
                forecast_rows.append(
                    {
                        "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
                        "road_segment_id": segment,
                        "horizon_minutes": horizon,
                        "predicted_speed": max(8.0, round(58 - congestion * 42, 1)),
                        "predicted_congestion_index": congestion,
                        "model_name": model_name,
                    }
                )
    write_csv("traffic_forecasts.csv", forecast_rows)


if __name__ == "__main__":
    main()
