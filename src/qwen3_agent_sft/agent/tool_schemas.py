from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictInput(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ToolOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: str | None = None
    error: str | None = None


class QueryTrafficMetricsInput(StrictInput):
    region_id: str | None = Field(default=None, pattern=r"^R[1-4]$")
    road_segment_id: str | None = Field(default=None, pattern=r"^S\d{3}$")
    start_time: str | None = None
    end_time: str | None = None


class QueryTrafficMetricsOutput(ToolOutput):
    record_count: int
    avg_speed_mean: float | None = None
    flow_mean: float | None = None
    occupancy_mean: float | None = None
    congestion_index_mean: float | None = None
    travel_time_index_mean: float | None = None
    sample_rows: list[dict[str, Any]] = Field(default_factory=list)


class GetTopCongestedSegmentsInput(StrictInput):
    k: int = Field(default=5, ge=1, le=20)
    region_id: str | None = Field(default=None, pattern=r"^R[1-4]$")


class GetTopCongestedSegmentsOutput(ToolOutput):
    items: list[dict[str, Any]] = Field(default_factory=list)


class QueryRoadSegmentInfoInput(StrictInput):
    road_segment_id: str = Field(pattern=r"^S\d{3}$")


class QueryRoadSegmentInfoOutput(ToolOutput):
    segment: dict[str, Any] | None = None


class QueryTrafficEventsInput(StrictInput):
    region_id: str | None = Field(default=None, pattern=r"^R[1-4]$")
    road_segment_id: str | None = Field(default=None, pattern=r"^S\d{3}$")
    status: Literal["open", "processing", "resolved"] | None = None


class QueryTrafficEventsOutput(ToolOutput):
    items: list[dict[str, Any]] = Field(default_factory=list)


class QueryTrafficForecastInput(StrictInput):
    road_segment_id: str | None = Field(default=None, pattern=r"^S\d{3}$")
    horizon_minutes: Literal[15, 30, 60] | None = None
    model_name: Literal["LastValue", "GraphWaveNetFull", "STGCNFull"] | None = None


class QueryTrafficForecastOutput(ToolOutput):
    items: list[dict[str, Any]] = Field(default_factory=list)


class QuerySignalPlanInput(StrictInput):
    road_segment_id: str | None = Field(default=None, pattern=r"^S\d{3}$")
    intersection_id: str | None = Field(default=None, pattern=r"^I\d{3}$")


class QuerySignalPlanOutput(ToolOutput):
    items: list[dict[str, Any]] = Field(default_factory=list)


class CreateIncidentActionRecordInput(StrictInput):
    event_id: str = Field(pattern=r"^E\d{3}$")
    action_type: str
    description: str
    operator: str = "demo_user"


class CreateIncidentActionRecordOutput(ToolOutput):
    record: dict[str, Any]
    path: str


class RetrieveKnowledgeInput(StrictInput):
    query: str
    top_k: int = Field(default=5, ge=1, le=20)


class RetrieveKnowledgeOutput(ToolOutput):
    chunks: list[dict[str, Any]] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)
    scores: list[float] = Field(default_factory=list)
    no_hit: bool = False
