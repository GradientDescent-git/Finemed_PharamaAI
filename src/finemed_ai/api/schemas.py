from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    forecast_store_loaded: bool
    medicines_available: int
    conversation_service_available: bool
    chat_available: bool


class ReadyResponse(BaseModel):
    status: str
    ready: bool
    store_ready: bool
    orchestrator_ready: bool
    detail: str


class VersionResponse(BaseModel):
    name: str = "Finemed PharmaAI"
    version: str = "1.0.0"
    environment: str = "production"
    forecasting_models: list[str] = ["TSB", "Chronos-2 P50"]


class ForecastDayItem(BaseModel):
    forecast_date: str
    predicted_demand: float
    quantiles: Optional[dict[str, float]] = None


class ForecastResponse(BaseModel):
    medicine_id: str
    medicine_name: Optional[str] = None
    found: bool = True
    forecast_start: Optional[str] = None
    forecast_end: Optional[str] = None
    forecast_days: int = 30
    total_predicted_demand: float = 0.0
    average_daily_demand: float = 0.0
    selected_model: str
    eligibility_status: str
    forecast_status: str
    generated_at: Optional[str] = None
    run_id: Optional[str] = None
    days: list[ForecastDayItem] = Field(default_factory=list)


class ForecastSummaryResponse(BaseModel):
    medicine_id: str
    medicine_name: Optional[str] = None
    selected_model: str
    eligibility_status: str
    forecast_status: str
    total_predicted_demand: float
    avg_daily_demand: float
    forecast_start: Optional[str] = None
    forecast_end: Optional[str] = None


class RankingItem(BaseModel):
    rank: int
    medicine_id: str
    medicine_name: Optional[str] = None
    total_predicted_demand: float
    selected_model: Optional[str] = None


class RankingResponse(BaseModel):
    direction: str
    limit: int
    items: list[RankingItem] = Field(default_factory=list)


class PipelineStageStatus(BaseModel):
    stage: str
    status: str  # QUEUED, RUNNING, SUCCEEDED, FAILED
    detail: Optional[str] = None
    updated_at: Optional[str] = None


class PipelineValidationDetail(BaseModel):
    valid: bool
    month: Optional[str] = None
    files_present: list[str] = Field(default_factory=list)
    missing_files: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class PipelineRunResponse(BaseModel):
    run_id: Optional[str] = None
    status: str = "idle"  # idle, running, succeeded, failed
    source_period: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    current_stage: Optional[str] = None
    stages: list[PipelineStageStatus] = Field(default_factory=list)
    validation: Optional[PipelineValidationDetail] = None
    error: Optional[str] = None


class FreshnessResponse(BaseModel):
    generated_at: Optional[str] = None
    source_period: Optional[str] = None
    forecast_start: Optional[str] = None
    forecast_end: Optional[str] = None
    run_id: Optional[str] = None
    freshness_status: str  # HEALTHY, STALE, FAILED, MISSING
    is_stale: bool = False


class OperationsSummaryResponse(BaseModel):
    api_status: str
    forecast_store_loaded: bool
    total_medicines: int
    chat_service_status: str
    latest_run_id: Optional[str] = None
    freshness: FreshnessResponse
    pipeline_running: bool
    active_alerts_count: int = 0
