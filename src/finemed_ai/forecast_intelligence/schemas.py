from __future__ import annotations

from datetime import date, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, ConfigDict


class ForecastPoint(BaseModel):
    """A single forecast observation."""

    model_config = ConfigDict(extra="forbid")

    forecast_date: date
    predicted_demand: float = Field(ge=0)
    lower_bound: float | None = Field(default=None, ge=0)
    upper_bound: float | None = Field(default=None, ge=0)


class ForecastSummary(BaseModel):
    """Summary of a medicine forecast."""

    model_config = ConfigDict(extra="forbid")

    medicine_code: str

    forecast_start: date
    forecast_end: date

    horizon_days: int = Field(gt=0)

    total_predicted_demand: float = Field(ge=0)
    average_daily_demand: float = Field(ge=0)

    minimum_daily_demand: float = Field(ge=0)
    maximum_daily_demand: float = Field(ge=0)

    selected_model: str | None = None
    forecast_status: Literal[
        "forecastable",
        "dormant",
        "unavailable",
    ]

    points: list[ForecastPoint] = Field(default_factory=list)


class MedicineResolution(BaseModel):
    """Result of resolving a user-provided medicine reference."""

    model_config = ConfigDict(extra="forbid")

    query: str
    resolved: bool

    medicine_code: str | None = None
    medicine_name: str | None = None

    match_type: Literal[
        "exact_code",
        "exact_name",
        "partial_name",
        "fuzzy_match",
        "ambiguous",
        "not_found",
    ]


class ForecastComparison(BaseModel):
    """Comparison between two medicine forecasts."""

    model_config = ConfigDict(extra="forbid")

    medicine_a: ForecastSummary
    medicine_b: ForecastSummary

    demand_difference: float
    demand_ratio: float | None = None

    higher_demand_medicine: str | None = None


class MedicineRankingItem(BaseModel):
    """A ranked medicine forecast result."""

    model_config = ConfigDict(extra="forbid")

    rank: int = Field(gt=0)

    medicine_code: str
    medicine_name: str | None = None

    total_predicted_demand: float = Field(ge=0)

    selected_model: str | None = None


class TrendAnalysis(BaseModel):
    """Trend analysis for a medicine forecast."""

    model_config = ConfigDict(extra="forbid")

    medicine_code: str

    trend: Literal[
        "increasing",
        "decreasing",
        "stable",
        "volatile",
        "insufficient_data",
    ]

    first_period_demand: float | None = Field(default=None, ge=0)
    second_period_demand: float | None = Field(default=None, ge=0)

    absolute_change: float | None = None
    percentage_change: float | None = None

    explanation: str


class RoutingExplanation(BaseModel):
    """Explanation of why a forecasting model was selected."""

    model_config = ConfigDict(extra="forbid")

    medicine_code: str

    selected_model: str

    chronos_advantage: float | None = None

    routing_reason: str

    validation_cutoffs: list[str] = Field(default_factory=list)


class DormantMedicine(BaseModel):
    """A medicine currently classified as dormant."""

    model_config = ConfigDict(extra="forbid")

    medicine_code: str
    medicine_name: str | None = None

    status: Literal["dormant"]

    last_observed_date: date | None = None


class IntelligenceResponse(BaseModel):
    """
    Generic deterministic response returned to the LLM/API layer.

    The LLM must explain this data but must not invent values
    outside this validated result.
    """

    model_config = ConfigDict(extra="forbid")

    success: bool

    message: str

    data: dict[str, Any] | None = None

    generated_at: datetime = Field(
        default_factory=datetime.utcnow,
    )