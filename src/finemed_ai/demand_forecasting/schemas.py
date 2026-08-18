from __future__ import annotations

from datetime import date, datetime
from typing import List

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Quantile forecast
# ---------------------------------------------------------------------------

class QuantileForecast(BaseModel):
    """
    Probabilistic forecast quantiles for one forecast day.

    The values are expected to satisfy:

        P10 <= P20 <= ... <= P90
    """

    p10: float
    p20: float
    p30: float
    p40: float
    p50: float
    p60: float
    p70: float
    p80: float
    p90: float


# ---------------------------------------------------------------------------
# Daily forecast result
# ---------------------------------------------------------------------------

class ForecastDayResult(BaseModel):
    """
    Forecast for one medicine on one future calendar day.
    """

    forecast_date: date
    predicted_demand: float
    quantiles: QuantileForecast


# ---------------------------------------------------------------------------
# Medicine-level forecast result
# ---------------------------------------------------------------------------

class MedicineForecastResult(BaseModel):
    """
    Complete forecast for one medicine over the prediction horizon.
    """

    medicine_id: str
    generated_at: datetime
    context_length_used: int
    prediction_length: int
    model_id: str
    days: List[ForecastDayResult] = Field(default_factory=list)

    def to_summary(self):

        if not self.days:
            raise ValueError("Cannot summarize an empty forecast.")

        values = [float(day.predicted_demand) for day in self.days]

        total_predicted_demand = sum(values)

        avg_daily_demand = (total_predicted_demand / len(values))

        if len(values) == 1:
            first_half = values
            second_half = values

        else:
            split_index = len(values) // 2

            first_half = values[:split_index]
            second_half = values[split_index:]

        # Defensive guard.
        if not second_half:
            second_half = first_half

        first_avg = (sum(first_half) / len(first_half))
        second_avg = (sum(second_half) / len(second_half))

        if first_avg > 0:
            trend_pct_change = ((second_avg - first_avg) / first_avg* 100.0)

        elif second_avg > 0:
            trend_pct_change = 100.0

        else:
            trend_pct_change = 0.0

        TREND_THRESHOLD_PCT = 5.0

        if trend_pct_change > TREND_THRESHOLD_PCT:
            trend = "increasing"

        elif trend_pct_change < -TREND_THRESHOLD_PCT:
            trend = "decreasing"

        else:
            trend = "stable"

        return ForecastSummary(
            medicine_id=self.medicine_id,
            generated_at=self.generated_at,
            model_id=self.model_id,
            prediction_length=self.prediction_length,
            total_predicted_demand=float(total_predicted_demand),
            avg_daily_demand=float(avg_daily_demand),
            first_half_avg=float(first_avg),
            second_half_avg=float(second_avg),
            trend=trend,
            trend_pct_change=float(trend_pct_change))
# ---------------------------------------------------------------------------
# Medicine-level summary
# ---------------------------------------------------------------------------

class ForecastSummary(BaseModel):
    """
    Medicine-level forecast summary.

    This schema is consumed by:
    - FastAPI
    - LLM tools
    - ranking/comparison tools
    - future dashboard/API consumers
    """

    medicine_id: str
    generated_at: datetime
    model_id: str
    prediction_length: int

    total_predicted_demand: float
    avg_daily_demand: float

    first_half_avg: float
    second_half_avg: float

    trend: str
    trend_pct_change: float


# ---------------------------------------------------------------------------
# Batch forecast run result
# ---------------------------------------------------------------------------

class BatchForecastRunResult(BaseModel):
    """
    Metadata for one production forecast batch.
    """

    run_id: str

    started_at: datetime
    completed_at: datetime

    medicines_requested: int
    medicines_succeeded: int
    medicines_failed: int

    failed_medicine_ids: List[str] = Field(
        default_factory=list
    )

    output_path: str

    published: bool
    publish_note: str = ""