from datetime import date, datetime
from typing import Dict, List, Optional
 
from pydantic import BaseModel, Field
 
 
class QuantileForecast(BaseModel):
    """Uncertainty band for a single forecasted day."""
    p10: float
    p20: float
    p30: float
    p40: float
    p50: float
    p60: float
    p70: float
    p80: float
    p90: float
 
 
class ForecastDayResult(BaseModel):
    forecast_date: date
    predicted_demand: float = Field(..., description="P50 point forecast, clipped at 0")
    quantiles: QuantileForecast
 
 
class MedicineForecastResult(BaseModel):
    """Full 30-day forecast for one medicine — the atomic unit consumed by
    the LLM layer's forecast/trend/summary tools."""
    medicine_id: str
    generated_at: datetime
    context_length_used: int
    prediction_length: int
    model_id: str
    days: List[ForecastDayResult]
 
    def to_summary(self) -> "ForecastSummary":
        total = sum(d.predicted_demand for d in self.days)
        avg = total / len(self.days) if self.days else 0.0
        peak_day = max(self.days, key=lambda d: d.predicted_demand) if self.days else None
        trough_day = min(self.days, key=lambda d: d.predicted_demand) if self.days else None
 
        first_half = self.days[: len(self.days) // 2]
        second_half = self.days[len(self.days) // 2 :]
        fh_avg = (sum(d.predicted_demand for d in first_half) / len(first_half)) if first_half else 0.0
        sh_avg = (sum(d.predicted_demand for d in second_half) / len(second_half)) if second_half else 0.0
 
        if fh_avg == 0:
            trend = "flat"
        else:
            pct_change = (sh_avg - fh_avg) / fh_avg
            if pct_change > 0.10:
                trend = "increasing"
            elif pct_change < -0.10:
                trend = "decreasing"
            else:
                trend = "stable"
 
        return ForecastSummary(
            medicine_id=self.medicine_id,
            total_predicted_demand=round(total, 2),
            avg_daily_demand=round(avg, 2),
            peak_day=peak_day.forecast_date if peak_day else None,
            peak_demand=peak_day.predicted_demand if peak_day else None,
            trough_day=trough_day.forecast_date if trough_day else None,
            trough_demand=trough_day.predicted_demand if trough_day else None,
            trend=trend,
            trend_pct_change=round(pct_change * 100, 1) if fh_avg else 0.0,
        )
 
 
class ForecastSummary(BaseModel):
    """Employee-friendly summary — what the LLM's Summary Tool hands back."""
    medicine_id: str
    total_predicted_demand: float
    avg_daily_demand: float
    peak_day: Optional[date]
    peak_demand: Optional[float]
    trough_day: Optional[date]
    trough_demand: Optional[float]
    trend: str
    trend_pct_change: float
 
 
class BatchForecastRunResult(BaseModel):
    """Result of a full monthly forecast run across all medicines."""
    run_id: str
    started_at: datetime
    completed_at: datetime
    medicines_requested: int
    medicines_succeeded: int
    medicines_failed: int
    failed_medicine_ids: List[str]
    output_path: str
    published: bool  # False if held back due to failing the quality gate
    publish_note: str = ""  # explains why, if published=False
 