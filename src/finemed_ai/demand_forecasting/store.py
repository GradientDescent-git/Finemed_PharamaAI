from __future__ import annotations
 
import logging
from pathlib import Path
from typing import List, Optional
 
import pandas as pd
 
from finemed_ai.demand_forecasting.schemas import (
    ForecastDayResult,
    ForecastSummary,
    MedicineForecastResult,
    QuantileForecast,
)
 
logger = logging.getLogger(__name__)
 
 
class ForecastNotFoundError(KeyError):
    pass
 
 
class ForecastStore:
    """Loads the latest batch forecast output and serves it in-memory.
 
    Usage:
        store = ForecastStore(Path("data/05_forecasts/latest.parquet"))
        result = store.get(medicine_id="42")
    """
 
    def __init__(self, latest_path: Path):
        self.latest_path = latest_path
        self._df: Optional[pd.DataFrame] = None
        self._loaded_at: Optional[float] = None
        self.reload()
 
    def reload(self) -> None:
        if not self.latest_path.exists():
            logger.warning(
                "No forecast file at %s yet — store is empty until the first "
                "monthly run completes.", self.latest_path,
            )
            self._df = pd.DataFrame()
            return
 
        df = pd.read_parquet(self.latest_path)
        df["Medicine_ID"] = df["Medicine_ID"].astype(str)
        self._df = df
        self._loaded_at = self.latest_path.stat().st_mtime
        logger.info(
            "ForecastStore loaded %d rows across %d medicines from %s",
            len(df), df["Medicine_ID"].nunique(), self.latest_path,
        )
 
    def is_stale(self) -> bool:
        """True if the file on disk has changed since we loaded it (e.g. a
        new monthly run finished) — call reload() if so."""
        if not self.latest_path.exists():
            return False
        return self.latest_path.stat().st_mtime != self._loaded_at
 
    def list_medicine_ids(self) -> List[str]:
        if self._df is None or self._df.empty:
            return []
        return sorted(self._df["Medicine_ID"].unique())
 
    def get(self, medicine_id: str) -> MedicineForecastResult:
        medicine_id = str(medicine_id)
        if self._df is None or self._df.empty:
            raise ForecastNotFoundError(
                f"No forecasts loaded yet (medicine_id={medicine_id})"
            )
 
        rows = self._df[self._df["Medicine_ID"] == medicine_id].sort_values("Forecast_Date")
        if rows.empty:
            raise ForecastNotFoundError(f"No forecast for medicine_id={medicine_id}")
 
        days = [
            ForecastDayResult(
                forecast_date=row["Forecast_Date"],
                predicted_demand=float(row["Predicted_Demand"]),
                quantiles=QuantileForecast(
                    p10=float(row["P10"]), p20=float(row["P20"]), p30=float(row["P30"]),
                    p40=float(row["P40"]), p50=float(row["P50"]), p60=float(row["P60"]),
                    p70=float(row["P70"]), p80=float(row["P80"]), p90=float(row["P90"]),
                ),
            )
            for _, row in rows.iterrows()
        ]
        first = rows.iloc[0]
        return MedicineForecastResult(
            medicine_id=medicine_id,
            generated_at=first["Generated_At"],
            context_length_used=int(first["Context_Length_Used"]),
            prediction_length=len(days),
            model_id=first["Model_ID"],
            days=days,
        )

    def get_all_summaries(self) -> List[ForecastSummary]:
        """
        One ForecastSummary per medicine currently forecasted. This is the
        backbone for ranking/comparison tools (top-demand, trend-based
        filtering, uncertainty ranking) -- computing all of them once here
        is much cheaper than each tool re-deriving summaries independently.
        """
        summaries = []
        for medicine_id in self.list_medicine_ids():
            try:
                summaries.append(self.get(medicine_id).to_summary())
            except ForecastNotFoundError:
                continue
        return summaries

    def get_top_demand(self, n: int = 10) -> List[ForecastSummary]:
        """Top N medicines by total predicted demand over the forecast horizon."""
        summaries = self.get_all_summaries()
        return sorted(summaries, key=lambda s: s.total_predicted_demand, reverse=True)[:n]

    def get_by_trend(self, trend: str, n: int = 10) -> List[ForecastSummary]:
        """Medicines matching a trend direction ('increasing', 'decreasing',
        'stable', 'flat'), sorted by magnitude of change."""
        summaries = [s for s in self.get_all_summaries() if s.trend == trend]
        return sorted(summaries, key=lambda s: abs(s.trend_pct_change), reverse=True)[:n]

    def get_most_uncertain(self, n: int = 10) -> List[dict]:
        """
        Medicines with the widest P10-P90 spread relative to their P50 --
        i.e. where the forecast itself is least confident. Useful for
        flagging medicines that need closer manual attention rather than
        blind trust in the point forecast.
        """
        if self._df is None or self._df.empty:
            return []

        results = []
        for medicine_id, group in self._df.groupby("Medicine_ID"):
            avg_p50 = group["P50"].mean()
            avg_spread = (group["P90"] - group["P10"]).mean()
            relative_uncertainty = (avg_spread / avg_p50 * 100) if avg_p50 > 0 else 0.0
            results.append({
                "medicine_id": medicine_id,
                "avg_p50": round(float(avg_p50), 2),
                "avg_p10_p90_spread": round(float(avg_spread), 2),
                "relative_uncertainty_pct": round(float(relative_uncertainty), 1),
            })
        return sorted(results, key=lambda r: r["relative_uncertainty_pct"], reverse=True)[:n]

    def compare(self, medicine_ids: List[str]) -> List[ForecastSummary]:
        """Summaries for a specific set of medicines, for direct comparison."""
        results = []
        for mid in medicine_ids:
            try:
                results.append(self.get(str(mid)).to_summary())
            except ForecastNotFoundError:
                continue
        return results
