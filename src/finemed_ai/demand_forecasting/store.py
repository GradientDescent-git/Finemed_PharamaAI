from __future__ import annotations
 
import logging
from pathlib import Path
from typing import List, Optional
 
import pandas as pd
 
from finemed_ai.demand_forecasting.schemas import (
    ForecastDayResult,
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
 