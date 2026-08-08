from __future__ import annotations
 
import logging
from datetime import datetime, timezone
from typing import List, Optional
 
import pandas as pd
 
from finemed_ai.demand_forecasting.config import DEFAULT_CONFIG, ForecastConfig
from finemed_ai.demand_forecasting.schemas import (
    ForecastDayResult,
    MedicineForecastResult,
    QuantileForecast,
)
 
logger = logging.getLogger(__name__)
 
 
class InsufficientHistoryError(ValueError):
    """Raised when a medicine has no usable history to forecast from."""
 
 
class PredictorService:
    """Singleton-style Chronos-2 predictor. Use `get_instance()` in the API
    and pipeline layers rather than constructing directly, so the model is
    loaded exactly once per process."""
 
    _instance: Optional["PredictorService"] = None
 
    def __init__(self, config: ForecastConfig = DEFAULT_CONFIG, device: Optional[str] = None):
        import torch  # deferred: keeps torch/chronos off the import path for
                       # unit tests and any code that only needs schemas/config
 
        self.config = config
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
 
        logger.info("Loading Chronos-2 (%s) on device=%s", config.model_id, self.device)
        t0 = datetime.now()
 
        from chronos import Chronos2Pipeline  # deferred import: keeps torch/chronos
 
        self.pipeline = Chronos2Pipeline.from_pretrained(
            config.model_id,
            device_map=self.device,
            torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
        )
 
        logger.info(
            "Chronos-2 loaded in %.1fs (device=%s)",
            (datetime.now() - t0).total_seconds(),
            self.device,
        )
 
    @classmethod
    def get_instance(cls, config: ForecastConfig = DEFAULT_CONFIG) -> "PredictorService":
        """Process-wide singleton accessor. FastAPI should call this once in
        its startup lifespan and store the reference; do not call this
        per-request (it's cheap after the first call, but be explicit)."""
        if cls._instance is None:
            cls._instance = cls(config=config)
        return cls._instance
 
    @classmethod
    def reset_instance(cls) -> None:
        """For tests only — forces a fresh load on next get_instance()."""
        cls._instance = None
 
    # ------------------------------------------------------------------
    # Core forecasting
    # ------------------------------------------------------------------
 
    def forecast_medicine(
        self, item_id: str, history_df: pd.DataFrame
    ) -> MedicineForecastResult:
        """
        Forecast the next `prediction_length` days for ONE medicine.
 
        Parameters
        ----------
        item_id : str
            MDCODE / Medicine_ID as a string.
        history_df : pd.DataFrame
            Long-format daily history with columns
            [item_id, timestamp, target] for ONE OR MANY medicines
            (this function filters to `item_id` internally). timestamp
            must be datetime64, target must be numeric, and the series
            must be a complete daily calendar (no gaps) — matches the
            `daily_demand_df` output of your Module 2 pipeline.
 
        Raises
        ------
        InsufficientHistoryError
            If the medicine has no rows in history_df.
        """
        cfg = self.config
        item_id = str(item_id)
 
        history = history_df[history_df[cfg.id_column].astype(str) == item_id].copy()
        history[cfg.id_column] = history[cfg.id_column].astype(str)
        history = history.sort_values(cfg.timestamp_column)
 
        if history.empty:
            raise InsufficientHistoryError(f"No history for item_id={item_id}")
 
        history = history.tail(cfg.context_length)
        actual_context = len(history)
 
        if actual_context < cfg.prediction_length:
            # Chronos-2 can still run on short series, but a forecast built
            # from less history than the forecast horizon itself is not
            # something the LLM layer should present with confidence — flag
            # it loudly rather than silently returning a low-quality number.
            logger.warning(
                "item_id=%s has only %d days of history (< prediction_length=%d). "
                "Forecast quality is not validated at this context length.",
                item_id, actual_context, cfg.prediction_length,
            )
 
        raw = self.pipeline.predict_df(
            history,
            prediction_length=cfg.prediction_length,
            quantile_levels=list(cfg.quantile_levels),
            id_column=cfg.id_column,
            timestamp_column=cfg.timestamp_column,
            target=cfg.target_column,
        )
 
        return self._to_result(item_id, actual_context, raw)
 
    def forecast_batch(
        self,
        history_df: pd.DataFrame,
        item_ids: Optional[List[str]] = None,
    ) -> tuple[List[MedicineForecastResult], List[str]]:
        """
        Forecast many medicines in one pass. Used by the monthly batch job.
 
        Returns (results, failed_item_ids) — failures are isolated per
        medicine so one bad SKU (e.g. brand-new product with 0 history)
        doesn't kill the whole monthly run.
        """
        cfg = self.config
        ids = item_ids or sorted(history_df[cfg.id_column].astype(str).unique())
 
        results: List[MedicineForecastResult] = []
        failed: List[str] = []
 
        for item_id in ids:
            try:
                results.append(self.forecast_medicine(item_id, history_df))
            except Exception:
                logger.exception("Forecast failed for item_id=%s", item_id)
                failed.append(str(item_id))
 
        return results, failed
 
    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
 
    def _to_result(
        self, item_id: str, context_used: int, raw_forecast: pd.DataFrame
    ) -> MedicineForecastResult:
        cfg = self.config
 
        # predict_df (per your notebook) renames to Medicine_ID/Forecast_Date/
        # predictions upstream in forecast_single_medicine — here we work with
        # the raw chronos output columns directly (quantile levels as string
        # column names, e.g. "0.1".."0.9") so this module has no dependency
        # on the notebook's renaming step.
        days: List[ForecastDayResult] = []
 
        for _, row in raw_forecast.iterrows():
            quantiles = QuantileForecast(
                p10=max(float(row["0.1"]), 0.0),
                p20=max(float(row["0.2"]), 0.0),
                p30=max(float(row["0.3"]), 0.0),
                p40=max(float(row["0.4"]), 0.0),
                p50=max(float(row["0.5"]), 0.0),
                p60=max(float(row["0.6"]), 0.0),
                p70=max(float(row["0.7"]), 0.0),
                p80=max(float(row["0.8"]), 0.0),
                p90=max(float(row["0.9"]), 0.0),
            )
            point = getattr(quantiles, f"p{int(cfg.point_quantile * 100)}")
 
            days.append(
                ForecastDayResult(
                    forecast_date=pd.Timestamp(row[cfg.timestamp_column]).date(),
                    predicted_demand=round(point, 2),
                    quantiles=quantiles,
                )
            )
 
        return MedicineForecastResult(
            medicine_id=item_id,
            generated_at=datetime.now(timezone.utc),
            context_length_used=context_used,
            prediction_length=self.config.prediction_length,
            model_id=self.config.model_id,
            days=days,
        )
 