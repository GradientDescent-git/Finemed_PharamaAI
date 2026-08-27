from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class PostEvaluationResult:
    run_id: str
    evaluated_at: str
    total_medicines: int
    wape: float
    bias: float
    mae: float
    status: str  # PASSED, DEGRADED, CRITICAL


class PostDeploymentEvaluator:
    """
    Evaluates published forecasts against actual demand after the forecast period completes.

    Calculates:
        - WAPE (Weighted Absolute Percentage Error)
        - Forecast Bias (Over-forecasting vs Under-forecasting ratio)
        - MAE (Mean Absolute Error)
    """

    def evaluate(
        self,
        run_id: str,
        predictions_df: pd.DataFrame,
        actuals_df: pd.DataFrame,
        wape_degradation_threshold: float = 0.40,
    ) -> PostEvaluationResult:
        """
        Merge predictions and actuals on (Medicine_ID, Forecast_Date) and compute accuracy metrics.
        """
        from datetime import datetime

        if predictions_df.empty or actuals_df.empty:
            return PostEvaluationResult(
                run_id=run_id,
                evaluated_at=datetime.now().isoformat(),
                total_medicines=0,
                wape=0.0,
                bias=0.0,
                mae=0.0,
                status="DEGRADED",
            )

        merged = pd.merge(
            predictions_df,
            actuals_df,
            on=["Medicine_ID", "Forecast_Date"],
            suffixes=("_pred", "_actual"),
            how="inner",
        )

        if merged.empty:
            logger.warning("No matching dates found between predictions and actuals for run %s", run_id)
            return PostEvaluationResult(
                run_id=run_id,
                evaluated_at=datetime.now().isoformat(),
                total_medicines=0,
                wape=0.0,
                bias=0.0,
                mae=0.0,
                status="DEGRADED",
            )

        pred_col = "Predicted_Demand" if "Predicted_Demand" in merged.columns else "P50"
        actual_col = "Daily_Demand" if "Daily_Demand" in merged.columns else "Actual_Demand"

        y_true = merged[actual_col].to_numpy(dtype=float)
        y_pred = merged[pred_col].to_numpy(dtype=float)

        total_actual = np.sum(y_true)
        total_abs_err = np.sum(np.abs(y_true - y_pred))

        wape = float(total_abs_err / total_actual) if total_actual > 0 else 0.0
        mae = float(np.mean(np.abs(y_true - y_pred)))
        bias = float(np.sum(y_pred - y_true) / total_actual) if total_actual > 0 else 0.0

        status = "PASSED" if wape <= wape_degradation_threshold else "DEGRADED"

        logger.info(
            "Post-Deployment Evaluation Run %s | WAPE=%.4f | Bias=%.4f | MAE=%.4f | Status=%s",
            run_id,
            wape,
            bias,
            mae,
            status,
        )

        return PostEvaluationResult(
            run_id=run_id,
            evaluated_at=datetime.now().isoformat(),
            total_medicines=int(merged["Medicine_ID"].nunique()),
            wape=round(wape, 4),
            bias=round(bias, 4),
            mae=round(mae, 4),
            status=status,
        )
