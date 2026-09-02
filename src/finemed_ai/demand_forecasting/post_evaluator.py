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

        # Create working copies
        preds = predictions_df.copy()
        acts = actuals_df.copy()

        # 1. Resolve Medicine_ID column name
        pred_med_col = next((c for c in ["Medicine_ID", "mdcode", "item_id"] if c in preds.columns), None)
        act_med_col = next((c for c in ["Medicine_ID", "mdcode", "item_id"] if c in acts.columns), None)

        if not pred_med_col or not act_med_col:
            logger.warning("PostDeploymentEvaluator missing Medicine_ID column in predictions (%s) or actuals (%s)", pred_med_col, act_med_col)
            return PostEvaluationResult(
                run_id=run_id, evaluated_at=datetime.now().isoformat(), total_medicines=0,
                wape=0.0, bias=0.0, mae=0.0, status="DEGRADED"
            )

        # 2. Resolve Forecast_Date column name
        pred_date_col = next((c for c in ["Forecast_Date", "date", "timestamp"] if c in preds.columns), None)
        act_date_col = next((c for c in ["Forecast_Date", "Daily_Demand_Date", "date", "timestamp"] if c in acts.columns), None)

        if not pred_date_col or not act_date_col:
            logger.warning("PostDeploymentEvaluator missing date column in predictions (%s) or actuals (%s)", pred_date_col, act_date_col)
            return PostEvaluationResult(
                run_id=run_id, evaluated_at=datetime.now().isoformat(), total_medicines=0,
                wape=0.0, bias=0.0, mae=0.0, status="DEGRADED"
            )

        # Normalize Medicine_ID (string zfilled to 4 digits)
        preds["Medicine_ID_norm"] = preds[pred_med_col].astype(str).str.strip().str.zfill(4)
        acts["Medicine_ID_norm"] = acts[act_med_col].astype(str).str.strip().str.zfill(4)

        # Normalize Forecast_Date (normalized calendar date)
        preds["Forecast_Date_norm"] = pd.to_datetime(preds[pred_date_col], errors="coerce").dt.normalize()
        acts["Forecast_Date_norm"] = pd.to_datetime(acts[act_date_col], errors="coerce").dt.normalize()

        merged = pd.merge(
            preds,
            acts,
            left_on=["Medicine_ID_norm", "Forecast_Date_norm"],
            right_on=["Medicine_ID_norm", "Forecast_Date_norm"],
            suffixes=("_pred", "_actual"),
            how="inner",
        )

        if merged.empty:
            logger.warning(
                "No matching records found in PostDeploymentEvaluator merge for run %s | "
                "preds: shape=%s, med_sample=%s, dates=%s..%s | "
                "acts: shape=%s, med_sample=%s, dates=%s..%s",
                run_id,
                preds.shape, preds["Medicine_ID_norm"].head(3).tolist(), preds["Forecast_Date_norm"].min(), preds["Forecast_Date_norm"].max(),
                acts.shape, acts["Medicine_ID_norm"].head(3).tolist(), acts["Forecast_Date_norm"].min(), acts["Forecast_Date_norm"].max(),
            )
            return PostEvaluationResult(
                run_id=run_id,
                evaluated_at=datetime.now().isoformat(),
                total_medicines=0,
                wape=0.0,
                bias=0.0,
                mae=0.0,
                status="DEGRADED",
            )

        pred_col = next((c for c in ["Predicted_Demand", "P50", "predicted_demand"] if c in merged.columns), None) or ("Predicted_Demand_pred" if "Predicted_Demand_pred" in merged.columns else "Predicted_Demand")
        actual_col = next((c for c in ["Daily_Demand", "Actual_Demand", "Actual", "demand", "quantity"] if c in merged.columns), None) or ("Daily_Demand_actual" if "Daily_Demand_actual" in merged.columns else "Daily_Demand")

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
            total_medicines=int(merged["Medicine_ID_norm"].nunique()),
            wape=round(wape, 4),
            bias=round(bias, 4),
            mae=round(mae, 4),
            status=status,
        )
