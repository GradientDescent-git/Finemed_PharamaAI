from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class MedicineEvaluation(BaseModel):
    medicine_id: str
    sample_count: int
    total_actual_qty: float
    total_predicted_qty: float
    wape_pct: float
    mae: float
    smape_pct: float
    mbe: float
    p10_p90_coverage_pct: float
    evaluated_at: str


class OverallEvaluation(BaseModel):
    evaluation_id: str
    evaluated_at: str
    total_medicines_evaluated: int
    total_actual_units: float
    total_predicted_units: float
    overall_wape_pct: float
    overall_mae: float
    overall_smape_pct: float
    overall_mbe: float
    overall_coverage_pct: float
    medicines: List[MedicineEvaluation] = Field(default_factory=list)


def compute_metrics(actuals: np.ndarray, predictions: np.ndarray, p10s: Optional[np.ndarray] = None, p90s: Optional[np.ndarray] = None) -> Dict[str, float]:
    actuals = np.asarray(actuals, dtype=float)
    predictions = np.asarray(predictions, dtype=float)
    
    total_actual = float(np.sum(actuals))
    total_pred = float(np.sum(predictions))
    abs_errors = np.abs(actuals - predictions)
    
    wape = (float(np.sum(abs_errors)) / total_actual * 100.0) if total_actual > 0 else 0.0
    mae = float(np.mean(abs_errors)) if len(actuals) > 0 else 0.0
    
    denom = np.abs(actuals) + np.abs(predictions)
    valid_denom = np.where(denom == 0, 1.0, denom)
    smape_array = np.where(denom == 0, 0.0, 200.0 * abs_errors / valid_denom)
    smape = float(np.mean(smape_array)) if len(actuals) > 0 else 0.0
    
    mbe = float(np.mean(predictions - actuals)) if len(actuals) > 0 else 0.0
    absolute_error_sum = float(np.sum(abs_errors))
    
    coverage = 0.0
    if p10s is not None and p90s is not None and len(actuals) > 0:
        in_bounds = (actuals >= p10s) & (actuals <= p90s)
        coverage = float(np.mean(in_bounds) * 100.0)
        
    return {
        "total_actual": total_actual,
        "total_predicted": total_pred,
        "total_absolute_error": absolute_error_sum,
        "absolute_error_sum": absolute_error_sum,
        "wape_pct": round(wape, 2),
        "mae": round(mae, 2),
        "smape_pct": round(smape, 2),
        "mbe": round(mbe, 2),
        "coverage_pct": round(coverage, 2),
        }


class ForecastEvaluator:
    def __init__(self, forecast_dir: Path):
        self.forecast_dir = forecast_dir
        self.evaluations_file = forecast_dir / "evaluations.json"

    def evaluate(self, actuals_df: pd.DataFrame, forecast_df: pd.DataFrame) -> OverallEvaluation:
        act_df = actuals_df.copy()
        fct_df = forecast_df.copy()
        
        id_col_act = "Medicine_ID" if "Medicine_ID" in act_df.columns else ("item_id" if "item_id" in act_df.columns else "MDCODE")
        date_col_act = "Forecast_Date" if "Forecast_Date" in act_df.columns else ("timestamp" if "timestamp" in act_df.columns else "INVDT")
        qty_col_act = "Demand_Qty" if "Demand_Qty" in act_df.columns else ("target" if "target" in act_df.columns else "QTY")
        
        act_df = act_df.rename(columns={id_col_act: "Medicine_ID", date_col_act: "Forecast_Date", qty_col_act: "Actual_Demand"})
        act_df["Medicine_ID"] = act_df["Medicine_ID"].astype(str)
        act_df["Forecast_Date"] = pd.to_datetime(act_df["Forecast_Date"]).dt.date
        
        fct_df["Medicine_ID"] = fct_df["Medicine_ID"].astype(str)
        fct_df["Forecast_Date"] = pd.to_datetime(fct_df["Forecast_Date"]).dt.date
        
        merged = pd.merge(fct_df, act_df, on=["Medicine_ID", "Forecast_Date"], how="inner")
        
        if merged.empty:
            logger.warning("No overlapping dates/medicines between actuals and forecast for evaluation.")
            eval_id = f"eval_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
            empty_eval = OverallEvaluation(
                evaluation_id=eval_id,
                evaluated_at=datetime.now(timezone.utc).isoformat(),
                total_medicines_evaluated=0,
                total_actual_units=0.0,
                total_predicted_units=0.0,
                overall_wape_pct=0.0,
                overall_mae=0.0,
                overall_smape_pct=0.0,
                overall_mbe=0.0,
                overall_coverage_pct=0.0,
                medicines=[],
            )
            self._save_evaluation(empty_eval)
            return empty_eval

        now_str = datetime.now(timezone.utc).isoformat()
        med_evals: List[MedicineEvaluation] = []
        
        for med_id, group in merged.groupby("Medicine_ID"):
            acts = group["Actual_Demand"].values
            preds = group["Predicted_Demand"].values
            p10s = group["P10"].values if "P10" in group.columns else None
            p90s = group["P90"].values if "P90" in group.columns else None
            
            m = compute_metrics(acts, preds, p10s, p90s)
            med_evals.append(MedicineEvaluation(
                medicine_id=str(med_id),
                sample_count=len(group),
                total_actual_qty=m["total_actual"],
                total_predicted_qty=m["total_predicted"],
                wape_pct=m["wape_pct"],
                mae=m["mae"],
                smape_pct=m["smape_pct"],
                mbe=m["mbe"],
                p10_p90_coverage_pct=m["coverage_pct"],
                evaluated_at=now_str,
            ))
            
        overall_acts = merged["Actual_Demand"].values
        overall_preds = merged["Predicted_Demand"].values
        overall_p10s = merged["P10"].values if "P10" in merged.columns else None
        overall_p90s = merged["P90"].values if "P90" in merged.columns else None
        om = compute_metrics(overall_acts, overall_preds, overall_p10s, overall_p90s)
        
        eval_id = f"eval_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        overall_eval = OverallEvaluation(
            evaluation_id=eval_id,
            evaluated_at=now_str,
            total_medicines_evaluated=len(med_evals),
            total_actual_units=om["total_actual"],
            total_predicted_units=om["total_predicted"],
            overall_wape_pct=om["wape_pct"],
            overall_mae=om["mae"],
            overall_smape_pct=om["smape_pct"],
            overall_mbe=om["mbe"],
            overall_coverage_pct=om["coverage_pct"],
            medicines=med_evals,
        )
        
        self._save_evaluation(overall_eval)
        return overall_eval

    def _save_evaluation(self, eval_result: OverallEvaluation) -> None:
        self.forecast_dir.mkdir(parents=True, exist_ok=True)
        eval_dict = eval_result.model_dump(mode="json")
        self.evaluations_file.write_text(json.dumps(eval_dict, indent=2))
        logger.info("Evaluation report saved to %s", self.evaluations_file)

    def load_latest_evaluation(self) -> Optional[OverallEvaluation]:
        if not self.evaluations_file.exists():
            return None
        try:
            data = json.loads(self.evaluations_file.read_text())
            return OverallEvaluation(**data)
        except Exception:
            logger.exception("Failed to load evaluation from %s", self.evaluations_file)
            return None
