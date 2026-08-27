from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class DriftReport:
    status: str  # NORMAL, WARNING, CRITICAL
    run_timestamp: str
    mean_shift_pct: float
    variance_shift_pct: float
    zero_ratio_current: float
    zero_ratio_baseline: float
    new_medicines_count: int
    discontinued_medicines_count: int
    details: Dict[str, Any] = field(default_factory=dict)


class DemandDriftDetector:
    """
    Production Data Drift Detector.

    Monitors new monthly demand series against a frozen historical baseline
    to detect distribution shifts, sparsity changes, or item churn before
    triggering retraining or forecast publication.
    """

    def __init__(
        self,
        mean_threshold_pct: float = 30.0,
        zero_ratio_threshold_diff: float = 0.20,
    ) -> None:
        self.mean_threshold_pct = mean_threshold_pct
        self.zero_ratio_threshold_diff = zero_ratio_threshold_diff

    def detect_drift(
        self,
        baseline_df: pd.DataFrame,
        current_df: pd.DataFrame,
    ) -> DriftReport:
        """
        Compare current demand series against baseline demand series.
        """
        now_str = datetime.now().isoformat()

        if baseline_df.empty or current_df.empty:
            return DriftReport(
                status="WARNING",
                run_timestamp=now_str,
                mean_shift_pct=0.0,
                variance_shift_pct=0.0,
                zero_ratio_current=0.0,
                zero_ratio_baseline=0.0,
                new_medicines_count=0,
                discontinued_medicines_count=0,
                details={"message": "Baseline or current demand dataset is empty."},
            )

        # 1. Demand stats
        base_demand = baseline_df["Daily_Demand"].to_numpy()
        curr_demand = current_df["Daily_Demand"].to_numpy()

        base_mean = float(np.mean(base_demand)) if len(base_demand) > 0 else 0.0
        curr_mean = float(np.mean(curr_demand)) if len(curr_demand) > 0 else 0.0

        mean_shift_pct = (
            abs(curr_mean - base_mean) / base_mean * 100.0 if base_mean > 0 else 0.0
        )

        base_var = float(np.var(base_demand)) if len(base_demand) > 0 else 0.0
        curr_var = float(np.var(curr_demand)) if len(curr_demand) > 0 else 0.0

        var_shift_pct = (
            abs(curr_var - base_var) / base_var * 100.0 if base_var > 0 else 0.0
        )

        # 2. Zero demand ratio
        base_zero = float(np.mean(base_demand == 0)) if len(base_demand) > 0 else 0.0
        curr_zero = float(np.mean(curr_demand == 0)) if len(curr_demand) > 0 else 0.0

        # 3. Item churn
        base_items = set(baseline_df["Medicine_ID"].unique())
        curr_items = set(current_df["Medicine_ID"].unique())

        new_items = len(curr_items - base_items)
        discontinued_items = len(base_items - curr_items)

        # 4. Status determination
        status = "NORMAL"
        reasons = []

        if mean_shift_pct > self.mean_threshold_pct * 1.5 or abs(curr_zero - base_zero) > 0.35:
            status = "CRITICAL"
            reasons.append("Extreme mean demand shift or zero-ratio disruption.")
        elif mean_shift_pct > self.mean_threshold_pct or abs(curr_zero - base_zero) > self.zero_ratio_threshold_diff:
            status = "WARNING"
            reasons.append("Moderate demand distribution drift detected.")

        logger.info(
            "Drift Detection Complete | status=%s | mean_shift=%.2f%% | zero_ratio_diff=%.2f",
            status,
            mean_shift_pct,
            abs(curr_zero - base_zero),
        )

        return DriftReport(
            status=status,
            run_timestamp=now_str,
            mean_shift_pct=round(mean_shift_pct, 2),
            variance_shift_pct=round(var_shift_pct, 2),
            zero_ratio_current=round(curr_zero, 4),
            zero_ratio_baseline=round(base_zero, 4),
            new_medicines_count=new_items,
            discontinued_medicines_count=discontinued_items,
            details={"reasons": reasons},
        )
