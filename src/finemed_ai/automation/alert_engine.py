from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class AlertSeverity(str, Enum):
    CRITICAL = "CRITICAL"
    WARNING = "WARNING"
    INFO = "INFO"


class AlertType(str, Enum):
    STOCKOUT_RISK = "STOCKOUT_RISK"
    DEMAND_SPIKE = "DEMAND_SPIKE"
    DEMAND_DROP = "DEMAND_DROP"
    OVERSTOCK_RISK = "OVERSTOCK_RISK"
    HIGH_UNCERTAINTY = "HIGH_UNCERTAINTY"


class Alert(BaseModel):
    id: str
    medicine_id: str
    severity: AlertSeverity
    alert_type: AlertType
    title: str
    description: str
    recommended_action: str
    metric_value: float
    created_at: str


class AlertStore(BaseModel):
    last_updated: str
    total_alerts: int
    critical_count: int
    warning_count: int
    alerts: List[Alert] = Field(default_factory=list)


class AlertEngine:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.alerts_file = output_dir / "alerts.json"

    def scan_forecasts(
        self,
        forecast_df: pd.DataFrame,
        historical_demand_df: Optional[pd.DataFrame] = None,
        inventory_df: Optional[pd.DataFrame] = None,
    ) -> AlertStore:
        """
        Scans forecasts and inventory data to identify operational risks.
        """
        now_str = datetime.now(timezone.utc).isoformat()
        alerts: List[Alert] = []
        alert_counter = 1

        fdf = forecast_df.copy()
        fdf["Medicine_ID"] = fdf["Medicine_ID"].astype(str)

        # Build medicine-level forecast summary
        for med_id, group in fdf.groupby("Medicine_ID"):
            med_id_str = str(med_id)
            total_30d_forecast = float(group["Predicted_Demand"].sum())
            avg_daily_forecast = float(group["Predicted_Demand"].mean())
            
            p10_sum = float(group["P10"].sum()) if "P10" in group.columns else total_30d_forecast * 0.7
            p90_sum = float(group["P90"].sum()) if "P90" in group.columns else total_30d_forecast * 1.3
            uncertainty_spread = p90_sum - p10_sum

            # Historical baseline comparison
            hist_avg_daily = 0.0
            if historical_demand_df is not None and not historical_demand_df.empty:
                id_col = "Medicine_ID" if "Medicine_ID" in historical_demand_df.columns else ("item_id" if "item_id" in historical_demand_df.columns else "MDCODE")
                qty_col = "Demand_Qty" if "Demand_Qty" in historical_demand_df.columns else ("target" if "target" in historical_demand_df.columns else "QTY")
                med_hist = historical_demand_df[historical_demand_df[id_col].astype(str) == med_id_str]
                if not med_hist.empty:
                    hist_avg_daily = float(med_hist[qty_col].mean())

            # Inventory baseline
            stock_on_hand = 0.0
            if inventory_df is not None and not inventory_df.empty:
                id_col_inv = "Medicine_ID" if "Medicine_ID" in inventory_df.columns else "MDCODE"
                soh_col = "Stock_On_Hand" if "Stock_On_Hand" in inventory_df.columns else ("SOH" if "SOH" in inventory_df.columns else "QTY")
                if id_col_inv in inventory_df.columns and soh_col in inventory_df.columns:
                    med_inv = inventory_df[inventory_df[id_col_inv].astype(str) == med_id_str]
                    if not med_inv.empty:
                        stock_on_hand = float(med_inv[soh_col].sum())

            # Rule 1: Stockout Risk (if inventory is tracked and less than 30d demand)
            if stock_on_hand > 0 and stock_on_hand < total_30d_forecast:
                shortage = total_30d_forecast - stock_on_hand
                alerts.append(Alert(
                    id=f"ALT-{alert_counter:04d}",
                    medicine_id=med_id_str,
                    severity=AlertSeverity.CRITICAL if shortage > 0.3 * total_30d_forecast else AlertSeverity.WARNING,
                    alert_type=AlertType.STOCKOUT_RISK,
                    title=f"Stockout Risk for Medicine {med_id_str}",
                    description=f"Current stock ({stock_on_hand:.0f} units) is below 30-day forecasted demand ({total_30d_forecast:.0f} units). Shortage of {shortage:.0f} units expected.",
                    recommended_action=f"Place purchase order for at least {shortage:.0f} units immediately.",
                    metric_value=round(shortage, 1),
                    created_at=now_str,
                ))
                alert_counter += 1

            # Rule 2: Demand Spike (> 50% above historical daily average)
            if hist_avg_daily > 0 and avg_daily_forecast > 1.5 * hist_avg_daily:
                pct_inc = ((avg_daily_forecast - hist_avg_daily) / hist_avg_daily) * 100.0
                alerts.append(Alert(
                    id=f"ALT-{alert_counter:04d}",
                    medicine_id=med_id_str,
                    severity=AlertSeverity.WARNING,
                    alert_type=AlertType.DEMAND_SPIKE,
                    title=f"Surge in Demand for Medicine {med_id_str}",
                    description=f"Forecasted daily demand ({avg_daily_forecast:.1f} units/day) is {pct_inc:.1f}% higher than historical average ({hist_avg_daily:.1f} units/day).",
                    recommended_action="Review procurement lead times and re-evaluate buffer stock.",
                    metric_value=round(pct_inc, 1),
                    created_at=now_str,
                ))
                alert_counter += 1

            # Rule 3: High Uncertainty Alert (Wide P90-P10 interval)
            if total_30d_forecast > 10 and uncertainty_spread > 2.5 * total_30d_forecast:
                alerts.append(Alert(
                    id=f"ALT-{alert_counter:04d}",
                    medicine_id=med_id_str,
                    severity=AlertSeverity.INFO,
                    alert_type=AlertType.HIGH_UNCERTAINTY,
                    title=f"High Forecast Uncertainty for Medicine {med_id_str}",
                    description=f"Wide confidence interval (P10: {p10_sum:.0f}, P90: {p90_sum:.0f} units) indicates volatile demand patterns.",
                    recommended_action="Monitor weekly sales closely to adjust orders as demand stabilizes.",
                    metric_value=round(uncertainty_spread, 1),
                    created_at=now_str,
                ))
                alert_counter += 1

        crit_count = sum(1 for a in alerts if a.severity == AlertSeverity.CRITICAL)
        warn_count = sum(1 for a in alerts if a.severity == AlertSeverity.WARNING)

        store = AlertStore(
            last_updated=now_str,
            total_alerts=len(alerts),
            critical_count=crit_count,
            warning_count=warn_count,
            alerts=alerts,
        )

        self._save_alerts(store)
        return store

    def _save_alerts(self, store: AlertStore) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.alerts_file.write_text(json.dumps(store.model_dump(mode="json"), indent=2))
        logger.info("Saved %d risk alerts to %s", len(store.alerts), self.alerts_file)

    def load_latest_alerts(self) -> AlertStore:
        if not self.alerts_file.exists():
            return AlertStore(
                last_updated=datetime.now(timezone.utc).isoformat(),
                total_alerts=0,
                critical_count=0,
                warning_count=0,
                alerts=[],
            )
        try:
            data = json.loads(self.alerts_file.read_text())
            return AlertStore(**data)
        except Exception:
            logger.exception("Failed to read alerts from %s", self.alerts_file)
            return AlertStore(
                last_updated=datetime.now(timezone.utc).isoformat(),
                total_alerts=0,
                critical_count=0,
                warning_count=0,
                alerts=[],
            )
