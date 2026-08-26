# src/finemed_ai/forecast_intelligence/analytics.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ForecastAnalytics:
    """
    Deterministic analytics layer for production demand forecasts.

    This class performs aggregate analysis on the published forecast
    dataframe. It does not load files, resolve medicine names, route
    intents, or generate LLM responses.

    Expected forecast columns:
        Medicine_ID
        Forecast_Date
        Selected_Model
        Predicted_Demand
        Eligibility_Status
        Forecast_Status
    """

    forecast_df: pd.DataFrame

    REQUIRED_COLUMNS = {
        "Medicine_ID",
        "Forecast_Date",
        "Selected_Model",
        "Predicted_Demand",
        "Eligibility_Status",
        "Forecast_Status",
    }

    def __post_init__(self) -> None:
        missing = self.REQUIRED_COLUMNS - set(self.forecast_df.columns)

        if missing:
            raise ValueError(
                "Forecast dataframe is missing required columns: "
                f"{sorted(missing)}"
            )

    def _working_frame(self) -> pd.DataFrame:
        """Return a safe normalized copy for analytics."""
        df = self.forecast_df.copy()

        df["Forecast_Date"] = pd.to_datetime(
            df["Forecast_Date"],
            errors="coerce",
        )

        df["Predicted_Demand"] = pd.to_numeric(
            df["Predicted_Demand"],
            errors="coerce",
        ).fillna(0.0)

        return df

    def overall_summary(self) -> dict[str, Any]:
        """Return aggregate production forecast statistics."""
        df = self._working_frame()

        if df.empty:
            return {
                "medicine_count": 0,
                "forecast_record_count": 0,
                "forecast_start": None,
                "forecast_end": None,
                "forecast_days": 0,
                "total_predicted_demand": 0.0,
                "average_daily_predicted_demand": 0.0,
                "average_demand_per_medicine": 0.0,
            }

        forecast_dates = df["Forecast_Date"].dropna()

        medicine_count = int(df["Medicine_ID"].nunique())
        record_count = int(len(df))

        total_demand = float(df["Predicted_Demand"].sum())

        forecast_days = int(forecast_dates.nunique())

        average_daily = (
            float(total_demand / forecast_days)
            if forecast_days > 0
            else 0.0
        )

        average_per_medicine = (
            float(total_demand / medicine_count)
            if medicine_count > 0
            else 0.0
        )

        return {
            "medicine_count": medicine_count,
            "forecast_record_count": record_count,
            "forecast_start": (
                forecast_dates.min().date().isoformat()
                if not forecast_dates.empty
                else None
            ),
            "forecast_end": (
                forecast_dates.max().date().isoformat()
                if not forecast_dates.empty
                else None
            ),
            "forecast_days": forecast_days,
            "total_predicted_demand": total_demand,
            "average_daily_predicted_demand": average_daily,
            "average_demand_per_medicine": average_per_medicine,
        }

    def model_distribution(self) -> dict[str, int]:
        """Return unique medicine count grouped by selected model."""
        df = self._working_frame()

        if df.empty:
            return {}

        grouped = (
            df.groupby("Selected_Model")["Medicine_ID"]
            .nunique()
            .sort_values(ascending=False)
        )

        return {
            str(model): int(count)
            for model, count in grouped.items()
        }

    def eligibility_distribution(self) -> dict[str, int]:
        """Return unique medicine count grouped by eligibility status."""
        df = self._working_frame()

        if df.empty:
            return {}

        grouped = (
            df.groupby("Eligibility_Status")["Medicine_ID"]
            .nunique()
            .sort_values(ascending=False)
        )

        return {
            str(status): int(count)
            for status, count in grouped.items()
        }

    def forecast_status_distribution(self) -> dict[str, int]:
        """Return unique medicine count grouped by forecast status."""
        df = self._working_frame()

        if df.empty:
            return {}

        grouped = (
            df.groupby("Forecast_Status")["Medicine_ID"]
            .nunique()
            .sort_values(ascending=False)
        )

        return {
            str(status): int(count)
            for status, count in grouped.items()
        }

    def medicine_ranking(
        self,
        limit: int = 10,
        ascending: bool = False,
    ) -> list[dict[str, Any]]:
        """
        Rank medicines by total predicted demand.

        ascending=False -> highest demand first
        ascending=True  -> lowest demand first
        """
        if limit <= 0:
            raise ValueError("limit must be greater than 0")

        df = self._working_frame()

        if df.empty:
            return []

        ranking = (
            df.groupby("Medicine_ID", as_index=False)
            .agg(
                total_predicted_demand=(
                    "Predicted_Demand",
                    "sum",
                ),
                forecast_days=(
                    "Forecast_Date",
                    "nunique",
                ),
                average_daily_demand=(
                    "Predicted_Demand",
                    "mean",
                ),
            )
            .sort_values(
                "total_predicted_demand",
                ascending=ascending,
            )
            .head(limit)
        )

        results: list[dict[str, Any]] = []

        for _, row in ranking.iterrows():
            results.append(
                {
                    "medicine_id": str(row["Medicine_ID"]),
                    "total_predicted_demand": float(
                        row["total_predicted_demand"]
                    ),
                    "forecast_days": int(row["forecast_days"]),
                    "average_daily_demand": float(
                        row["average_daily_demand"]
                    ),
                }
            )

        return results

    def top_medicines(
        self,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Return medicines with the highest predicted demand."""
        return self.medicine_ranking(
            limit=limit,
            ascending=False,
        )

    def bottom_medicines(
        self,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Return medicines with the lowest predicted demand."""
        return self.medicine_ranking(
            limit=limit,
            ascending=True,
        )

    def model_summary(self) -> list[dict[str, Any]]:
        """Return demand contribution and medicine counts by model."""
        df = self._working_frame()

        if df.empty:
            return []

        grouped = (
            df.groupby("Selected_Model", as_index=False)
            .agg(
                medicine_count=(
                    "Medicine_ID",
                    "nunique",
                ),
                total_predicted_demand=(
                    "Predicted_Demand",
                    "sum",
                ),
                forecast_record_count=(
                    "Medicine_ID",
                    "size",
                ),
            )
            .sort_values(
                "total_predicted_demand",
                ascending=False,
            )
        )

        total = float(
            grouped["total_predicted_demand"].sum()
        )

        results: list[dict[str, Any]] = []

        for _, row in grouped.iterrows():
            demand = float(row["total_predicted_demand"])

            results.append(
                {
                    "selected_model": str(
                        row["Selected_Model"]
                    ),
                    "medicine_count": int(
                        row["medicine_count"]
                    ),
                    "forecast_record_count": int(
                        row["forecast_record_count"]
                    ),
                    "total_predicted_demand": demand,
                    "demand_share_pct": (
                        float(demand / total * 100)
                        if total > 0
                        else 0.0
                    ),
                }
            )

        return results

    def full_dashboard(self) -> dict[str, Any]:
        """
        Return a deterministic aggregate analytics payload.

        Suitable for:
        - Forecast Intelligence service
        - LLM context construction
        - API responses
        - Dashboard endpoints
        """
        return {
            "summary": self.overall_summary(),
            "model_distribution": self.model_distribution(),
            "eligibility_distribution": (
                self.eligibility_distribution()
            ),
            "forecast_status_distribution": (
                self.forecast_status_distribution()
            ),
            "model_summary": self.model_summary(),
            "top_medicines": self.top_medicines(limit=10),
            "bottom_medicines": self.bottom_medicines(
                limit=10
            ),
        }