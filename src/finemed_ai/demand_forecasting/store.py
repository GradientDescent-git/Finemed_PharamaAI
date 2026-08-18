from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from finemed_ai.demand_forecasting.schemas import (
    ForecastDayResult,
    ForecastSummary,
    MedicineForecastResult,
    QuantileForecast,
)

logger = logging.getLogger(__name__)


class ForecastNotFoundError(KeyError):
    """Raised when a requested medicine has no forecast."""


class ForecastStore:
    """
    Loads the latest production forecast and serves it in memory.

    The store provides a stable read layer for:
        - FastAPI
        - LLM tools
        - ranking/comparison
        - uncertainty analysis

    Expected forecast columns:

        Medicine_ID
        Forecast_Date
        Selected_Model
        Predicted_Demand
        P10
        P20 (optional)
        P30 (optional)
        P40 (optional)
        P50
        P60 (optional)
        P70 (optional)
        P80 (optional)
        P90
        Context_Length_Used
        Prediction_Length
        Generated_At
    """

    REQUIRED_COLUMNS = {
        "Medicine_ID",
        "Forecast_Date",
        "Selected_Model",
        "Predicted_Demand",
        "P10",
        "P50",
        "P90",
        "Context_Length_Used",
        "Prediction_Length",
        "Generated_At",
    }

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, latest_path: Path):
        self.latest_path = Path(latest_path)

        self._df: Optional[pd.DataFrame] = None
        self._loaded_at: Optional[float] = None

        self.reload()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def reload(self) -> None:
        """
        Reload the latest forecast from disk.

        If the file does not exist, the store remains empty.
        """

        if not self.latest_path.exists():
            logger.warning(
                "No forecast file at %s yet - store is empty "
                "until the first monthly run completes.",
                self.latest_path,
            )

            self._df = pd.DataFrame()
            self._loaded_at = None
            return

        df = pd.read_parquet(
            self.latest_path
        )
        if "Selected_Model" not in df.columns and "Model_ID" in df.columns:
            df["Selected_Model"] = df["Model_ID"]

        if "Prediction_Length" not in df.columns:
            if "Forecast_Date" in df.columns:
                df["Prediction_Length"] = (df.groupby("Medicine_ID")["Forecast_Date"].transform("size").astype(int))

        self._validate_loaded_dataframe(df)

        # Normalize identifiers consistently.
        df["Medicine_ID"] = (
            df["Medicine_ID"]
            .astype(str)
            .str.strip()
        )

        df["Forecast_Date"] = pd.to_datetime(
            df["Forecast_Date"],
            errors="coerce",
        )

        if df["Forecast_Date"].isna().any():
            raise ValueError(
                "Forecast store contains invalid Forecast_Date values."
            )

        self._df = (
            df.sort_values(
                [
                    "Medicine_ID",
                    "Forecast_Date",
                ]
            )
            .reset_index(drop=True)
        )

        self._loaded_at = (
            self.latest_path.stat().st_mtime
        )

        logger.info(
            "ForecastStore loaded %d rows across %d medicines from %s",
            len(self._df),
            self._df["Medicine_ID"].nunique(),
            self.latest_path,
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @classmethod
    def _validate_loaded_dataframe(
        cls,
        df: pd.DataFrame,
    ) -> None:
        if not isinstance(
            df,
            pd.DataFrame,
        ):
            raise TypeError(
                "Forecast file did not load as a pandas DataFrame."
            )

        if df.empty:
            raise ValueError(
                "Forecast file exists but contains no rows."
            )

        missing = (
            cls.REQUIRED_COLUMNS
            - set(df.columns)
        )

        if missing:
            raise ValueError(
                "Forecast file is missing required columns: "
                f"{sorted(missing)}"
            )

        if df["Medicine_ID"].isna().any():
            raise ValueError(
                "Forecast file contains null Medicine_ID values."
            )

        medicine_ids = (
            df["Medicine_ID"]
            .astype(str)
            .str.strip()
        )

        if medicine_ids.eq("").any():
            raise ValueError(
                "Forecast file contains empty Medicine_ID values."
            )

        predicted = pd.to_numeric(
            df["Predicted_Demand"],
            errors="coerce",
        )

        if predicted.isna().any():
            raise ValueError(
                "Forecast file contains invalid Predicted_Demand values."
            )

        if not np.isfinite(
            predicted.to_numpy(dtype=float)
        ).all():
            raise ValueError(
                "Forecast file contains non-finite Predicted_Demand values."
            )

        if (
            predicted < 0
        ).any():
            raise ValueError(
                "Forecast file contains negative Predicted_Demand values."
            )

        dates = pd.to_datetime(
            df["Forecast_Date"],
            errors="coerce",
        )

        if dates.isna().any():
            raise ValueError(
                "Forecast file contains invalid Forecast_Date values."
            )

        duplicate_dates = df.duplicated(
            subset=[
                "Medicine_ID",
                "Forecast_Date",
            ]
        )

        if duplicate_dates.any():
            raise ValueError(
                "Forecast file contains duplicate "
                "Medicine_ID + Forecast_Date rows."
            )

    # ------------------------------------------------------------------
    # Staleness
    # ------------------------------------------------------------------

    def is_stale(self) -> bool:
        """
        Return True when latest.parquet changed since the store loaded it.
        """

        if not self.latest_path.exists():
            return False

        if self._loaded_at is None:
            return True

        return (
            self.latest_path.stat().st_mtime
            != self._loaded_at
        )

    # ------------------------------------------------------------------
    # Medicine IDs
    # ------------------------------------------------------------------

    def list_medicine_ids(self) -> List[str]:
        if (
            self._df is None
            or self._df.empty
        ):
            return []

        return sorted(
            self._df["Medicine_ID"]
            .astype(str)
            .unique()
            .tolist()
        )

    # ------------------------------------------------------------------
    # Single medicine
    # ------------------------------------------------------------------

    def get(
        self,
        medicine_id: str,
    ) -> MedicineForecastResult:
        medicine_id = str(
            medicine_id
        ).strip()

        if (
            self._df is None
            or self._df.empty
        ):
            raise ForecastNotFoundError(
                f"No forecasts loaded yet "
                f"(medicine_id={medicine_id})"
            )

        rows = self._df[
            self._df["Medicine_ID"]
            == medicine_id
        ].copy()

        if rows.empty:
            raise ForecastNotFoundError(
                f"No forecast for medicine_id={medicine_id}"
            )

        rows = rows.sort_values(
            "Forecast_Date"
        )

        days = []

        for _, row in rows.iterrows():

            # ----------------------------------------------------------
            # TSB has no probabilistic quantiles.
            #
            # For the unified API schema, use the point prediction as
            # all quantile values when quantiles are unavailable.
            #
            # This keeps the public MedicineForecastResult contract
            # usable by existing API/LLM consumers while preserving
            # the Selected_Model metadata in the underlying dataframe.
            # ----------------------------------------------------------

            if (
                pd.isna(row.get("P10"))
                or pd.isna(row.get("P50"))
                or pd.isna(row.get("P90"))
            ):
                point = float(
                    row["Predicted_Demand"]
                )

                p10 = point
                p20 = point
                p30 = point
                p40 = point
                p50 = point
                p60 = point
                p70 = point
                p80 = point
                p90 = point

            else:
                p10 = float(row["P10"])
                p20 = float(
                    row.get("P20", row["P10"])
                )
                p30 = float(
                    row.get("P30", row["P10"])
                )
                p40 = float(
                    row.get("P40", row["P10"])
                )
                p50 = float(row["P50"])
                p60 = float(
                    row.get("P60", row["P50"])
                )
                p70 = float(
                    row.get("P70", row["P50"])
                )
                p80 = float(
                    row.get("P80", row["P50"])
                )
                p90 = float(row["P90"])

            days.append(
                ForecastDayResult(
                    forecast_date=(
                        pd.Timestamp(
                            row["Forecast_Date"]
                        ).date()
                    ),
                    predicted_demand=float(
                        row["Predicted_Demand"]
                    ),
                    quantiles=QuantileForecast(
                        p10=p10,
                        p20=p20,
                        p30=p30,
                        p40=p40,
                        p50=p50,
                        p60=p60,
                        p70=p70,
                        p80=p80,
                        p90=p90,
                    ),
                )
            )

        first = rows.iloc[0]

        return MedicineForecastResult(
            medicine_id=medicine_id,
            generated_at=(
                pd.Timestamp(
                    first["Generated_At"]
                ).to_pydatetime()
            ),
            context_length_used=int(
                first["Context_Length_Used"]
            ),
            prediction_length=len(days),
            model_id=str(
                first["Selected_Model"]
            ),
            days=days,
        )

    # ------------------------------------------------------------------
    # All summaries
    # ------------------------------------------------------------------

    def get_all_summaries(
        self,
    ) -> List[ForecastSummary]:
        """
        Build one ForecastSummary per medicine.
        """

        summaries: List[ForecastSummary] = []

        for medicine_id in self.list_medicine_ids():
            try:
                summaries.append(
                    self.get(
                        medicine_id
                    ).to_summary()
                )
            except ForecastNotFoundError:
                continue

        return summaries

    # ------------------------------------------------------------------
    # Ranking
    # ------------------------------------------------------------------

    def get_top_demand(
        self,
        n: int = 10,
    ) -> List[ForecastSummary]:
        """
        Top N medicines by total predicted demand.
        """

        if n <= 0:
            return []

        summaries = self.get_all_summaries()

        return sorted(
            summaries,
            key=lambda s: (
                s.total_predicted_demand
            ),
            reverse=True,
        )[:n]

    # ------------------------------------------------------------------
    # Trend filtering
    # ------------------------------------------------------------------

    def get_by_trend(
        self,
        trend: str,
        n: int = 10,
    ) -> List[ForecastSummary]:
        """
        Medicines matching a trend direction.

        Supported values:
            increasing
            decreasing
            stable
            flat

        "flat" is treated as "stable".
        """

        if n <= 0:
            return []

        normalized_trend = (
            str(trend)
            .strip()
            .lower()
        )

        if normalized_trend == "flat":
            normalized_trend = "stable"

        summaries = [
            summary
            for summary in self.get_all_summaries()
            if summary.trend
            == normalized_trend
        ]

        return sorted(
            summaries,
            key=lambda s: abs(
                s.trend_pct_change
            ),
            reverse=True,
        )[:n]

    # ------------------------------------------------------------------
    # Uncertainty
    # ------------------------------------------------------------------

    def get_most_uncertain(
        self,
        n: int = 10,
    ) -> List[dict]:
        """
        Return medicines with the highest relative P10-P90 spread.

        TSB forecasts have no true probabilistic interval, so they are
        excluded from uncertainty ranking.
        """

        if (
            self._df is None
            or self._df.empty
            or n <= 0
        ):
            return []

        results = []

        for medicine_id, group in self._df.groupby(
            "Medicine_ID"
        ):

            p10 = pd.to_numeric(
                group["P10"],
                errors="coerce",
            )

            p50 = pd.to_numeric(
                group["P50"],
                errors="coerce",
            )

            p90 = pd.to_numeric(
                group["P90"],
                errors="coerce",
            )

            if (
                p10.isna().all()
                or p50.isna().all()
                or p90.isna().all()
            ):
                # TSB / point-only forecast.
                continue

            avg_p50 = float(
                p50.mean()
            )

            avg_spread = float(
                (p90 - p10).mean()
            )

            if avg_p50 > 0:
                relative_uncertainty = (
                    avg_spread
                    / avg_p50
                    * 100.0
                )
            else:
                relative_uncertainty = 0.0

            results.append(
                {
                    "medicine_id": str(
                        medicine_id
                    ),
                    "avg_p50": round(
                        avg_p50,
                        2,
                    ),
                    "avg_p10_p90_spread": round(
                        avg_spread,
                        2,
                    ),
                    "relative_uncertainty_pct": round(
                        relative_uncertainty,
                        1,
                    ),
                }
            )

        return sorted(
            results,
            key=lambda r: (
                r["relative_uncertainty_pct"]
            ),
            reverse=True,
        )[:n]

    # ------------------------------------------------------------------
    # Direct comparison
    # ------------------------------------------------------------------

    def compare(
        self,
        medicine_ids: List[str],
    ) -> List[ForecastSummary]:
        """
        Return summaries for the requested medicines.
        """

        results = []

        for medicine_id in medicine_ids:

            try:
                results.append(
                    self.get(
                        str(medicine_id)
                    ).to_summary()
                )
            except ForecastNotFoundError:
                continue

        return results