from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)


# ============================================================================
# Pydantic models
# ============================================================================


class MedicineEvaluation(BaseModel):
    """
    Evaluation metrics for a single medicine.
    """

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
    """
    Aggregate evaluation result for a forecast run.

    The overlap fields are intentionally persisted so a zero-metric
    evaluation can be distinguished from a genuinely perfect forecast.
    """

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

    # ------------------------------------------------------------------
    # Evaluation observability
    # ------------------------------------------------------------------

    forecast_rows: int = 0
    actual_rows: int = 0
    matched_rows: int = 0

    forecast_medicines: int = 0
    actual_medicines: int = 0
    matched_medicines: int = 0

    forecast_start_date: Optional[str] = None
    forecast_end_date: Optional[str] = None

    actual_start_date: Optional[str] = None
    actual_end_date: Optional[str] = None

    overlap_start_date: Optional[str] = None
    overlap_end_date: Optional[str] = None

    has_overlap: bool = False

    medicines: List[MedicineEvaluation] = Field(
        default_factory=list
    )


# ============================================================================
# Constants
# ============================================================================


MEDICINE_ID_CANDIDATES: Sequence[str] = (
    "Medicine_ID",
    "medicine_id",
    "item_id",
    "MDCODE",
)


# Actual demand date candidates.
#
# IMPORTANT:
# Historical demand data should prefer its native timestamp/date columns
# rather than Forecast_Date. Forecast_Date is kept only as a compatibility
# fallback because some already-normalized datasets may use that name.

ACTUAL_DATE_CANDIDATES: Sequence[str] = (
    "timestamp",
    "Timestamp",
    "date",
    "Date",
    "INVDT",
    "Invoice_Date",
    "invoice_date",
    "Demand_Date",
    "demand_date",
    "Daily_Demand_Date",
    "daily_demand_date",
    "Actual_Date",
    "actual_date",
    "Forecast_Date",
    "forecast_date",
)


# Forecast date candidates.
#
# Forecast_Date is the canonical production forecast field.

FORECAST_DATE_CANDIDATES: Sequence[str] = (
    "Forecast_Date",
    "forecast_date",
    "timestamp",
    "Timestamp",
    "date",
    "Date",
)


ACTUAL_QTY_CANDIDATES: Sequence[str] = (
    "Actual_Demand",
    "actual_demand",
    "Daily_Demand",
    "daily_demand",
    "Demand_Qty",
    "demand_qty",
    "target",
    "QTY",
)


PREDICTION_CANDIDATES: Sequence[str] = (
    "Predicted_Demand",
    "predicted_demand",
    "prediction",
    "forecast",
)


P10_CANDIDATES: Sequence[str] = (
    "P10",
    "p10",
    "lower",
    "lower_bound",
)


P90_CANDIDATES: Sequence[str] = (
    "P90",
    "p90",
    "upper",
    "upper_bound",
)


# ============================================================================
# Utility helpers
# ============================================================================


def _utc_now() -> datetime:
    """
    Return the current UTC timestamp.
    """

    return datetime.now(timezone.utc)


def _generate_evaluation_id() -> str:
    """
    Generate a collision-resistant evaluation identifier.
    """

    timestamp = _utc_now().strftime(
        "%Y%m%d_%H%M%S"
    )

    suffix = uuid.uuid4().hex[:8]

    return f"eval_{timestamp}_{suffix}"


def _find_column(
    dataframe: pd.DataFrame,
    candidates: Sequence[str],
    field_name: str,
    required: bool = True,
) -> Optional[str]:
    """
    Resolve a dataframe column from a list of supported candidates.
    """

    for column in candidates:
        if column in dataframe.columns:
            return column

    if required:
        raise ValueError(
            f"Unable to resolve required field "
            f"'{field_name}'. "
            f"Expected one of: {list(candidates)}. "
            f"Available columns: "
            f"{list(dataframe.columns)}"
        )

    return None


def _safe_numeric(
    series: pd.Series,
) -> pd.Series:
    """
    Convert a series to numeric values.

    Invalid values become NaN.
    """

    return pd.to_numeric(
        series,
        errors="coerce",
    )


def _date_to_string(
    value: Optional[pd.Timestamp],
) -> Optional[str]:
    """
    Convert a pandas timestamp to an ISO date string.
    """

    if value is None:
        return None

    if pd.isna(value):
        return None

    return pd.Timestamp(value).date().isoformat()


def _date_range(
    dataframe: pd.DataFrame,
    column: str,
) -> tuple[Optional[str], Optional[str]]:
    """
    Return the min/max date range for a normalized dataframe.
    """

    if dataframe.empty:
        return None, None

    if column not in dataframe.columns:
        return None, None

    valid = dataframe[column].dropna()

    if valid.empty:
        return None, None

    return (
        _date_to_string(valid.min()),
        _date_to_string(valid.max()),
    )


def _empty_overall_evaluation(
    *,
    forecast_rows: int = 0,
    actual_rows: int = 0,
    forecast_medicines: int = 0,
    actual_medicines: int = 0,
    forecast_start_date: Optional[str] = None,
    forecast_end_date: Optional[str] = None,
    actual_start_date: Optional[str] = None,
    actual_end_date: Optional[str] = None,
) -> OverallEvaluation:
    """
    Create an empty evaluation result.

    Empty evaluation means there are no matched forecast/actual rows.
    It must never be interpreted as a perfect forecast.
    """

    now = _utc_now().isoformat()

    return OverallEvaluation(
        evaluation_id=_generate_evaluation_id(),
        evaluated_at=now,

        total_medicines_evaluated=0,

        total_actual_units=0.0,
        total_predicted_units=0.0,

        overall_wape_pct=0.0,
        overall_mae=0.0,
        overall_smape_pct=0.0,
        overall_mbe=0.0,

        overall_coverage_pct=0.0,

        forecast_rows=forecast_rows,
        actual_rows=actual_rows,
        matched_rows=0,

        forecast_medicines=forecast_medicines,
        actual_medicines=actual_medicines,
        matched_medicines=0,

        forecast_start_date=forecast_start_date,
        forecast_end_date=forecast_end_date,

        actual_start_date=actual_start_date,
        actual_end_date=actual_end_date,

        overlap_start_date=None,
        overlap_end_date=None,

        has_overlap=False,

        medicines=[],
    )


# ============================================================================
# Metric calculation
# ============================================================================


def compute_metrics(
    actuals: np.ndarray,
    predictions: np.ndarray,
    p10s: Optional[np.ndarray] = None,
    p90s: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute forecast evaluation metrics.

    Metrics
    -------
    WAPE:
        sum(abs(actual - prediction)) / sum(abs(actual))

    MAE:
        mean(abs(actual - prediction))

    SMAPE:
        mean(
            200 * abs(actual - prediction)
            / (abs(actual) + abs(prediction))
        )

    MBE:
        mean(prediction - actual)

    Coverage:
        Percentage of actual observations lying within
        the [P10, P90] prediction interval.

    Invalid or non-finite observations are excluded.
    """

    actuals = np.asarray(
        actuals,
        dtype=float,
    )

    predictions = np.asarray(
        predictions,
        dtype=float,
    )

    if actuals.shape != predictions.shape:
        raise ValueError(
            "Actuals and predictions must have "
            "the same shape. "
            f"Got actuals={actuals.shape}, "
            f"predictions={predictions.shape}."
        )

    valid_mask = (
        np.isfinite(actuals)
        & np.isfinite(predictions)
    )

    actuals = actuals[valid_mask]
    predictions = predictions[valid_mask]

    if len(actuals) == 0:
        return {
            "sample_count": 0.0,

            "total_actual": 0.0,
            "total_predicted": 0.0,

            "total_absolute_error": 0.0,
            "absolute_error_sum": 0.0,

            "wape_pct": 0.0,
            "mae": 0.0,
            "smape_pct": 0.0,
            "mbe": 0.0,

            "coverage_pct": 0.0,
        }

    errors = actuals - predictions

    absolute_errors = np.abs(
        errors
    )

    total_actual = float(
        np.sum(actuals)
    )

    total_predicted = float(
        np.sum(predictions)
    )

    absolute_error_sum = float(
        np.sum(absolute_errors)
    )

    # ------------------------------------------------------------------
    # WAPE
    # ------------------------------------------------------------------

    total_actual_absolute = float(
        np.sum(np.abs(actuals))
    )

    if total_actual_absolute > 0:

        wape = (
            absolute_error_sum
            / total_actual_absolute
            * 100.0
        )

    else:

        # Explicit zero-demand behavior.
        #
        # If all actuals are zero and predictions are also zero,
        # the forecast is perfect.
        #
        # If actuals are zero but predictions are non-zero,
        # assigning 100% makes the failure visible.

        if np.allclose(
            predictions,
            0.0,
        ):
            wape = 0.0

        else:
            wape = 100.0

    # ------------------------------------------------------------------
    # MAE
    # ------------------------------------------------------------------

    mae = float(
        np.mean(
            absolute_errors
        )
    )

    # ------------------------------------------------------------------
    # SMAPE
    # ------------------------------------------------------------------

    denominator = (
        np.abs(actuals)
        + np.abs(predictions)
    )

    smape_values = np.zeros_like(
        denominator,
        dtype=float,
    )

    non_zero_mask = (
        denominator > 0
    )

    smape_values[non_zero_mask] = (
        200.0
        * absolute_errors[non_zero_mask]
        / denominator[non_zero_mask]
    )

    smape = float(
        np.mean(
            smape_values
        )
    )

    # ------------------------------------------------------------------
    # Mean Bias Error
    # ------------------------------------------------------------------

    mbe = float(
        np.mean(
            predictions - actuals
        )
    )

    # ------------------------------------------------------------------
    # Prediction interval coverage
    # ------------------------------------------------------------------

    coverage = 0.0

    if (
        p10s is not None
        and p90s is not None
    ):

        p10s = np.asarray(
            p10s,
            dtype=float,
        )

        p90s = np.asarray(
            p90s,
            dtype=float,
        )

        if (
            p10s.shape == valid_mask.shape
            and p90s.shape == valid_mask.shape
        ):

            p10s = p10s[
                valid_mask
            ]

            p90s = p90s[
                valid_mask
            ]

            interval_mask = (
                np.isfinite(p10s)
                & np.isfinite(p90s)
            )

            if np.any(
                interval_mask
            ):

                lower = np.minimum(
                    p10s[interval_mask],
                    p90s[interval_mask],
                )

                upper = np.maximum(
                    p10s[interval_mask],
                    p90s[interval_mask],
                )

                actual_subset = actuals[
                    interval_mask
                ]

                in_interval = (
                    (actual_subset >= lower)
                    & (actual_subset <= upper)
                )

                coverage = float(
                    np.mean(
                        in_interval
                    )
                    * 100.0
                )

    return {
        "sample_count": float(
            len(actuals)
        ),

        "total_actual": round(
            total_actual,
            6,
        ),

        "total_predicted": round(
            total_predicted,
            6,
        ),

        "total_absolute_error": round(
            absolute_error_sum,
            6,
        ),

        "absolute_error_sum": round(
            absolute_error_sum,
            6,
        ),

        "wape_pct": round(
            wape,
            4,
        ),

        "mae": round(
            mae,
            4,
        ),

        "smape_pct": round(
            smape,
            4,
        ),

        "mbe": round(
            mbe,
            4,
        ),

        "coverage_pct": round(
            coverage,
            4,
        ),
    }


# ============================================================================
# Forecast evaluator
# ============================================================================


class ForecastEvaluator:
    """
    Evaluate production forecasts against newly available actual demand.

    Matching is performed using:

        Medicine_ID
        Forecast_Date

    The evaluator supports the canonical production schema while
    accepting known legacy aliases.

    It explicitly records date ranges and overlap information so that
    an evaluation with zero matched rows cannot be mistaken for a
    perfect forecast.
    """

    def __init__(
        self,
        forecast_dir: Path,
    ) -> None:

        self.forecast_dir = Path(
            forecast_dir
        )

        self.evaluations_file = (
            self.forecast_dir
            / "evaluations.json"
        )

        self.history_dir = (
            self.forecast_dir
            / "history"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        actuals_df: pd.DataFrame,
        forecast_df: pd.DataFrame,
    ) -> OverallEvaluation:
        """
        Evaluate forecast rows against actual demand.

        The previous production forecast is matched only against actual
        observations whose medicine/date keys exist in that forecast.

        A result with has_overlap=False means metrics are unavailable
        because no forecast rows have corresponding actual observations.
        """

        # --------------------------------------------------------------
        # Handle completely empty inputs
        # --------------------------------------------------------------

        if actuals_df.empty:

            logger.warning(
                "Actual demand dataframe is empty. "
                "No forecast evaluation can be performed."
            )

            result = _empty_overall_evaluation(
                forecast_rows=len(forecast_df),
                actual_rows=0,
                forecast_medicines=(
                    int(
                        forecast_df[
                            _find_column(
                                forecast_df,
                                MEDICINE_ID_CANDIDATES,
                                "medicine identifier",
                            )
                        ].nunique()
                    )
                    if not forecast_df.empty
                    else 0
                ),
                actual_medicines=0,
            )

            self._save_evaluation(
                result
            )

            return result

        if forecast_df.empty:

            logger.warning(
                "Forecast dataframe is empty. "
                "No forecast evaluation can be performed."
            )

            result = _empty_overall_evaluation(
                forecast_rows=0,
                actual_rows=len(actuals_df),
                forecast_medicines=0,
                actual_medicines=(
                    int(
                        actuals_df[
                            _find_column(
                                actuals_df,
                                MEDICINE_ID_CANDIDATES,
                                "medicine identifier",
                            )
                        ].nunique()
                    )
                ),
            )

            self._save_evaluation(
                result
            )

            return result

        # --------------------------------------------------------------
        # Normalize
        # --------------------------------------------------------------

        actuals = self._normalize_actuals(
            actuals_df
        )

        forecasts = self._normalize_forecasts(
            forecast_df
        )

        # --------------------------------------------------------------
        # Build observability metadata before merging
        # --------------------------------------------------------------

        forecast_start_date, forecast_end_date = (
            _date_range(
                forecasts,
                "Forecast_Date",
            )
        )

        actual_start_date, actual_end_date = (
            _date_range(
                actuals,
                "Forecast_Date",
            )
        )

        logger.info(
            "Forecast evaluation date ranges | "
            "forecast=%s -> %s | "
            "actual=%s -> %s",
            forecast_start_date,
            forecast_end_date,
            actual_start_date,
            actual_end_date,
        )

        logger.info(
            "Forecast evaluation input summary | "
            "forecast_rows=%d | "
            "forecast_medicines=%d | "
            "actual_rows=%d | "
            "actual_medicines=%d",
            len(forecasts),
            forecasts["Medicine_ID"].nunique(),
            len(actuals),
            actuals["Medicine_ID"].nunique(),
        )

        # --------------------------------------------------------------
        # Merge
        # --------------------------------------------------------------

        merged = self._merge(
            actuals,
            forecasts,
        )

        # --------------------------------------------------------------
        # No overlap
        # --------------------------------------------------------------

        if merged.empty:

            if actual_end_date and forecast_start_date and actual_end_date < forecast_start_date:
                logger.warning(
                    "Forecast evaluation pending: actual demand for forecast window (%s to %s) is not yet available in historical records (actuals available through %s).",
                    forecast_start_date,
                    forecast_end_date,
                    actual_end_date,
                )
            else:
                logger.warning(
                    "No forecast/actual overlap found | forecast_window=%s..%s (meds=%d) | actuals_window=%s..%s (meds=%d). Metrics unavailable.",
                    forecast_start_date, forecast_end_date, forecasts["Medicine_ID"].nunique(),
                    actual_start_date, actual_end_date, actuals["Medicine_ID"].nunique(),
                )

            result = _empty_overall_evaluation(
                forecast_rows=len(forecasts),
                actual_rows=len(actuals),

                forecast_medicines=int(
                    forecasts[
                        "Medicine_ID"
                    ].nunique()
                ),

                actual_medicines=int(
                    actuals[
                        "Medicine_ID"
                    ].nunique()
                ),

                forecast_start_date=(
                    forecast_start_date
                ),

                forecast_end_date=(
                    forecast_end_date
                ),

                actual_start_date=(
                    actual_start_date
                ),

                actual_end_date=(
                    actual_end_date
                ),
            )

            self._save_evaluation(
                result
            )

            return result

        # --------------------------------------------------------------
        # Overlap exists
        # --------------------------------------------------------------

        evaluated_at = (
            _utc_now().isoformat()
        )

        overlap_start_date, overlap_end_date = (
            _date_range(
                merged,
                "Forecast_Date",
            )
        )

        medicine_evaluations = (
            self._evaluate_by_medicine(
                merged,
                evaluated_at,
            )
        )

        overall_metrics = (
            self._compute_overall_metrics(
                merged
            )
        )

        result = OverallEvaluation(
            evaluation_id=(
                _generate_evaluation_id()
            ),

            evaluated_at=evaluated_at,

            total_medicines_evaluated=(
                len(
                    medicine_evaluations
                )
            ),

            total_actual_units=(
                overall_metrics[
                    "total_actual"
                ]
            ),

            total_predicted_units=(
                overall_metrics[
                    "total_predicted"
                ]
            ),

            overall_wape_pct=(
                overall_metrics[
                    "wape_pct"
                ]
            ),

            overall_mae=(
                overall_metrics[
                    "mae"
                ]
            ),

            overall_smape_pct=(
                overall_metrics[
                    "smape_pct"
                ]
            ),

            overall_mbe=(
                overall_metrics[
                    "mbe"
                ]
            ),

            overall_coverage_pct=(
                overall_metrics[
                    "coverage_pct"
                ]
            ),

            forecast_rows=len(
                forecasts
            ),

            actual_rows=len(
                actuals
            ),

            matched_rows=len(
                merged
            ),

            forecast_medicines=int(
                forecasts[
                    "Medicine_ID"
                ].nunique()
            ),

            actual_medicines=int(
                actuals[
                    "Medicine_ID"
                ].nunique()
            ),

            matched_medicines=int(
                merged[
                    "Medicine_ID"
                ].nunique()
            ),

            forecast_start_date=(
                forecast_start_date
            ),

            forecast_end_date=(
                forecast_end_date
            ),

            actual_start_date=(
                actual_start_date
            ),

            actual_end_date=(
                actual_end_date
            ),

            overlap_start_date=(
                overlap_start_date
            ),

            overlap_end_date=(
                overlap_end_date
            ),

            has_overlap=True,

            medicines=(
                medicine_evaluations
            ),
        )

        self._save_evaluation(
            result
        )

        logger.info(
            "Forecast evaluation completed | "
            "evaluation_id=%s | "
            "matched_rows=%d | "
            "medicines=%d | "
            "WAPE=%.2f%% | "
            "SMAPE=%.2f%% | "
            "coverage=%.2f%% | "
            "overlap=%s -> %s",
            result.evaluation_id,
            result.matched_rows,
            result.total_medicines_evaluated,
            result.overall_wape_pct,
            result.overall_smape_pct,
            result.overall_coverage_pct,
            result.overlap_start_date,
            result.overlap_end_date,
        )

        return result

    # ------------------------------------------------------------------
    # Normalization: actuals
    # ------------------------------------------------------------------

    def _normalize_actuals(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Normalize actual demand data to the canonical schema.

        Canonical output:

            Medicine_ID
            Forecast_Date
            Actual_Demand

        Forecast_Date is used as the common normalized date field for
        merging, even though the source is actual historical demand.
        """

        df = dataframe.copy()

        id_column = _find_column(
            df,
            MEDICINE_ID_CANDIDATES,
            "medicine identifier",
        )

        date_column = _find_column(
            df,
            ACTUAL_DATE_CANDIDATES,
            "actual demand date",
        )

        quantity_column = _find_column(
            df,
            ACTUAL_QTY_CANDIDATES,
            "actual demand",
        )

        logger.info(
            "Actual normalization schema | "
            "medicine_id=%s | "
            "date=%s | "
            "quantity=%s",
            id_column,
            date_column,
            quantity_column,
        )

        normalized = df[
            [
                id_column,
                date_column,
                quantity_column,
            ]
        ].copy()

        normalized.columns = [
            "Medicine_ID",
            "Forecast_Date",
            "Actual_Demand",
        ]

        normalized["Medicine_ID"] = (
            normalized["Medicine_ID"]
            .astype("string")
            .str.strip()
            .str.zfill(4)
        )

        normalized["Forecast_Date"] = (
            pd.to_datetime(
                normalized[
                    "Forecast_Date"
                ],
                errors="coerce",
            )
            .dt.normalize()
        )

        normalized["Actual_Demand"] = (
            _safe_numeric(
                normalized[
                    "Actual_Demand"
                ]
            )
        )

        before = len(
            normalized
        )

        normalized = normalized.dropna(
            subset=[
                "Medicine_ID",
                "Forecast_Date",
                "Actual_Demand",
            ]
        )

        normalized = normalized[
            normalized["Medicine_ID"].ne("")
        ]

        dropped = (
            before
            - len(normalized)
        )

        if dropped > 0:

            logger.warning(
                "Dropped %d invalid actual demand rows "
                "during evaluation normalization.",
                dropped,
            )

        # Historical demand can contain multiple rows for the same
        # medicine/day. These must be aggregated before a one-to-one
        # evaluation merge.

        normalized = (
            normalized
            .groupby(
                [
                    "Medicine_ID",
                    "Forecast_Date",
                ],
                as_index=False,
                dropna=False,
            )[
                "Actual_Demand"
            ]
            .sum()
        )

        return normalized

    # ------------------------------------------------------------------
    # Normalization: forecasts
    # ------------------------------------------------------------------

    def _normalize_forecasts(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Normalize forecast data to the canonical schema.

        Canonical output:

            Medicine_ID
            Forecast_Date
            Predicted_Demand
            P10 (optional)
            P90 (optional)
        """

        df = dataframe.copy()

        id_column = _find_column(
            df,
            MEDICINE_ID_CANDIDATES,
            "medicine identifier",
        )

        date_column = _find_column(
            df,
            FORECAST_DATE_CANDIDATES,
            "forecast date",
        )

        prediction_column = _find_column(
            df,
            PREDICTION_CANDIDATES,
            "predicted demand",
        )

        p10_column = _find_column(
            df,
            P10_CANDIDATES,
            "P10 interval",
            required=False,
        )

        p90_column = _find_column(
            df,
            P90_CANDIDATES,
            "P90 interval",
            required=False,
        )

        logger.info(
            "Forecast normalization schema | "
            "medicine_id=%s | "
            "date=%s | "
            "prediction=%s | "
            "P10=%s | "
            "P90=%s",
            id_column,
            date_column,
            prediction_column,
            p10_column,
            p90_column,
        )

        columns = [
            id_column,
            date_column,
            prediction_column,
        ]

        if p10_column is not None:

            columns.append(
                p10_column
            )

        if p90_column is not None:

            columns.append(
                p90_column
            )

        normalized = df[
            columns
        ].copy()

        rename_map = {
            id_column: "Medicine_ID",
            date_column: "Forecast_Date",
            prediction_column: (
                "Predicted_Demand"
            ),
        }

        if p10_column is not None:

            rename_map[
                p10_column
            ] = "P10"

        if p90_column is not None:

            rename_map[
                p90_column
            ] = "P90"

        normalized = normalized.rename(
            columns=rename_map
        )

        normalized["Medicine_ID"] = (
            normalized["Medicine_ID"]
            .astype("string")
            .str.strip()
            .str.zfill(4)
        )

        normalized["Forecast_Date"] = (
            pd.to_datetime(
                normalized[
                    "Forecast_Date"
                ],
                errors="coerce",
            )
            .dt.normalize()
        )

        normalized[
            "Predicted_Demand"
        ] = _safe_numeric(
            normalized[
                "Predicted_Demand"
            ]
        )

        if "P10" in normalized.columns:

            normalized["P10"] = (
                _safe_numeric(
                    normalized["P10"]
                )
            )

        if "P90" in normalized.columns:

            normalized["P90"] = (
                _safe_numeric(
                    normalized["P90"]
                )
            )

        before = len(
            normalized
        )

        normalized = normalized.dropna(
            subset=[
                "Medicine_ID",
                "Forecast_Date",
                "Predicted_Demand",
            ]
        )

        normalized = normalized[
            normalized["Medicine_ID"].ne("")
        ]

        dropped = (
            before
            - len(normalized)
        )

        if dropped > 0:

            logger.warning(
                "Dropped %d invalid forecast rows "
                "during evaluation normalization.",
                dropped,
            )

        # A production forecast should contain one row per medicine/date.
        # If duplicates exist, retain the final generated row.

        duplicate_mask = (
            normalized.duplicated(
                subset=[
                    "Medicine_ID",
                    "Forecast_Date",
                ],
                keep=False,
            )
        )

        if duplicate_mask.any():

            duplicate_count = int(
                duplicate_mask.sum()
            )

            logger.warning(
                "Detected %d duplicate forecast rows. "
                "Keeping the last row for each "
                "Medicine_ID / Forecast_Date pair.",
                duplicate_count,
            )

            normalized = (
                normalized
                .drop_duplicates(
                    subset=[
                        "Medicine_ID",
                        "Forecast_Date",
                    ],
                    keep="last",
                )
            )

        return normalized

    # ------------------------------------------------------------------
    # Merge
    # ------------------------------------------------------------------

    def _merge(
        self,
        actuals: pd.DataFrame,
        forecasts: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Merge normalized forecasts with normalized actuals.

        Both inputs are guaranteed to have one row per:

            Medicine_ID
            Forecast_Date
        """

        merged = pd.merge(
            forecasts,
            actuals,
            on=[
                "Medicine_ID",
                "Forecast_Date",
            ],
            how="inner",
            validate="one_to_one",
        )

        logger.info(
            "Forecast evaluation overlap | "
            "forecast_rows=%d | "
            "actual_rows=%d | "
            "matched_rows=%d | "
            "matched_medicines=%d",
            len(forecasts),
            len(actuals),
            len(merged),
            (
                merged[
                    "Medicine_ID"
                ].nunique()
                if not merged.empty
                else 0
            ),
        )

        return merged

    # ------------------------------------------------------------------
    # Per-medicine evaluation
    # ------------------------------------------------------------------

    def _evaluate_by_medicine(
        self,
        merged: pd.DataFrame,
        evaluated_at: str,
    ) -> List[MedicineEvaluation]:
        """
        Calculate metrics for every medicine.
        """

        evaluations: List[
            MedicineEvaluation
        ] = []

        for medicine_id, group in (
            merged.groupby(
                "Medicine_ID",
                sort=True,
            )
        ):

            p10s = None
            p90s = None

            if "P10" in group.columns:

                p10s = (
                    group["P10"]
                    .to_numpy(
                        dtype=float
                    )
                )

            if "P90" in group.columns:

                p90s = (
                    group["P90"]
                    .to_numpy(
                        dtype=float
                    )
                )

            metrics = compute_metrics(
                actuals=(
                    group[
                        "Actual_Demand"
                    ]
                    .to_numpy(
                        dtype=float
                    )
                ),

                predictions=(
                    group[
                        "Predicted_Demand"
                    ]
                    .to_numpy(
                        dtype=float
                    )
                ),

                p10s=p10s,
                p90s=p90s,
            )

            evaluations.append(
                MedicineEvaluation(
                    medicine_id=str(
                        medicine_id
                    ),

                    sample_count=int(
                        metrics[
                            "sample_count"
                        ]
                    ),

                    total_actual_qty=(
                        metrics[
                            "total_actual"
                        ]
                    ),

                    total_predicted_qty=(
                        metrics[
                            "total_predicted"
                        ]
                    ),

                    wape_pct=(
                        metrics[
                            "wape_pct"
                        ]
                    ),

                    mae=(
                        metrics[
                            "mae"
                        ]
                    ),

                    smape_pct=(
                        metrics[
                            "smape_pct"
                        ]
                    ),

                    mbe=(
                        metrics[
                            "mbe"
                        ]
                    ),

                    p10_p90_coverage_pct=(
                        metrics[
                            "coverage_pct"
                        ]
                    ),

                    evaluated_at=(
                        evaluated_at
                    ),
                )
            )

        return evaluations

    # ------------------------------------------------------------------
    # Overall evaluation
    # ------------------------------------------------------------------

    def _compute_overall_metrics(
        self,
        merged: pd.DataFrame,
    ) -> Dict[str, float]:
        """
        Calculate aggregate metrics across all matched rows.
        """

        p10s = None
        p90s = None

        if "P10" in merged.columns:

            p10s = (
                merged["P10"]
                .to_numpy(
                    dtype=float
                )
            )

        if "P90" in merged.columns:

            p90s = (
                merged["P90"]
                .to_numpy(
                    dtype=float
                )
            )

        return compute_metrics(
            actuals=(
                merged[
                    "Actual_Demand"
                ]
                .to_numpy(
                    dtype=float
                )
            ),

            predictions=(
                merged[
                    "Predicted_Demand"
                ]
                .to_numpy(
                    dtype=float
                )
            ),

            p10s=p10s,
            p90s=p90s,
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save_evaluation(
        self,
        evaluation: OverallEvaluation,
    ) -> None:
        """
        Persist:

        1. Latest evaluations.json
        2. Immutable historical evaluation snapshot

        Writes are atomic.
        """

        self.forecast_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.history_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        payload = (
            evaluation.model_dump(
                mode="json"
            )
        )

        serialized = json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
        )

        # Latest evaluation.

        self._atomic_write_text(
            self.evaluations_file,
            serialized,
        )

        # Immutable historical evaluation.

        history_file = (
            self.history_dir
            / f"{evaluation.evaluation_id}.json"
        )

        self._atomic_write_text(
            history_file,
            serialized,
        )

        logger.info(
            "Evaluation saved | "
            "latest=%s | "
            "history=%s",
            self.evaluations_file,
            history_file,
        )

    @staticmethod
    def _atomic_write_text(
        path: Path,
        content: str,
    ) -> None:
        """
        Atomically write text to disk.

        The temporary file is replaced only after the
        content has been successfully written.
        """

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        temporary_path = (
            path.with_suffix(
                path.suffix
                + ".tmp"
            )
        )

        try:

            temporary_path.write_text(
                content,
                encoding="utf-8",
            )

            os.replace(
                temporary_path,
                path,
            )

        finally:

            if temporary_path.exists():

                try:

                    temporary_path.unlink()

                except OSError:

                    logger.warning(
                        "Failed to remove temporary "
                        "evaluation file: %s",
                        temporary_path,
                    )

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_latest_evaluation(
        self,
    ) -> Optional[OverallEvaluation]:
        """
        Load the latest evaluation report.
        """

        if not (
            self.evaluations_file.exists()
        ):

            logger.info(
                "No evaluation report found at %s",
                self.evaluations_file,
            )

            return None

        try:

            payload = json.loads(
                self.evaluations_file.read_text(
                    encoding="utf-8"
                )
            )

            return OverallEvaluation(
                **payload
            )

        except Exception:

            logger.exception(
                "Failed to load evaluation "
                "from %s",
                self.evaluations_file,
            )

            return None