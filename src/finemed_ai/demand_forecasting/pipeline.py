from __future__ import annotations

"""
Finemed PharmaAI — Production Demand Forecasting Pipeline.

Production policy lock
----------------------

1. Routing uses validation evidence only.
2. Holdout results never influence production routing.
3. Chronos validation advantage >= 30% -> Chronos-2 P50.
4. Otherwise -> TSB.
5. Medicines without validation evidence -> deterministic TSB fallback.
6. ACTIVE and STALE medicines enter model forecasting.
7. DORMANT medicines do not enter model inference.
8. DORMANT medicines receive deterministic zero-demand records.
9. Every production run is written to an immutable versioned directory.
10. latest.parquet is promoted only when the complete batch succeeds.
11. The previous published latest.parquet is never modified by a failed run.
"""

import json
import logging
import os
import uuid

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from pydantic import BaseModel

from finemed_ai.demand_forecasting.config import (
    DEFAULT_CONFIG,
    ForecastConfig,
)

from finemed_ai.demand_forecasting.forecast_eligibility import (
    build_eligibility_table,
)

from finemed_ai.demand_forecasting.predictor_service import (
    PredictorService,
)

from finemed_ai.demand_forecasting.production_forecast_router import (
    ProductionForecastRouter,
)

from finemed_ai.demand_forecasting.production_forecast_service import (
    ProductionForecastService,
)


logger = logging.getLogger(__name__)


# ============================================================================
# Constants
# ============================================================================

MEDICINE_ID_COLUMN = "Medicine_ID"
TIMESTAMP_COLUMN = "timestamp"
TARGET_COLUMN = "target"

FORECAST_DATE_COLUMN = "Forecast_Date"
PREDICTION_COLUMN = "Predicted_Demand"
MODEL_COLUMN = "Selected_Model"

ROUTING_RESULTS_PATH = Path(
    "data/05_gold/demand_forecasting/model_routing/"
    "model_routing_results.parquet"
)

ROUTING_TABLE_PATH = Path(
    "data/05_gold/demand_forecasting/medicine_robustness/"
    "production_routing_table.parquet"
)

MIN_SUCCESS_RATE = 1.00
ROUTING_THRESHOLD = 30.0

LOCKED_VALIDATION_CUTOFFS = (
    pd.Timestamp("2025-11-01"),
    pd.Timestamp("2025-12-01"),
    pd.Timestamp("2026-01-01"),
    pd.Timestamp("2026-02-01"),
)

ALLOWED_MODELS = {
    "chronos-2-P50",
    "tsb",
}

FORECAST_STATUS_FORECASTED = "FORECASTED"

FORECAST_STATUS_FORECASTED_STALE = (
    "FORECASTED_STALE"
)

FORECAST_STATUS_NOT_FORECASTED = (
    "NOT_FORECASTED"
)

DORMANT_MODEL_NAME = "dormant_policy"

DORMANT_ROUTING_REASON = (
    "dormant_medicine_no_recent_demand"
)

ELIGIBILITY_ACTIVE = "ACTIVE"
ELIGIBILITY_STALE = "STALE"
ELIGIBILITY_DORMANT = "DORMANT"

FORECASTABLE_ELIGIBILITY_STATUSES = {
    ELIGIBILITY_ACTIVE,
    ELIGIBILITY_STALE,
}


REQUIRED_ROUTING_OUTPUT_COLUMNS = {
    MEDICINE_ID_COLUMN,
    "Chronos_AE",
    "TSB_AE",
    "Validation_Windows",
    "Validation_Advantage_Pct",
    "Selected_Model",
    "Routing_Rule",
    "Routing_Reason",
}


REQUIRED_FORECAST_COLUMNS = {
    MEDICINE_ID_COLUMN,
    FORECAST_DATE_COLUMN,
    PREDICTION_COLUMN,
    MODEL_COLUMN,
}


# ============================================================================
# Run manifest schema
# ============================================================================


class BatchForecastRunResult(BaseModel):
    """
    Immutable metadata describing one production forecasting run.

    output_path always points to the versioned forecast artifact.

    published indicates whether that version was promoted to
    latest.parquet.
    """

    run_id: str

    started_at: datetime
    completed_at: datetime

    medicines_requested: int
    medicines_succeeded: int
    medicines_failed: int

    failed_medicine_ids: list[str]

    output_path: str

    published: bool
    publish_note: str


# ============================================================================
# Internal helpers
# ============================================================================


def _normalize_medicine_ids(
    values: pd.Series,
    *,
    context: str,
) -> pd.Series:
    """
    Normalize Medicine_ID values consistently.
    """

    if not isinstance(
        values,
        pd.Series,
    ):
        raise TypeError(
            f"{context} Medicine_ID values must be a pandas Series."
        )

    if values.isna().any():
        raise ValueError(
            f"{context} contains NULL Medicine_ID values."
        )

    normalized = (
        values
        .astype(str)
        .str.strip()
    )

    if normalized.eq("").any():
        raise ValueError(
            f"{context} contains empty Medicine_ID values."
        )

    return normalized


def _validate_finite_non_negative(
    series: pd.Series,
    *,
    column_name: str,
    context: str,
) -> pd.Series:
    """
    Convert a numeric series and validate that all values are finite
    and non-negative.
    """

    numeric = pd.to_numeric(
        series,
        errors="coerce",
    )

    if numeric.isna().any():
        raise ValueError(
            f"{context} contains NULL/non-numeric values in "
            f"{column_name}."
        )

    values = numeric.to_numpy(
        dtype=float,
    )

    if not np.isfinite(values).all():
        raise ValueError(
            f"{context} contains non-finite values in "
            f"{column_name}."
        )

    if (numeric < 0).any():
        raise ValueError(
            f"{context} contains negative values in "
            f"{column_name}."
        )

    return numeric.astype(float)


def _atomic_replace(
    source: Path,
    destination: Path,
) -> None:
    """
    Atomically replace destination with source.

    source and destination must be on the same filesystem.
    """

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    os.replace(
        source,
        destination,
    )


def _atomic_write_parquet(
    dataframe: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Write parquet through a temporary file and atomically promote it.
    """

    if not isinstance(
        dataframe,
        pd.DataFrame,
    ):
        raise TypeError(
            "dataframe must be a pandas DataFrame."
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = output_path.with_name(
        f".{output_path.name}.{uuid.uuid4().hex}.tmp"
    )

    try:

        dataframe.to_parquet(
            temporary_path,
            index=False,
        )

        _atomic_replace(
            temporary_path,
            output_path,
        )

    finally:

        if temporary_path.exists():

            temporary_path.unlink()


def _atomic_write_json(
    payload: dict[str, Any],
    output_path: Path,
) -> None:
    """
    Atomically write JSON metadata.
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = output_path.with_name(
        f".{output_path.name}.{uuid.uuid4().hex}.tmp"
    )

    try:

        temporary_path.write_text(
            json.dumps(
                payload,
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )

        _atomic_replace(
            temporary_path,
            output_path,
        )

    finally:

        if temporary_path.exists():

            temporary_path.unlink()


# ============================================================================
# Routing table builder
# ============================================================================


def build_routing_table(
    routing_results: pd.DataFrame,
    all_medicine_ids: list[str],
) -> pd.DataFrame:
    """
    Build deterministic production routing decisions.

    Chronos advantage:

        (TSB_AE - Chronos_AE) / TSB_AE * 100

    Policy:

        Validation advantage >= ROUTING_THRESHOLD
            -> chronos-2-P50

        Otherwise
            -> tsb

        No validation evidence
            -> tsb fallback
    """

    if not isinstance(
        routing_results,
        pd.DataFrame,
    ):
        raise TypeError(
            "routing_results must be a pandas DataFrame."
        )

    if not isinstance(
        all_medicine_ids,
        list,
    ):
        raise TypeError(
            "all_medicine_ids must be a list."
        )

    if routing_results.empty:
        raise ValueError(
            "Routing results dataset is empty."
        )

    if not all_medicine_ids:
        raise ValueError(
            "all_medicine_ids must not be empty."
        )

    required_columns = {
        MEDICINE_ID_COLUMN,
        "Chronos_AE",
        "TSB_AE",
        "Cutoff_Date",
    }

    missing_columns = (
        required_columns
        - set(routing_results.columns)
    )

    if missing_columns:
        raise ValueError(
            "Routing results dataset is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    production_ids = _normalize_medicine_ids(
        pd.Series(
            all_medicine_ids,
            dtype="object",
        ),
        context="Production medicine IDs",
    )

    production_ids = (
        production_ids
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    if not production_ids:
        raise ValueError(
            "No valid production medicine IDs remain after normalization."
        )

    routing = routing_results.copy()

    routing[MEDICINE_ID_COLUMN] = (
        _normalize_medicine_ids(
            routing[MEDICINE_ID_COLUMN],
            context="Routing results",
        )
    )

    routing["Chronos_AE"] = (
        _validate_finite_non_negative(
            routing["Chronos_AE"],
            column_name="Chronos_AE",
            context="Routing results",
        )
    )

    routing["TSB_AE"] = (
        _validate_finite_non_negative(
            routing["TSB_AE"],
            column_name="TSB_AE",
            context="Routing results",
        )
    )

    routing["Cutoff_Date"] = pd.to_datetime(
        routing["Cutoff_Date"],
        errors="coerce",
    )

    if routing["Cutoff_Date"].isna().any():
        raise ValueError(
            "Routing results contain invalid Cutoff_Date values."
        )

    actual_cutoffs = set(
        routing["Cutoff_Date"].drop_duplicates()
    )

    expected_cutoffs = set(
        LOCKED_VALIDATION_CUTOFFS
    )

    unexpected_cutoffs = (
        actual_cutoffs
        - expected_cutoffs
    )

    if unexpected_cutoffs:
        raise ValueError(
            "Routing results contain cutoffs outside the locked "
            "validation set: "
            f"{sorted(str(x.date()) for x in unexpected_cutoffs)}"
        )

    missing_cutoffs = (
        expected_cutoffs
        - actual_cutoffs
    )

    if missing_cutoffs:
        raise ValueError(
            "Routing results are missing required locked "
            "validation cutoffs: "
            f"{sorted(str(x.date()) for x in missing_cutoffs)}"
        )

    grouped = (
        routing
        .groupby(
            MEDICINE_ID_COLUMN,
            as_index=False,
        )
        .agg(
            Chronos_AE=(
                "Chronos_AE",
                "sum",
            ),
            TSB_AE=(
                "TSB_AE",
                "sum",
            ),
            Validation_Windows=(
                "Cutoff_Date",
                "nunique",
            ),
        )
    )

    grouped["Validation_Advantage_Pct"] = (
        np.where(
            grouped["TSB_AE"] > 0,
            (
                (
                    grouped["TSB_AE"]
                    - grouped["Chronos_AE"]
                )
                / grouped["TSB_AE"]
            )
            * 100.0,
            0.0,
        )
    )

    if not np.isfinite(
        grouped[
            "Validation_Advantage_Pct"
        ].to_numpy(
            dtype=float
        )
    ).all():
        raise RuntimeError(
            "Routing calculation produced non-finite validation advantages."
        )

    chronos_mask = (
        grouped[
            "Validation_Advantage_Pct"
        ]
        >= ROUTING_THRESHOLD
    )

    grouped["Selected_Model"] = np.where(
        chronos_mask,
        "chronos-2-P50",
        "tsb",
    )

    grouped["Routing_Rule"] = np.where(
        chronos_mask,
        (
            "Chronos-2 selected because validation advantage "
            f">= {ROUTING_THRESHOLD:.1f}%."
        ),
        (
            "TSB selected because Chronos-2 validation advantage "
            f"< {ROUTING_THRESHOLD:.1f}%."
        ),
    )

    grouped["Routing_Reason"] = np.where(
        chronos_mask,
        (
            "chronos_validation_advantage_"
            f"meets_{ROUTING_THRESHOLD:.0f}pct_threshold"
        ),
        (
            "chronos_validation_advantage_"
            f"below_{ROUTING_THRESHOLD:.0f}pct_threshold"
        ),
    )

    production_df = pd.DataFrame(
        {
            MEDICINE_ID_COLUMN: production_ids,
        }
    )

    result = production_df.merge(
        grouped,
        on=MEDICINE_ID_COLUMN,
        how="left",
        validate="one_to_one",
    )

    no_validation_mask = (
        result["Selected_Model"].isna()
    )

    result.loc[
        no_validation_mask,
        "Selected_Model",
    ] = "tsb"

    result.loc[
        no_validation_mask,
        "Routing_Rule",
    ] = (
        "fallback_to_tsb_when_no_validation_evidence"
    )

    result.loc[
        no_validation_mask,
        "Routing_Reason",
    ] = (
        "no_validation_evidence_deterministic_tsb_fallback"
    )

    result["Chronos_AE"] = (
        pd.to_numeric(
            result["Chronos_AE"],
            errors="coerce",
        )
        .fillna(0.0)
        .astype(float)
    )

    result["TSB_AE"] = (
        pd.to_numeric(
            result["TSB_AE"],
            errors="coerce",
        )
        .fillna(0.0)
        .astype(float)
    )

    result["Validation_Windows"] = (
        pd.to_numeric(
            result["Validation_Windows"],
            errors="coerce",
        )
        .fillna(0)
        .astype(int)
    )

    result["Validation_Advantage_Pct"] = (
        pd.to_numeric(
            result["Validation_Advantage_Pct"],
            errors="coerce",
        )
        .fillna(0.0)
        .astype(float)
    )

    missing_output_columns = (
        REQUIRED_ROUTING_OUTPUT_COLUMNS
        - set(result.columns)
    )

    if missing_output_columns:
        raise RuntimeError(
            "Built routing table is missing required columns: "
            f"{sorted(missing_output_columns)}"
        )

    if result[MEDICINE_ID_COLUMN].duplicated().any():
        raise RuntimeError(
            "Production routing table contains duplicate Medicine_ID values."
        )

    invalid_models = (
        set(
            result[
                "Selected_Model"
            ]
            .astype(str)
            .str.strip()
        )
        - ALLOWED_MODELS
    )

    if invalid_models:
        raise RuntimeError(
            "Invalid production routing models: "
            f"{sorted(invalid_models)}"
        )

    return (
        result[
            [
                MEDICINE_ID_COLUMN,
                "Chronos_AE",
                "TSB_AE",
                "Validation_Windows",
                "Validation_Advantage_Pct",
                "Selected_Model",
                "Routing_Rule",
                "Routing_Reason",
            ]
        ]
        .sort_values(
            MEDICINE_ID_COLUMN,
        )
        .reset_index(
            drop=True,
        )
    )


# ============================================================================
# Production input loading
# ============================================================================


def _load_forecasting_series(
    forecasting_series_path: Path,
) -> pd.DataFrame:
    """
    Load and validate the production forecasting dataset.

    Supported schemas:

    1.
        Medicine_ID
        timestamp
        target

    2.
        MDCODE
        INVDT
        Demand_Qty
    """

    if not forecasting_series_path.exists():
        raise FileNotFoundError(
            "Forecasting input file does not exist: "
            f"{forecasting_series_path}"
        )

    if not forecasting_series_path.is_file():
        raise ValueError(
            "Forecasting input path is not a file: "
            f"{forecasting_series_path}"
        )

    dataframe = pd.read_parquet(
        forecasting_series_path,
    )

    if dataframe.empty:
        raise ValueError(
            "Forecasting input dataset is empty."
        )

    canonical_schema = {
        MEDICINE_ID_COLUMN,
        TIMESTAMP_COLUMN,
        TARGET_COLUMN,
    }

    silver_schema = {
        "MDCODE",
        "INVDT",
        "Demand_Qty",
    }

    if canonical_schema.issubset(
        dataframe.columns
    ):

        result = dataframe[
            [
                MEDICINE_ID_COLUMN,
                TIMESTAMP_COLUMN,
                TARGET_COLUMN,
            ]
        ].copy()

    elif silver_schema.issubset(
        dataframe.columns
    ):

        result = (
            dataframe
            .rename(
                columns={
                    "MDCODE": MEDICINE_ID_COLUMN,
                    "INVDT": TIMESTAMP_COLUMN,
                    "Demand_Qty": TARGET_COLUMN,
                }
            )[
                [
                    MEDICINE_ID_COLUMN,
                    TIMESTAMP_COLUMN,
                    TARGET_COLUMN,
                ]
            ]
            .copy()
        )

    else:
        raise ValueError(
            "Forecasting input has unsupported schema. "
            "Expected either "
            "[Medicine_ID, timestamp, target] "
            "or "
            "[MDCODE, INVDT, Demand_Qty]. "
            f"Found: {list(dataframe.columns)}"
        )

    result[MEDICINE_ID_COLUMN] = (
        _normalize_medicine_ids(
            result[MEDICINE_ID_COLUMN],
            context="Forecasting input",
        )
    )

    result[TIMESTAMP_COLUMN] = pd.to_datetime(
        result[TIMESTAMP_COLUMN],
        errors="coerce",
    )

    if result[TIMESTAMP_COLUMN].isna().any():
        raise ValueError(
            "Forecasting input contains invalid timestamps."
        )

    result[TARGET_COLUMN] = (
        _validate_finite_non_negative(
            result[TARGET_COLUMN],
            column_name=TARGET_COLUMN,
            context="Forecasting input",
        )
    )

    return (
        result
        .sort_values(
            [
                MEDICINE_ID_COLUMN,
                TIMESTAMP_COLUMN,
            ]
        )
        .reset_index(
            drop=True,
        )
    )


# ============================================================================
# Routing evidence loading
# ============================================================================


def _load_routing_results() -> pd.DataFrame:

    if not ROUTING_RESULTS_PATH.exists():
        raise FileNotFoundError(
            "Production routing results not found: "
            f"{ROUTING_RESULTS_PATH}"
        )

    dataframe = pd.read_parquet(
        ROUTING_RESULTS_PATH,
    )

    if dataframe.empty:
        raise ValueError(
            "Production routing results are empty."
        )

    required_columns = {
        MEDICINE_ID_COLUMN,
        "Chronos_AE",
        "TSB_AE",
        "Cutoff_Date",
    }

    missing_columns = (
        required_columns
        - set(dataframe.columns)
    )

    if missing_columns:
        raise ValueError(
            "Production routing results are missing required columns: "
            f"{sorted(missing_columns)}"
        )

    dataframe = dataframe.copy()

    dataframe["Cutoff_Date"] = pd.to_datetime(
        dataframe["Cutoff_Date"],
        errors="coerce",
    )

    if dataframe["Cutoff_Date"].isna().any():
        raise ValueError(
            "Production routing results contain invalid Cutoff_Date values."
        )

    dataframe = dataframe[
        dataframe[
            "Cutoff_Date"
        ].isin(
            LOCKED_VALIDATION_CUTOFFS
        )
    ].copy()

    if dataframe.empty:
        raise ValueError(
            "No routing evidence remains after filtering to "
            "the locked validation cutoffs."
        )

    actual_cutoffs = set(
        dataframe[
            "Cutoff_Date"
        ].drop_duplicates()
    )

    expected_cutoffs = set(
        LOCKED_VALIDATION_CUTOFFS
    )

    missing_cutoffs = (
        expected_cutoffs
        - actual_cutoffs
    )

    if missing_cutoffs:
        raise ValueError(
            "Production routing evidence is missing locked "
            "validation cutoffs: "
            f"{sorted(str(x.date()) for x in missing_cutoffs)}"
        )

    logger.info(
        "Locked validation routing evidence loaded | rows=%d | cutoffs=%s",
        len(dataframe),
        [
            cutoff.strftime(
                "%Y-%m-%d"
            )
            for cutoff in LOCKED_VALIDATION_CUTOFFS
        ],
    )

    return (
        dataframe
        .sort_values(
            [
                MEDICINE_ID_COLUMN,
                "Cutoff_Date",
            ]
        )
        .reset_index(
            drop=True,
        )
    )


# ============================================================================
# Routing artifact persistence
# ============================================================================


def _persist_routing_table(
    routing_table: pd.DataFrame,
) -> None:
    """
    Persist the exact production routing decisions used by the pipeline.
    """

    if not isinstance(
        routing_table,
        pd.DataFrame,
    ):
        raise TypeError(
            "routing_table must be a pandas DataFrame."
        )

    missing_columns = (
        REQUIRED_ROUTING_OUTPUT_COLUMNS
        - set(routing_table.columns)
    )

    if missing_columns:
        raise RuntimeError(
            "Cannot persist invalid routing table. Missing columns: "
            f"{sorted(missing_columns)}"
        )

    _atomic_write_parquet(
        routing_table,
        ROUTING_TABLE_PATH,
    )

    logger.info(
        "Production routing table written: %s",
        ROUTING_TABLE_PATH,
    )


# ============================================================================
# Dormant forecast generation
# ============================================================================


def _build_dormant_forecast_records(
    dormant_eligibility: pd.DataFrame,
    forecast_start_date: pd.Timestamp,
    config: ForecastConfig,
    generated_at: datetime,
) -> pd.DataFrame:
    """
    Build deterministic zero-demand forecast records for dormant medicines.

    Dormant medicines do not enter model routing or model inference.
    They remain in the published artifact for population completeness and
    auditability, but are explicitly marked as NOT_FORECASTED.
    """

    required_columns = {
        MEDICINE_ID_COLUMN,
        "Gap_Days",
        "Eligibility_Status",
    }

    missing_columns = (
        required_columns
        - set(dormant_eligibility.columns)
    )

    if missing_columns:
        raise ValueError(
            "Dormant eligibility table is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if dormant_eligibility.empty:
        return pd.DataFrame()

    if int(config.prediction_length) <= 0:
        raise ValueError(
            "prediction_length must be positive."
        )

    forecast_dates = pd.date_range(
        start=pd.Timestamp(
            forecast_start_date
        ),
        periods=int(
            config.prediction_length
        ),
        freq="D",
    )

    rows: list[dict[str, Any]] = []

    for record in dormant_eligibility.itertuples(
        index=False,
    ):

        medicine_id = str(
            getattr(
                record,
                MEDICINE_ID_COLUMN,
            )
        ).strip()

        gap_days = int(
            getattr(
                record,
                "Gap_Days",
            )
        )

        for forecast_date in forecast_dates:

            rows.append(
                {
                    MEDICINE_ID_COLUMN: medicine_id,
                    FORECAST_DATE_COLUMN: forecast_date,
                    MODEL_COLUMN: DORMANT_MODEL_NAME,
                    PREDICTION_COLUMN: 0.0,
                    "P10": 0.0,
                    "P50": 0.0,
                    "P90": 0.0,
                    "Validation_Advantage_Pct": 0.0,
                    "Routing_Reason": DORMANT_ROUTING_REASON,
                    "Context_Length_Used": 0,
                    "Prediction_Length": int(
                        config.prediction_length
                    ),
                    "Generated_At": generated_at,
                    "Eligibility_Status": ELIGIBILITY_DORMANT,
                    "Gap_Days": gap_days,
                    "Forecast_Status": (
                        FORECAST_STATUS_NOT_FORECASTED
                    ),
                }
            )

    return pd.DataFrame(
        rows,
    )


# ============================================================================
# Forecast output normalization
# ============================================================================


def _attach_eligibility_metadata(
    forecast_df: pd.DataFrame,
    eligibility_table: pd.DataFrame,
    generated_at: datetime,
) -> pd.DataFrame:
    """
    Attach eligibility status and gap days to model forecast output.

    This function does not modify model predictions.
    """

    if forecast_df.empty:
        return forecast_df.copy()

    required_eligibility_columns = {
        MEDICINE_ID_COLUMN,
        "Gap_Days",
        "Eligibility_Status",
    }

    missing_columns = (
        required_eligibility_columns
        - set(eligibility_table.columns)
    )

    if missing_columns:
        raise ValueError(
            "Eligibility table missing required columns: "
            f"{sorted(missing_columns)}"
        )

    result = forecast_df.copy()

    eligibility_metadata = (
        eligibility_table[
            [
                MEDICINE_ID_COLUMN,
                "Gap_Days",
                "Eligibility_Status",
            ]
        ]
        .copy()
        .drop_duplicates(
            subset=[
                MEDICINE_ID_COLUMN
            ]
        )
    )

    result = result.merge(
        eligibility_metadata,
        on=MEDICINE_ID_COLUMN,
        how="left",
        validate="many_to_one",
    )

    if result[
        "Eligibility_Status"
    ].isna().any():
        raise RuntimeError(
            "Model forecast output contains medicines missing "
            "eligibility metadata."
        )

    result["Forecast_Status"] = np.where(
        result[
            "Eligibility_Status"
        ]
        == ELIGIBILITY_STALE,
        FORECAST_STATUS_FORECASTED_STALE,
        FORECAST_STATUS_FORECASTED,
    )

    if "Generated_At" not in result.columns:
        result[
            "Generated_At"
        ] = generated_at

    if "Prediction_Length" not in result.columns:
        result[
            "Prediction_Length"
        ] = int(
            len(
                result[
                    FORECAST_DATE_COLUMN
                ].unique()
            )
        )

    return result


# ============================================================================
# Final forecast validation
# ============================================================================


def _validate_forecast_output(
    forecast_df: pd.DataFrame,
    *,
    expected_medicine_ids: set[str],
    forecastable_medicine_ids: set[str],
    dormant_medicine_ids: set[str],
    config: ForecastConfig,
) -> None:
    """
    Validate the complete production forecast before publication.

    Final output must contain:

        ACTIVE + STALE medicines:
            exactly prediction_length rows each

        DORMANT medicines:
            exactly prediction_length deterministic zero-demand rows each
    """

    if not isinstance(
        forecast_df,
        pd.DataFrame,
    ):
        raise TypeError(
            "Production forecast output must be a pandas DataFrame."
        )

    if forecast_df.empty:
        raise RuntimeError(
            "Production forecast output is empty."
        )

    if expected_medicine_ids != (
        forecastable_medicine_ids
        | dormant_medicine_ids
    ):
        raise RuntimeError(
            "Expected medicine population does not match "
            "forecastable + dormant populations."
        )

    if (
        forecastable_medicine_ids
        & dormant_medicine_ids
    ):
        raise RuntimeError(
            "Forecastable and dormant medicine populations overlap."
        )

    missing_columns = (
        REQUIRED_FORECAST_COLUMNS
        - set(forecast_df.columns)
    )

    if missing_columns:
        raise RuntimeError(
            "Forecast output missing required columns: "
            f"{sorted(missing_columns)}"
        )

    normalized_ids = _normalize_medicine_ids(
        forecast_df[
            MEDICINE_ID_COLUMN
        ],
        context="Forecast output",
    )

    output_ids = set(
        normalized_ids.tolist()
    )

    missing_ids = (
        expected_medicine_ids
        - output_ids
    )

    if missing_ids:
        raise RuntimeError(
            "Forecast output missing medicines: "
            f"{sorted(missing_ids)[:20]}"
        )

    unexpected_ids = (
        output_ids
        - expected_medicine_ids
    )

    if unexpected_ids:
        raise RuntimeError(
            "Forecast output contains unexpected medicines: "
            f"{sorted(unexpected_ids)[:20]}"
        )

    predictions = pd.to_numeric(
        forecast_df[
            PREDICTION_COLUMN
        ],
        errors="coerce",
    )

    if predictions.isna().any():
        raise RuntimeError(
            "Forecast output contains NULL/non-numeric predictions."
        )

    prediction_values = predictions.to_numpy(
        dtype=float,
    )

    if not np.isfinite(
        prediction_values
    ).all():
        raise RuntimeError(
            "Forecast output contains non-finite predictions."
        )

    if (predictions < 0).any():
        raise RuntimeError(
            "Forecast output contains negative predictions."
        )

    forecast_dates = pd.to_datetime(
        forecast_df[
            FORECAST_DATE_COLUMN
        ],
        errors="coerce",
    )

    if forecast_dates.isna().any():
        raise RuntimeError(
            "Forecast output contains invalid Forecast_Date values."
        )

    expected_rows_per_medicine = int(
        config.prediction_length
    )

    if expected_rows_per_medicine <= 0:
        raise RuntimeError(
            "Forecast configuration has invalid prediction_length: "
            f"{expected_rows_per_medicine}"
        )

    validation_df = pd.DataFrame(
        {
            MEDICINE_ID_COLUMN: normalized_ids,
            FORECAST_DATE_COLUMN: forecast_dates,
        }
    )

    counts = (
        validation_df
        .groupby(
            MEDICINE_ID_COLUMN
        )
        .size()
    )

    invalid_counts = counts[
        counts
        != expected_rows_per_medicine
    ]

    if not invalid_counts.empty:
        raise RuntimeError(
            "Forecast horizon mismatch for medicines: "
            f"{invalid_counts.head(20).to_dict()}"
        )

    duplicate_dates = (
        validation_df
        .duplicated(
            subset=[
                MEDICINE_ID_COLUMN,
                FORECAST_DATE_COLUMN,
            ]
        )
    )

    if duplicate_dates.any():
        raise RuntimeError(
            "Forecast output contains duplicate "
            "(Medicine_ID, Forecast_Date) pairs."
        )

    output_model_map = pd.DataFrame(
        {
            MEDICINE_ID_COLUMN: normalized_ids,
            MODEL_COLUMN: (
                forecast_df[
                    MODEL_COLUMN
                ]
                .astype(str)
                .str.strip()
            ),
        }
    )

    if output_model_map[
        MODEL_COLUMN
    ].eq("").any():
        raise RuntimeError(
            "Forecast output contains empty Selected_Model values."
        )

    forecastable_models = set(
        output_model_map.loc[
            output_model_map[
                MEDICINE_ID_COLUMN
            ].isin(
                forecastable_medicine_ids
            ),
            MODEL_COLUMN,
        ]
    )

    invalid_forecastable_models = (
        forecastable_models
        - ALLOWED_MODELS
    )

    if invalid_forecastable_models:
        raise RuntimeError(
            "Forecastable medicines contain unsupported models: "
            f"{sorted(invalid_forecastable_models)}"
        )

    dormant_output = forecast_df.loc[
        normalized_ids.isin(
            dormant_medicine_ids
        )
    ].copy()

    if not dormant_output.empty:

        dormant_models = set(
            dormant_output[
                MODEL_COLUMN
            ]
            .astype(str)
            .str.strip()
        )

        if dormant_models != {
            DORMANT_MODEL_NAME
        }:
            raise RuntimeError(
                "Dormant medicines must use "
                f"{DORMANT_MODEL_NAME!r} only."
            )

        dormant_predictions = pd.to_numeric(
            dormant_output[
                PREDICTION_COLUMN
            ],
            errors="coerce",
        )

        if not (
            dormant_predictions
            == 0.0
        ).all():
            raise RuntimeError(
                "Dormant medicines must have deterministic "
                "zero-demand predictions."
            )

    if "Eligibility_Status" in forecast_df.columns:

        dormant_statuses = set(
            forecast_df.loc[
                normalized_ids.isin(
                    dormant_medicine_ids
                ),
                "Eligibility_Status",
            ]
            .astype(str)
            .str.strip()
        )

        if dormant_medicine_ids and dormant_statuses != {
            ELIGIBILITY_DORMANT
        }:
            raise RuntimeError(
                "Dormant medicines have invalid eligibility status."
            )


# ============================================================================
# Publication
# ============================================================================


def _publish_latest_forecast(
    forecast_df: pd.DataFrame,
    *,
    output_dir: Path,
) -> Path:
    """
    Atomically publish a fully validated forecast as latest.parquet.

    The previous latest.parquet remains untouched until the new
    artifact has been completely written.
    """

    latest_path = (
        output_dir
        / "latest.parquet"
    )

    _atomic_write_parquet(
        forecast_df,
        latest_path,
    )

    logger.info(
        "Forecast promoted to latest.parquet: %s",
        latest_path,
    )

    return latest_path


# ============================================================================
# Main production pipeline
# ============================================================================


def run_monthly_forecast(
    forecasting_series_path: Path,
    output_dir: Path,
    config: Optional[
        ForecastConfig
    ] = None,
) -> BatchForecastRunResult:
    """
    Execute the production demand forecasting workflow.

    Pipeline:

        validated demand input
                |
                v
        forecast eligibility classification
                |
                +---------------------------+
                |                           |
                v                           v
        ACTIVE + STALE                   DORMANT
                |                           |
                v                           v
        validation routing             zero-demand
                |                       deterministic
                v                           |
        model forecasting                 |
                |                           |
                +-------------+-------------+
                              |
                              v
                     complete population
                         validation
                              |
                              v
                     versioned forecast
                              |
                              v
                    publication quality gate
                              |
                              v
                       latest.parquet
                              |
                              v
                        manifest.json
    """

    started_at = datetime.now(
        timezone.utc,
    )

    run_id = (
        f"{started_at.strftime('%Y%m%d_%H%M%S')}_"
        f"{uuid.uuid4().hex[:6]}"
    )

    forecasting_series_path = Path(
        forecasting_series_path
    )

    output_dir = Path(
        output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if config is None:
        config = DEFAULT_CONFIG

    logger.info(
        "Starting production forecast run %s",
        run_id,
    )

    logger.info(
        "Forecast input: %s",
        forecasting_series_path,
    )

    logger.info(
        "Output directory: %s",
        output_dir,
    )

    # ------------------------------------------------------------------
    # 1. Load and validate production demand history.
    # ------------------------------------------------------------------

    forecasting_df = (
        _load_forecasting_series(
            forecasting_series_path
        )
    )

    # ------------------------------------------------------------------
    # 2. Build deterministic medicine forecast eligibility table.
    # ------------------------------------------------------------------

    eligibility_table = (
        build_eligibility_table(
            history_df=forecasting_df,
            medicine_id_column=MEDICINE_ID_COLUMN,
            timestamp_column=TIMESTAMP_COLUMN,
        )
    )

    if eligibility_table.empty:
        raise RuntimeError(
            "Forecast eligibility table is empty."
        )

    required_eligibility_columns = {
        MEDICINE_ID_COLUMN,
        "Gap_Days",
        "Eligibility_Status",
    }

    missing_eligibility_columns = (
        required_eligibility_columns
        - set(eligibility_table.columns)
    )

    if missing_eligibility_columns:
        raise RuntimeError(
            "Eligibility table missing required columns: "
            f"{sorted(missing_eligibility_columns)}"
        )

    eligibility_table = (
        eligibility_table
        .copy()
    )

    eligibility_table[
        MEDICINE_ID_COLUMN
    ] = _normalize_medicine_ids(
        eligibility_table[
            MEDICINE_ID_COLUMN
        ],
        context="Forecast eligibility table",
    )

    if eligibility_table[
        MEDICINE_ID_COLUMN
    ].duplicated().any():
        raise RuntimeError(
            "Eligibility table contains duplicate Medicine_ID values."
        )

    allowed_eligibility_statuses = {
        ELIGIBILITY_ACTIVE,
        ELIGIBILITY_STALE,
        ELIGIBILITY_DORMANT,
    }

    actual_eligibility_statuses = set(
        eligibility_table[
            "Eligibility_Status"
        ]
        .astype(str)
        .str.strip()
    )

    invalid_eligibility_statuses = (
        actual_eligibility_statuses
        - allowed_eligibility_statuses
    )

    if invalid_eligibility_statuses:
        raise RuntimeError(
            "Eligibility table contains invalid statuses: "
            f"{sorted(invalid_eligibility_statuses)}"
        )

    eligibility_counts = (
        eligibility_table[
            "Eligibility_Status"
        ]
        .value_counts()
        .to_dict()
    )

    logger.info(
        "Forecast eligibility decisions: %s",
        eligibility_counts,
    )

    # ------------------------------------------------------------------
    # 3. Split medicines into forecastable and dormant populations.
    # ------------------------------------------------------------------

    eligible_eligibility = (
        eligibility_table[
            eligibility_table[
                "Eligibility_Status"
            ].isin(
                FORECASTABLE_ELIGIBILITY_STATUSES
            )
        ]
        .copy()
        .sort_values(
            MEDICINE_ID_COLUMN
        )
        .reset_index(
            drop=True
        )
    )

    dormant_eligibility = (
        eligibility_table[
            eligibility_table[
                "Eligibility_Status"
            ]
            == ELIGIBILITY_DORMANT
        ]
        .copy()
        .sort_values(
            MEDICINE_ID_COLUMN
        )
        .reset_index(
            drop=True
        )
    )

    eligible_medicine_ids = (
        eligible_eligibility[
            MEDICINE_ID_COLUMN
        ]
        .astype(str)
        .tolist()
    )

    dormant_medicine_ids = (
        dormant_eligibility[
            MEDICINE_ID_COLUMN
        ]
        .astype(str)
        .tolist()
    )

    logger.info(
        "Eligible medicines=%d | dormant medicines=%d",
        len(eligible_medicine_ids),
        len(dormant_medicine_ids),
    )

    # ------------------------------------------------------------------
    # 4. Production population accounting.
    #
    # requested means medicines sent into model forecasting.
    #
    # DORMANT medicines are intentionally excluded from model inference.
    # ------------------------------------------------------------------

    medicine_ids = eligible_medicine_ids

    requested = len(
        medicine_ids
    )

    requested_ids = set(
        medicine_ids
    )

    dormant_ids = set(
        dormant_medicine_ids
    )

    all_medicine_ids = (
        requested_ids
        | dormant_ids
    )

    if not all_medicine_ids:
        raise RuntimeError(
            "No medicines available for production forecasting."
        )

    if requested_ids & dormant_ids:
        raise RuntimeError(
            "Eligible and dormant medicine populations overlap."
        )

    logger.info(
        "Production population loaded | total=%d | "
        "forecastable=%d | dormant=%d | history_rows=%d",
        len(all_medicine_ids),
        requested,
        len(dormant_ids),
        len(forecasting_df),
    )

    # ------------------------------------------------------------------
    # 5. Determine forecast start date.
    #
    # Forecast begins one day after the latest available demand date.
    # ------------------------------------------------------------------

    latest_history_timestamp = (
        pd.to_datetime(
            forecasting_df[
                TIMESTAMP_COLUMN
            ],
            errors="raise",
        )
        .max()
    )

    forecast_start_date = (
        pd.Timestamp(
            latest_history_timestamp
        ).normalize()
        + pd.Timedelta(
            days=1
        )
    )

    logger.info(
        "Forecast start date: %s",
        forecast_start_date.strftime(
            "%Y-%m-%d"
        ),
    )

    # ------------------------------------------------------------------
    # 6. Load routing evidence and build routing table.
    #
    # Only required when there are ACTIVE or STALE medicines.
    # ------------------------------------------------------------------

    if requested > 0:

        routing_results = (
            _load_routing_results()
        )

        logger.info(
            "Routing evidence loaded | rows=%d",
            len(routing_results),
        )

        routing_table = (
            build_routing_table(
                routing_results,
                medicine_ids,
            )
        )

        _persist_routing_table(
            routing_table
        )

        routing_counts = (
            routing_table[
                "Selected_Model"
            ]
            .value_counts()
            .to_dict()
        )

        logger.info(
            "Production routing decisions: %s",
            routing_counts,
        )

    else:

        routing_table = pd.DataFrame(
            columns=[
                MEDICINE_ID_COLUMN,
                "Chronos_AE",
                "TSB_AE",
                "Validation_Windows",
                "Validation_Advantage_Pct",
                "Selected_Model",
                "Routing_Rule",
                "Routing_Reason",
            ]
        )

        logger.info(
            "No ACTIVE or STALE medicines. "
            "Skipping routing and model initialization."
        )

    # ------------------------------------------------------------------
    # 7. Generate forecasts for ACTIVE and STALE medicines only.
    # ------------------------------------------------------------------

    generated_at = datetime.now(
        timezone.utc,
    )

    model_forecast_df = pd.DataFrame()

    failed: list[str] = []

    if requested > 0:

        predictor = (
            PredictorService.get_instance(
                config
            )
        )

        router = (
            ProductionForecastRouter(
                predictor_service=predictor,
            )
        )

        forecast_service = (
            ProductionForecastService(
                router=router,
                forecast_config=config,
                predictor=predictor,
            )
        )

        model_forecast_df, failed = (
            forecast_service.forecast_batch(
                history_df=forecasting_df,
                routing_table=routing_table,
                item_ids=medicine_ids,
            )
        )

        if not isinstance(
            model_forecast_df,
            pd.DataFrame,
        ):
            raise TypeError(
                "forecast_batch must return a pandas DataFrame "
                "as its first result."
            )

        if not isinstance(
            failed,
            list,
        ):
            raise TypeError(
                "forecast_batch must return a list of failed "
                "medicine IDs as its second result."
            )

    failed = [
        str(
            medicine_id
        ).strip()
        for medicine_id in failed
        if str(
            medicine_id
        ).strip()
    ]

    failed = list(
        dict.fromkeys(
            failed
        )
    )

    failed_set = set(
        failed
    )

    unknown_failed_ids = (
        failed_set
        - requested_ids
    )

    if unknown_failed_ids:
        raise RuntimeError(
            "forecast_batch returned failed medicine IDs outside "
            "the requested forecastable population: "
            f"{sorted(unknown_failed_ids)[:20]}"
        )

    successful_ids = (
        requested_ids
        - failed_set
    )

    successful = len(
        successful_ids
    )

    if successful + len(failed) != requested:
        raise RuntimeError(
            "Forecast batch accounting mismatch: "
            f"successful={successful}, "
            f"failed={len(failed)}, "
            f"requested={requested}."
        )

    logger.info(
        "Model forecasting complete | "
        "successful=%d/%d | failed=%d",
        successful,
        requested,
        len(failed),
    )

    # ------------------------------------------------------------------
    # 8. Attach eligibility metadata to model forecasts.
    # ------------------------------------------------------------------

    if not model_forecast_df.empty:

        model_forecast_df = (
            _attach_eligibility_metadata(
                model_forecast_df,
                eligibility_table,
                generated_at,
            )
        )

    # ------------------------------------------------------------------
    # 9. Generate deterministic dormant forecasts.
    # ------------------------------------------------------------------

    dormant_forecast_df = (
        _build_dormant_forecast_records(
            dormant_eligibility=dormant_eligibility,
            forecast_start_date=forecast_start_date,
            config=config,
            generated_at=generated_at,
        )
    )

    logger.info(
        "Dormant deterministic records generated | "
        "medicines=%d | rows=%d",
        len(dormant_ids),
        len(dormant_forecast_df),
    )

    # ------------------------------------------------------------------
    # 10. Combine model forecasts and dormant records.
    # ------------------------------------------------------------------

    output_frames: list[pd.DataFrame] = []

    if not model_forecast_df.empty:
        output_frames.append(
            model_forecast_df
        )

    if not dormant_forecast_df.empty:
        output_frames.append(
            dormant_forecast_df
        )

    if not output_frames:
        forecast_df = pd.DataFrame()

    elif len(output_frames) == 1:
        forecast_df = (
            output_frames[0]
            .copy()
        )

    else:
        forecast_df = (
            pd.concat(
                output_frames,
                ignore_index=True,
                sort=False,
            )
        )

    if not forecast_df.empty:

        forecast_df[
            MEDICINE_ID_COLUMN
        ] = _normalize_medicine_ids(
            forecast_df[
                MEDICINE_ID_COLUMN
            ],
            context="Combined production forecast",
        )

        forecast_df[
            FORECAST_DATE_COLUMN
        ] = pd.to_datetime(
            forecast_df[
                FORECAST_DATE_COLUMN
            ],
            errors="raise",
        )

        forecast_df = (
            forecast_df
            .sort_values(
                [
                    MEDICINE_ID_COLUMN,
                    FORECAST_DATE_COLUMN,
                ]
            )
            .reset_index(
                drop=True
            )
        )

    # ------------------------------------------------------------------
    # 11. Create immutable versioned run directory.
    # ------------------------------------------------------------------

    run_dir = (
        output_dir
        / run_id
    )

    run_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    forecast_path = (
        run_dir
        / "forecast.parquet"
    )

    # ------------------------------------------------------------------
    # 12. Validate complete output before publication.
    #
    # Validation is against the FULL medicine population.
    # ------------------------------------------------------------------

    validation_error: Optional[
        str
    ] = None

    try:

        _validate_forecast_output(
            forecast_df,
            expected_medicine_ids=all_medicine_ids,
            forecastable_medicine_ids=requested_ids,
            dormant_medicine_ids=dormant_ids,
            config=config,
        )

    except Exception as exc:

        validation_error = str(
            exc
        )

        logger.exception(
            "Production forecast output validation failed."
        )

    # ------------------------------------------------------------------
    # 13. Write immutable versioned forecast artifact.
    #
    # Failed runs retain available output for auditability, but never
    # replace latest.parquet unless all publication gates pass.
    # ------------------------------------------------------------------

    if not forecast_df.empty:

        _atomic_write_parquet(
            forecast_df,
            forecast_path,
        )

        logger.info(
            "Versioned forecast written: %s",
            forecast_path,
        )

    else:

        logger.error(
            "No forecast rows produced for run %s.",
            run_id,
        )

    # ------------------------------------------------------------------
    # 14. Publication quality gate.
    #
    # All forecastable medicines must succeed.
    #
    # If requested == 0, the run can still publish when the entire
    # population consists of valid dormant deterministic records.
    # ------------------------------------------------------------------

    if requested > 0:

        success_rate = (
            successful
            / requested
        )

        complete_success = (
            success_rate
            >= MIN_SUCCESS_RATE
        )

    else:

        success_rate = 1.0

        complete_success = True

    output_valid = (
        validation_error
        is None
    )

    forecast_artifact_exists = (
        forecast_path.exists()
    )

    published = (
        complete_success
        and output_valid
        and forecast_artifact_exists
    )

    if published:

        latest_path = (
            _publish_latest_forecast(
                forecast_df,
                output_dir=output_dir,
            )
        )

        publish_note = (
            "Published successfully: "
            f"forecastable succeeded={successful}/{requested} "
            f"({success_rate:.1%}); "
            f"dormant={len(dormant_ids)}; "
            f"total_population={len(all_medicine_ids)}. "
            f"Validated output promoted to "
            f"{latest_path.name}."
        )

        logger.info(
            publish_note
        )

    else:

        reasons: list[
            str
        ] = []

        if not complete_success:
            reasons.append(
                f"success rate {success_rate:.1%} below required "
                f"{MIN_SUCCESS_RATE:.0%}"
            )

        if not output_valid:
            reasons.append(
                f"output validation failed: "
                f"{validation_error}"
            )

        if not forecast_artifact_exists:
            reasons.append(
                "versioned forecast artifact was not created"
            )

        reason_text = (
            "; ".join(
                reasons
            )
        )

        publish_note = (
            "NOT published: "
            f"{reason_text}. "
            "The previous latest.parquet was retained."
        )

        logger.error(
            publish_note
        )

    # ------------------------------------------------------------------
    # 15. Build immutable manifest.
    #
    # medicines_requested/succeeded/failed refer to model forecasting
    # population only. Dormant medicines are deterministic policy records.
    # ------------------------------------------------------------------

    completed_at = datetime.now(
        timezone.utc,
    )

    manifest = (
        BatchForecastRunResult(
            run_id=run_id,
            started_at=started_at,
            completed_at=completed_at,
            medicines_requested=requested,
            medicines_succeeded=successful,
            medicines_failed=len(
                failed
            ),
            failed_medicine_ids=failed,
            output_path=str(
                forecast_path
            ),
            published=published,
            publish_note=publish_note,
        )
    )

    # ------------------------------------------------------------------
    # 16. Write manifest.
    # ------------------------------------------------------------------

    manifest_path = (
        run_dir
        / "manifest.json"
    )

    _atomic_write_json(
        manifest.model_dump(
            mode="json"
        ),
        manifest_path,
    )

    # ------------------------------------------------------------------
    # 17. Final logging.
    # ------------------------------------------------------------------

    elapsed_seconds = (
        completed_at
        - started_at
    ).total_seconds()

    logger.info(
        "Forecast run %s complete | "
        "forecastable_successful=%d/%d | "
        "failed=%d | "
        "dormant=%d | "
        "total_population=%d | "
        "published=%s | "
        "elapsed=%.1fs",
        run_id,
        successful,
        requested,
        len(failed),
        len(dormant_ids),
        len(all_medicine_ids),
        published,
        elapsed_seconds,
    )

    if failed:

        logger.warning(
            "Failed medicine IDs (%d): %s",
            len(failed),
            failed[:20],
        )

    return manifest


# ============================================================================
# CLI
# ============================================================================


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Run the Finemed PharmaAI production demand "
            "forecasting pipeline."
        ),
    )

    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help=(
            "Path to the validated forecasting series parquet "
            "containing either "
            "[MDCODE, INVDT, Demand_Qty] or "
            "[Medicine_ID, timestamp, target]."
        ),
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help=(
            "Directory where immutable versioned production forecasts, "
            "manifest.json, and latest.parquet will be written."
        ),
    )

    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=[
            "DEBUG",
            "INFO",
            "WARNING",
            "ERROR",
            "CRITICAL",
        ],
        help="Logging verbosity.",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(
            logging,
            args.log_level,
        ),
        format=(
            "%(asctime)s | %(levelname)s | "
            "%(name)s | %(message)s"
        ),
    )

    result = run_monthly_forecast(
        forecasting_series_path=args.input,
        output_dir=args.output_dir,
    )

    print()
    print("=" * 80)
    print(
        "FINEMED PRODUCTION FORECAST RUN COMPLETE"
    )
    print("=" * 80)
    print(
        f"Run ID: {result.run_id}"
    )
    print(
        f"Requested: {result.medicines_requested}"
    )
    print(
        f"Succeeded: {result.medicines_succeeded}"
    )
    print(
        f"Failed: {result.medicines_failed}"
    )
    print(
        f"Published: {result.published}"
    )
    print(
        f"Output: {result.output_path}"
    )
    print(
        f"Note: {result.publish_note}"
    )
    print("=" * 80)

    if result.failed_medicine_ids:

        print(
            "Failed medicine IDs: "
            f"{result.failed_medicine_ids[:20]}"
        )

    if not result.published:

        raise SystemExit(
            "Forecast run completed but did not pass "
            "the publication quality gate."
        )