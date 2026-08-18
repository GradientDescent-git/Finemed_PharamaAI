from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
from pydantic import BaseModel

from finemed_ai.demand_forecasting.config import (
    DEFAULT_CONFIG,
    ForecastConfig,
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

ROBUSTNESS_PATH = Path(
    "data/05_gold/demand_forecasting/medicine_robustness/"
    "medicine_model_robustness.parquet"
)

ROUTING_TABLE_PATH = Path(
    "data/05_gold/demand_forecasting/medicine_robustness/"
    "production_routing_table.parquet"
)

MIN_SUCCESS_RATE = 0.50


# ============================================================================
# Run manifest schema
# ============================================================================

class BatchForecastRunResult(BaseModel):
    """
    Production forecast run manifest.

    Records the identity, timing, success/failure statistics,
    output location, and publication status of one forecast run.
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
# Routing table builder
# ============================================================================

def build_routing_table(
    robustness: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the frozen production routing table from the validated
    medicine robustness dataset.

    Routing policy:

        Validation_Advantage_Pct >= 30%
            -> chronos-2-P50

        Validation_Advantage_Pct < 30%
            -> tsb

    The routing table is intentionally frozen for production inference.
    """

    required_columns = {
        MEDICINE_ID_COLUMN,
        "Validation_Chronos_AE",
        "Validation_TSB_AE",
        "Validation_Advantage_Pct",
    }

    if not isinstance(
        robustness,
        pd.DataFrame,
    ):
        raise TypeError(
            "robustness must be a pandas DataFrame."
        )

    missing = (
        required_columns
        - set(robustness.columns)
    )

    if missing:
        raise ValueError(
            "Robustness dataset is missing required columns: "
            f"{sorted(missing)}"
        )

    if robustness.empty:
        raise ValueError(
            "Robustness dataset is empty."
        )

    routing = robustness[
        list(required_columns)
    ].copy()

    # ------------------------------------------------------------------
    # Normalize Medicine_ID
    # ------------------------------------------------------------------

    routing[MEDICINE_ID_COLUMN] = (
        routing[MEDICINE_ID_COLUMN]
        .astype(str)
        .str.strip()
    )

    if routing[
        MEDICINE_ID_COLUMN
    ].eq("").any():
        raise ValueError(
            "Robustness dataset contains empty Medicine_ID values."
        )

    duplicate_ids = (
        routing[
            MEDICINE_ID_COLUMN
        ]
        .duplicated()
    )

    if duplicate_ids.any():
        duplicated = (
            routing.loc[
                duplicate_ids,
                MEDICINE_ID_COLUMN,
            ]
            .tolist()
        )

        raise ValueError(
            "Robustness dataset contains duplicate Medicine_ID values: "
            f"{duplicated}"
        )

    # ------------------------------------------------------------------
    # Numeric validation
    # ------------------------------------------------------------------

    numeric_columns = [
        "Validation_Chronos_AE",
        "Validation_TSB_AE",
        "Validation_Advantage_Pct",
    ]

    for column in numeric_columns:

        routing[column] = pd.to_numeric(
            routing[column],
            errors="coerce",
        )

        if routing[column].isna().any():
            raise ValueError(
                f"{column} contains null/non-numeric values."
            )

    # ------------------------------------------------------------------
    # Frozen routing threshold
    # ------------------------------------------------------------------

    threshold = 30.0

    routing["Selected_Model"] = (
        routing["Validation_Advantage_Pct"]
        .ge(threshold)
        .map(
            {
                True: "chronos-2-P50",
                False: "tsb",
            }
        )
    )

    routing["Routing_Rule"] = (
        "Validation_Advantage_Pct >= 30% -> "
        "chronos-2-P50; otherwise -> tsb"
    )

    routing["Threshold"] = threshold

    routing = routing[
        [
            MEDICINE_ID_COLUMN,
            "Validation_Chronos_AE",
            "Validation_TSB_AE",
            "Validation_Advantage_Pct",
            "Selected_Model",
            "Routing_Rule",
            "Threshold",
        ]
    ].sort_values(
        MEDICINE_ID_COLUMN
    ).reset_index(
        drop=True
    )

    logger.info(
        "Built routing table | medicines=%d | "
        "chronos=%d | tsb=%d | threshold=%.1f%%",
        len(routing),
        int(
            (
                routing["Selected_Model"]
                == "chronos-2-P50"
            ).sum()
        ),
        int(
            (
                routing["Selected_Model"]
                == "tsb"
            ).sum()
        ),
        threshold,
    )

    return routing


# ============================================================================
# Silver forecasting series loader
# ============================================================================

def _load_forecasting_series(
    forecasting_series_path: str | Path,
) -> pd.DataFrame:
    """
    Load the validated Silver daily-demand forecasting series.

    Source schema:

        MDCODE
        INVDT
        Demand_Qty

    Production schema:

        Medicine_ID
        timestamp
        target

    The production forecasting service consumes the normalized schema.
    """

    # ------------------------------------------------------------------
    # Accept both str and Path.
    # ------------------------------------------------------------------

    forecasting_series_path = Path(
        forecasting_series_path
    )

    if not forecasting_series_path.exists():
        raise FileNotFoundError(
            "Forecasting series does not exist: "
            f"{forecasting_series_path}"
        )

    if not forecasting_series_path.is_file():
        raise ValueError(
            "Forecasting series path is not a file: "
            f"{forecasting_series_path}"
        )

    df = pd.read_parquet(
        forecasting_series_path
    )

    required_columns = {
        "MDCODE",
        "INVDT",
        "Demand_Qty",
    }

    missing_columns = (
        required_columns
        - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            "Forecasting series missing required columns: "
            f"{sorted(missing_columns)}"
        )

    df = df[
        [
            "MDCODE",
            "INVDT",
            "Demand_Qty",
        ]
    ].copy()

    # ------------------------------------------------------------------
    # Medicine IDs
    # ------------------------------------------------------------------

    df["MDCODE"] = (
        df["MDCODE"]
        .astype(str)
        .str.strip()
    )

    if df["MDCODE"].eq("").any():
        raise ValueError(
            "Forecasting series contains empty MDCODE values."
        )

    # ------------------------------------------------------------------
    # Dates
    # ------------------------------------------------------------------

    df["INVDT"] = pd.to_datetime(
        df["INVDT"],
        errors="coerce",
    )

    if df["INVDT"].isna().any():
        raise ValueError(
            "Forecasting series contains invalid INVDT values."
        )

    # Normalize to calendar-day timestamps.

    df["INVDT"] = (
        df["INVDT"]
        .dt.normalize()
    )

    # ------------------------------------------------------------------
    # Demand
    # ------------------------------------------------------------------

    df["Demand_Qty"] = pd.to_numeric(
        df["Demand_Qty"],
        errors="coerce",
    )

    if df["Demand_Qty"].isna().any():
        raise ValueError(
            "Forecasting series contains NULL/non-numeric "
            "Demand_Qty values."
        )

    demand_values = (
        df["Demand_Qty"]
        .to_numpy(dtype=float)
    )

    if not pd.Series(
        demand_values
    ).map(
        lambda x: pd.notna(x)
    ).all():
        raise ValueError(
            "Forecasting series contains invalid demand values."
        )

    if not (
        pd.Series(
            demand_values
        ).map(
            lambda x: abs(float(x)) != float("inf")
        ).all()
    ):
        raise ValueError(
            "Forecasting series contains non-finite demand values."
        )

    if (
        df["Demand_Qty"] < 0
    ).any():
        raise ValueError(
            "Forecasting series contains negative demand."
        )

    # ------------------------------------------------------------------
    # Silver contract:
    #
    # Exactly one row per medicine/day.
    # ------------------------------------------------------------------

    duplicate_count = (
        df.duplicated(
            [
                "MDCODE",
                "INVDT",
            ]
        )
        .sum()
    )

    if duplicate_count:
        raise ValueError(
            "Forecasting series contains "
            f"{duplicate_count} duplicate MDCODE/INVDT rows."
        )

    # ------------------------------------------------------------------
    # Normalize into ProductionForecastService schema.
    # ------------------------------------------------------------------

    production_df = (
        df.rename(
            columns={
                "MDCODE": MEDICINE_ID_COLUMN,
                "INVDT": TIMESTAMP_COLUMN,
                "Demand_Qty": TARGET_COLUMN,
            }
        )
        .sort_values(
            [
                MEDICINE_ID_COLUMN,
                TIMESTAMP_COLUMN,
            ]
        )
        .reset_index(
            drop=True
        )
    )

    # ------------------------------------------------------------------
    # Final normalized schema validation.
    # ------------------------------------------------------------------

    expected_columns = {
        MEDICINE_ID_COLUMN,
        TIMESTAMP_COLUMN,
        TARGET_COLUMN,
    }

    if set(
        production_df.columns
    ) != expected_columns:
        raise ValueError(
            "Normalized forecasting series has unexpected schema: "
            f"{production_df.columns.tolist()}"
        )

    logger.info(
        "Loaded validated forecasting series: "
        "%d rows, %d medicines, %s -> %s",
        len(production_df),
        production_df[
            MEDICINE_ID_COLUMN
        ].nunique(),
        production_df[
            TIMESTAMP_COLUMN
        ].min().date(),
        production_df[
            TIMESTAMP_COLUMN
        ].max().date(),
    )

    return production_df


# ============================================================================
# Routing table loader
# ============================================================================

def _load_routing_table() -> pd.DataFrame:
    """
    Load the frozen production routing table.

    If it does not exist, rebuild it once from the validated robustness
    dataset and persist the resulting frozen table.
    """

    if ROUTING_TABLE_PATH.exists():

        logger.info(
            "Loading existing production routing table: %s",
            ROUTING_TABLE_PATH,
        )

        routing = pd.read_parquet(
            ROUTING_TABLE_PATH
        )

    else:

        logger.warning(
            "Production routing table not found. "
            "Rebuilding from robustness dataset."
        )

        if not ROBUSTNESS_PATH.exists():
            raise FileNotFoundError(
                "Neither production routing table nor robustness "
                "dataset exists.\n"
                f"Routing table: {ROUTING_TABLE_PATH}\n"
                f"Robustness: {ROBUSTNESS_PATH}"
            )

        robustness = pd.read_parquet(
            ROBUSTNESS_PATH
        )

        routing = build_routing_table(
            robustness
        )

        ROUTING_TABLE_PATH.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        routing.to_parquet(
            ROUTING_TABLE_PATH,
            index=False,
        )

        logger.info(
            "Production routing table created: %s",
            ROUTING_TABLE_PATH,
        )

    # ------------------------------------------------------------------
    # Validate routing table before returning.
    # ------------------------------------------------------------------

    required_columns = {
        MEDICINE_ID_COLUMN,
        "Validation_Advantage_Pct",
        "Selected_Model",
        "Routing_Rule",
        "Threshold",
    }

    missing = (
        required_columns
        - set(routing.columns)
    )

    if missing:
        raise ValueError(
            "Production routing table missing required columns: "
            f"{sorted(missing)}"
        )

    if routing.empty:
        raise ValueError(
            "Production routing table is empty."
        )

    routing[
        MEDICINE_ID_COLUMN
    ] = (
        routing[
            MEDICINE_ID_COLUMN
        ]
        .astype(str)
        .str.strip()
    )

    if routing[
        MEDICINE_ID_COLUMN
    ].eq("").any():
        raise ValueError(
            "Production routing table contains empty Medicine_ID."
        )

    if routing[
        MEDICINE_ID_COLUMN
    ].duplicated().any():
        raise ValueError(
            "Production routing table contains duplicate Medicine_ID."
        )

    allowed_models = {
        "chronos-2-P50",
        "tsb",
    }

    invalid_models = (
        set(
            routing[
                "Selected_Model"
            ].dropna().unique()
        )
        - allowed_models
    )

    if invalid_models:
        raise ValueError(
            "Production routing table contains unsupported models: "
            f"{sorted(invalid_models)}"
        )

    advantages = pd.to_numeric(
        routing[
            "Validation_Advantage_Pct"
        ],
        errors="coerce",
    )

    if advantages.isna().any():
        raise ValueError(
            "Routing table contains invalid "
            "Validation_Advantage_Pct values."
        )

    threshold_values = pd.to_numeric(
        routing[
            "Threshold"
        ],
        errors="coerce",
    )

    if threshold_values.isna().any():
        raise ValueError(
            "Routing table contains invalid Threshold values."
        )

    if not (
        threshold_values == 30.0
    ).all():
        raise ValueError(
            "Production routing table does not use the frozen "
            "30% routing threshold."
        )

    # ------------------------------------------------------------------
    # Validate selected model against advantage.
    # ------------------------------------------------------------------

    expected_models = (
        advantages
        .ge(30.0)
        .map(
            {
                True: "chronos-2-P50",
                False: "tsb",
            }
        )
    )

    actual_models = (
        routing[
            "Selected_Model"
        ]
    )

    inconsistent = (
        expected_models
        != actual_models
    )

    if inconsistent.any():

        bad_rows = routing.loc[
            inconsistent,
            [
                MEDICINE_ID_COLUMN,
                "Validation_Advantage_Pct",
                "Selected_Model",
            ],
        ]

        raise ValueError(
            "Production routing table violates the frozen routing "
            "policy:\n"
            f"{bad_rows.to_dict(orient='records')}"
        )

    logger.info(
        "Loaded production routing table: %d medicines | "
        "chronos=%d | tsb=%d",
        routing[
            MEDICINE_ID_COLUMN
        ].nunique(),
        int(
            (
                routing["Selected_Model"]
                == "chronos-2-P50"
            ).sum()
        ),
        int(
            (
                routing["Selected_Model"]
                == "tsb"
            ).sum()
        ),
    )

    return routing


# ============================================================================
# Production forecast runner
# ============================================================================

def run_monthly_forecast(
    forecasting_series_path: str | Path,
    output_dir: str | Path,
    config: ForecastConfig = DEFAULT_CONFIG,
    predictor: Optional[PredictorService] = None,
) -> BatchForecastRunResult:
    """
    Execute one complete production forecasting run.

    Pipeline:

        Silver daily demand
            |
            v
        Input validation
            |
            v
        Production schema normalization
            |
            v
        Frozen routing table
            |
            v
        ProductionForecastService
            |
            +---- Chronos-2 P50
            |
            +---- TSB
            |
            v
        Output validation
            |
            v
        Versioned forecast
            |
            v
        Publication quality gate
            |
            v
        latest.parquet
            |
            v
        manifest.json
    """

    forecasting_series_path = Path(
        forecasting_series_path
    )

    output_dir = Path(
        output_dir
    )

    # ------------------------------------------------------------------
    # Run identity
    # ------------------------------------------------------------------

    run_id = (
        f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_"
        f"{uuid.uuid4().hex[:6]}"
    )

    started_at = datetime.now(
        timezone.utc
    )

    logger.info(
        "Starting forecast run %s",
        run_id,
    )

    # ------------------------------------------------------------------
    # 1. Load Silver daily demand
    # ------------------------------------------------------------------

    history_df = _load_forecasting_series(
        forecasting_series_path
    )

    medicine_ids = sorted(
        history_df[
            MEDICINE_ID_COLUMN
        ]
        .astype(str)
        .str.strip()
        .unique()
    )

    requested = len(
        medicine_ids
    )

    logger.info(
        "Loaded daily demand for %d medicines",
        requested,
    )

    if requested == 0:
        raise RuntimeError(
            "No medicines found in forecasting history."
        )

    # ------------------------------------------------------------------
    # 2. Load frozen routing table
    # ------------------------------------------------------------------

    routing_table = _load_routing_table()

    routing_ids = set(
        routing_table[
            MEDICINE_ID_COLUMN
        ]
        .astype(str)
        .str.strip()
        .unique()
    )

    missing_routing_ids = [
        medicine_id
        for medicine_id in medicine_ids
        if medicine_id not in routing_ids
    ]

    if missing_routing_ids:

        logger.warning(
            "%d/%d medicines have no routing record. "
            "ProductionForecastService will fall back to TSB. "
            "Examples: %s",
            len(missing_routing_ids),
            requested,
            missing_routing_ids[:20],
        )

    # ------------------------------------------------------------------
    # 3. Initialize production service
    # ------------------------------------------------------------------

    router_service = ProductionForecastService(
        router=ProductionForecastRouter(),
        forecast_config=config,
        predictor=predictor,
    )

    # ------------------------------------------------------------------
    # 4. Execute production forecast
    #
    # IMPORTANT:
    #
    # Do not pass item_ids.
    #
    # This ensures every medicine in Silver is attempted.
    # Missing routing records fall back to TSB.
    # ------------------------------------------------------------------

    forecast_df, failed = (
        router_service.forecast_batch(
            history_df=history_df,
            routing_table=routing_table,
        )
    )

    # ------------------------------------------------------------------
    # 5. Final pipeline-level success accounting
    # ------------------------------------------------------------------

    successful_ids = set(
        forecast_df[
            MEDICINE_ID_COLUMN
        ]
        .astype(str)
        .str.strip()
        .unique()
    )

    successful = len(
        successful_ids
    )

    # Add any medicine-level discrepancy defensively.

    failed_set = set(
        str(x).strip()
        for x in failed
    )

    missing_from_both = (
        set(medicine_ids)
        - successful_ids
        - failed_set
    )

    if missing_from_both:

        logger.error(
            "Internal accounting discrepancy: medicines were neither "
            "successful nor reported as failed: %s",
            sorted(missing_from_both)[:20],
        )

        failed.extend(
            sorted(missing_from_both)
        )

        failed = list(
            dict.fromkeys(
                failed
            )
        )

    # ------------------------------------------------------------------
    # 6. Create versioned run directory
    # ------------------------------------------------------------------

    run_dir = (
        output_dir
        / run_id
    )

    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    forecast_path = (
        run_dir
        / "forecast.parquet"
    )

    forecast_df.to_parquet(
        forecast_path,
        index=False,
    )

    logger.info(
        "Versioned forecast written: %s",
        forecast_path,
    )

    # ------------------------------------------------------------------
    # 7. Production publication quality gate
    # ------------------------------------------------------------------

    success_rate = (
        successful / requested
        if requested
        else 0.0
    )

    published = (
        success_rate
        >= MIN_SUCCESS_RATE
    )

    latest_path = (
        output_dir
        / "latest.parquet"
    )

    if published:

        forecast_df.to_parquet(
            latest_path,
            index=False,
        )

        publish_note = (
            f"Published successfully: "
            f"{successful}/{requested} medicines succeeded "
            f"({success_rate:.1%})."
        )

        logger.info(
            "Forecast promoted to latest.parquet: "
            "%d/%d medicines succeeded (%.1f%%)",
            successful,
            requested,
            success_rate * 100.0,
        )

    else:

        publish_note = (
            f"NOT published: only "
            f"{successful}/{requested} medicines succeeded "
            f"({success_rate:.1%}), below the "
            f"{MIN_SUCCESS_RATE:.0%} publication threshold. "
            f"The previous latest.parquet was retained. "
            f"Check failed_medicine_ids and logs."
        )

        logger.error(
            publish_note
        )

    # ------------------------------------------------------------------
    # 8. Build manifest
    # ------------------------------------------------------------------

    completed_at = datetime.now(
        timezone.utc
    )

    manifest = BatchForecastRunResult(
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

    # ------------------------------------------------------------------
    # 9. Write manifest
    # ------------------------------------------------------------------

    manifest_path = (
        run_dir
        / "manifest.json"
    )

    manifest_path.write_text(
        manifest.model_dump_json(
            indent=2
        ),
        encoding="utf-8",
    )

    # ------------------------------------------------------------------
    # 10. Final logging
    # ------------------------------------------------------------------

    elapsed_seconds = (
        completed_at
        - started_at
    ).total_seconds()

    logger.info(
        "Forecast run %s complete | "
        "successful=%d/%d | failed=%d | "
        "published=%s | elapsed=%.1fs",
        run_id,
        successful,
        requested,
        len(failed),
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