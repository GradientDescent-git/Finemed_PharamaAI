from __future__ import annotations

import numpy as np
import pandas as pd

from finemed_ai.demand_forecasting.production_forecast_router import (
    ProductionForecastRouter,
)
from finemed_ai.demand_forecasting.production_forecast_service import (
    ProductionForecastService,
)


# ============================================================================
# PATHS
# ============================================================================

HISTORY_PATH = (
    "data/04_silver/demand_forecasting/"
    "daily_demand.parquet"
)

ROUTING_PATH = (
    "data/05_gold/demand_forecasting/"
    "medicine_robustness/"
    "production_routing_table.parquet"
)

OUTPUT_PATH = (
    "data/05_gold/demand_forecasting/"
    "production_scale_audit.parquet"
)


# ============================================================================
# HELPERS
# ============================================================================


def load_history() -> pd.DataFrame:
    """Load and normalize production demand history."""

    history = pd.read_parquet(HISTORY_PATH)

    history = history.rename(
        columns={
            "MDCODE": "Medicine_ID",
            "INVDT": "timestamp",
            "Demand_Qty": "target",
        }
    )

    required = {
        "Medicine_ID",
        "timestamp",
        "target",
    }

    missing = required - set(history.columns)

    if missing:
        raise ValueError(
            "History is missing required columns: "
            f"{sorted(missing)}"
        )

    history["Medicine_ID"] = (
        history["Medicine_ID"]
        .astype(str)
        .str.strip()
    )

    history["timestamp"] = pd.to_datetime(
        history["timestamp"],
        errors="raise",
    )

    history["target"] = pd.to_numeric(
        history["target"],
        errors="raise",
    )

    if history["Medicine_ID"].eq("").any():
        raise ValueError(
            "History contains empty Medicine_ID values."
        )

    if not np.isfinite(
        history["target"].to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "History contains non-finite target values."
        )

    return history


def load_frozen_routing() -> pd.DataFrame:
    """
    Load the already-created production routing artifact.

    IMPORTANT:
    This audit does not rebuild model selection.

    Model selection must already have been performed from
    validation-only evidence.
    """

    routing = pd.read_parquet(
        ROUTING_PATH
    )

    required = {
        "Medicine_ID",
        "Validation_Chronos_AE",
        "Validation_TSB_AE",
        "Validation_Advantage_Pct",
        "Selected_Model",
        "Routing_Rule",
        "Threshold",
    }

    missing = required - set(routing.columns)

    if missing:
        raise ValueError(
            "Frozen routing table is missing required "
            f"columns: {sorted(missing)}"
        )

    if routing.empty:
        raise ValueError(
            "Frozen routing table is empty."
        )

    routing["Medicine_ID"] = (
        routing["Medicine_ID"]
        .astype(str)
        .str.strip()
    )

    if routing["Medicine_ID"].eq("").any():
        raise ValueError(
            "Frozen routing table contains empty Medicine_ID."
        )

    if routing["Medicine_ID"].duplicated().any():
        duplicates = (
            routing.loc[
                routing["Medicine_ID"].duplicated(
                    keep=False
                ),
                "Medicine_ID",
            ]
            .unique()
            .tolist()
        )

        raise ValueError(
            "Frozen routing table contains duplicate "
            f"Medicine_ID values: {duplicates}"
        )

    supported_models = {
        "chronos-2-P50",
        "tsb",
    }

    unsupported = set(
        routing["Selected_Model"].astype(str)
    ) - supported_models

    if unsupported:
        raise ValueError(
            "Frozen routing table contains unsupported "
            f"models: {sorted(unsupported)}"
        )

    routing["Validation_Advantage_Pct"] = pd.to_numeric(
        routing["Validation_Advantage_Pct"],
        errors="raise",
    )

    routing["Threshold"] = pd.to_numeric(
        routing["Threshold"],
        errors="raise",
    )

    if not np.isfinite(
        routing["Validation_Advantage_Pct"]
        .to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "Frozen routing table contains non-finite "
            "Validation_Advantage_Pct values."
        )

    # Frozen policy verification.
    expected_model = np.where(
        routing["Validation_Advantage_Pct"] >= 30.0,
        "chronos-2-P50",
        "tsb",
    )

    actual_model = (
        routing["Selected_Model"]
        .astype(str)
        .to_numpy()
    )

    if not np.array_equal(
        expected_model,
        actual_model,
    ):
        raise ValueError(
            "Frozen routing table violates the frozen "
            "30% routing policy."
        )

    if not np.allclose(
        routing["Threshold"].to_numpy(dtype=float),
        30.0,
    ):
        raise ValueError(
            "Frozen routing table does not use the "
            "frozen 30% threshold."
        )

    return routing.sort_values(
        "Medicine_ID"
    ).reset_index(drop=True)


# ============================================================================
# MAIN AUDIT
# ============================================================================


def main() -> None:

    # ------------------------------------------------------------------------
    # LOAD DATA
    # ------------------------------------------------------------------------

    history = load_history()

    routing = load_frozen_routing()

    print("=" * 100)
    print("PRODUCTION FORECAST SCALE AUDIT")
    print("=" * 100)
    print()

    print(
        f"History rows       : {len(history):,}"
    )

    print(
        "History medicines  : "
        f"{history['Medicine_ID'].nunique()}"
    )

    print(
        f"Routing medicines  : {len(routing)}"
    )

    chronos_count = int(
        (
            routing["Selected_Model"]
            == "chronos-2-P50"
        ).sum()
    )

    tsb_count = int(
        (
            routing["Selected_Model"]
            == "tsb"
        ).sum()
    )

    print(
        f"Chronos routed     : {chronos_count}"
    )

    print(
        f"TSB routed         : {tsb_count}"
    )

    print()

    # ------------------------------------------------------------------------
    # PRODUCTION SERVICE
    # ------------------------------------------------------------------------

    service = ProductionForecastService(
        router=ProductionForecastRouter()
    )

    rows: list[dict] = []
    failed: list[tuple[str, str]] = []

    # ------------------------------------------------------------------------
    # FORECAST EVERY ROUTED MEDICINE
    # ------------------------------------------------------------------------

    for medicine_id in routing["Medicine_ID"].astype(str):

        try:

            medicine_history = history[
                history["Medicine_ID"]
                == medicine_id
            ].copy()

            if medicine_history.empty:
                raise ValueError(
                    "No production history found."
                )

            result = service.forecast_medicine(
                medicine_id=medicine_id,
                history_df=history,
                routing_table=routing,
            )

            # ---------------------------------------------------------------
            # Recent 90-day demand scale
            # ---------------------------------------------------------------

            last_date = medicine_history[
                "timestamp"
            ].max()

            recent_start = (
                last_date
                - pd.Timedelta(days=89)
            )

            recent = medicine_history[
                medicine_history["timestamp"]
                >= recent_start
            ]

            recent_total = float(
                recent["target"].sum()
            )

            recent_mean = (
                recent_total / 90.0
            )

            # ---------------------------------------------------------------
            # Forecast statistics
            # ---------------------------------------------------------------

            predicted = np.asarray(
                result.predicted_demand,
                dtype=float,
            )

            if predicted.size == 0:
                raise ValueError(
                    "Forecast returned zero predictions."
                )

            if not np.isfinite(
                predicted
            ).all():
                raise ValueError(
                    "Forecast contains non-finite values."
                )

            if (
                predicted < 0
            ).any():
                raise ValueError(
                    "Forecast contains negative values."
                )

            forecast_mean = float(
                np.mean(predicted)
            )

            forecast_total = float(
                np.sum(predicted)
            )

            # ---------------------------------------------------------------
            # P50
            # ---------------------------------------------------------------

            if result.p50 is not None:

                p50_values = np.asarray(
                    result.p50,
                    dtype=float,
                )

                p50_mean = float(
                    np.mean(p50_values)
                )

            else:

                p50_mean = np.nan

            # ---------------------------------------------------------------
            # P90
            # ---------------------------------------------------------------

            if result.p90 is not None:

                p90_values = np.asarray(
                    result.p90,
                    dtype=float,
                )

                p90_mean = float(
                    np.mean(p90_values)
                )

            else:

                p90_mean = np.nan

            # ---------------------------------------------------------------
            # Scale ratio
            # ---------------------------------------------------------------

            if recent_mean > 0:

                scale_ratio = (
                    forecast_mean
                    / recent_mean
                )

            else:

                scale_ratio = np.nan

            # ---------------------------------------------------------------
            # Routing metadata
            # ---------------------------------------------------------------

            routing_row = routing.loc[
                routing["Medicine_ID"]
                == medicine_id
            ].iloc[0]

            rows.append(
                {
                    "Medicine_ID": medicine_id,

                    "Selected_Model": (
                        result.selected_model
                    ),

                    "Validation_Advantage_Pct": (
                        float(
                            routing_row[
                                "Validation_Advantage_Pct"
                            ]
                        )
                    ),

                    "Routing_Rule": (
                        routing_row[
                            "Routing_Rule"
                        ]
                    ),

                    "Threshold": float(
                        routing_row[
                            "Threshold"
                        ]
                    ),

                    "Recent_90D_Total": (
                        recent_total
                    ),

                    "Recent_90D_Daily_Mean": (
                        recent_mean
                    ),

                    "Forecast_30D_Total": (
                        forecast_total
                    ),

                    "Forecast_Daily_Mean": (
                        forecast_mean
                    ),

                    "P50_Daily_Mean": (
                        p50_mean
                    ),

                    "P90_Daily_Mean": (
                        p90_mean
                    ),

                    "Forecast_to_Recent_Ratio": (
                        scale_ratio
                    ),

                    "Forecast_Min": float(
                        np.min(predicted)
                    ),

                    "Forecast_Max": float(
                        np.max(predicted)
                    ),
                }
            )

        except Exception as exc:

            failed.append(
                (
                    medicine_id,
                    str(exc),
                )
            )

    # ------------------------------------------------------------------------
    # BUILD AUDIT DATAFRAME
    # ------------------------------------------------------------------------

    audit = pd.DataFrame(rows)

    if audit.empty:
        raise RuntimeError(
            "No successful production forecasts were generated."
        )

    # ------------------------------------------------------------------------
    # EXECUTION SUMMARY
    # ------------------------------------------------------------------------

    print("=" * 100)
    print("EXECUTION")
    print("=" * 100)

    successful_count = int(
        audit["Medicine_ID"].nunique()
    )

    print(
        f"Successful medicines : {successful_count}"
    )

    print(
        f"Failed medicines     : {len(failed)}"
    )

    print()

    if failed:

        print("FAILED:")

        for medicine_id, error in failed:

            print(
                f"  {medicine_id}: {error}"
            )

        print()

    # ------------------------------------------------------------------------
    # MODEL SCALE SUMMARY
    # ------------------------------------------------------------------------

    print("=" * 100)
    print("MODEL SCALE SUMMARY")
    print("=" * 100)

    summary = (
        audit
        .groupby("Selected_Model")[
            [
                "Recent_90D_Daily_Mean",
                "Forecast_Daily_Mean",
                "Forecast_to_Recent_Ratio",
            ]
        ]
        .agg(
            [
                "count",
                "mean",
                "median",
                "min",
                "max",
            ]
        )
    )

    print(
        summary.to_string()
    )

    print()

    # ------------------------------------------------------------------------
    # CHRONOS SCALE AUDIT
    # ------------------------------------------------------------------------

    print("=" * 100)
    print("CHRONOS SCALE AUDIT")
    print("=" * 100)

    chronos = audit[
        audit["Selected_Model"]
        == "chronos-2-P50"
    ].copy()

    if chronos.empty:

        print(
            "No Chronos-routed medicines."
        )

        print()

    else:

        display_columns = [
            "Medicine_ID",
            "Validation_Advantage_Pct",
            "Recent_90D_Daily_Mean",
            "Forecast_Daily_Mean",
            "P50_Daily_Mean",
            "P90_Daily_Mean",
            "Forecast_to_Recent_Ratio",
            "Forecast_30D_Total",
        ]

        print(
            chronos[
                display_columns
            ]
            .sort_values(
                "Forecast_to_Recent_Ratio"
            )
            .to_string(
                index=False
            )
        )

        print()

        ratio = chronos[
            "Forecast_to_Recent_Ratio"
        ]

        print(
            "Chronos ratio < 0.10 : "
            f"{int((ratio < 0.10).sum())}"
        )

        print(
            "Chronos ratio < 0.25 : "
            f"{int((ratio < 0.25).sum())}"
        )

        print(
            "Chronos ratio < 0.50 : "
            f"{int((ratio < 0.50).sum())}"
        )

        print(
            "Chronos ratio >= 0.50: "
            f"{int((ratio >= 0.50).sum())}"
        )

        print()

    # ------------------------------------------------------------------------
    # NEAR-ZERO FORECAST DETECTION
    # ------------------------------------------------------------------------

    print("=" * 100)
    print("NEAR-ZERO FORECAST DETECTION")
    print("=" * 100)

    near_zero = audit[
        (
            audit[
                "Recent_90D_Daily_Mean"
            ] > 0
        )
        & (
            audit[
                "Forecast_to_Recent_Ratio"
            ] < 0.10
        )
    ].copy()

    print(
        "Medicines with forecast <10% "
        "of recent demand: "
        f"{len(near_zero)}"
    )

    if not near_zero.empty:

        print()

        print(
            near_zero[
                [
                    "Medicine_ID",
                    "Selected_Model",
                    "Recent_90D_Daily_Mean",
                    "Forecast_Daily_Mean",
                    "Forecast_to_Recent_Ratio",
                ]
            ]
            .sort_values(
                "Forecast_to_Recent_Ratio"
            )
            .to_string(
                index=False
            )
        )

    print()

    # ------------------------------------------------------------------------
    # OVERALL SCALE DISTRIBUTION
    # ------------------------------------------------------------------------

    print("=" * 100)
    print("OVERALL SCALE DISTRIBUTION")
    print("=" * 100)

    valid_ratios = audit[
        "Forecast_to_Recent_Ratio"
    ].dropna()

    if not valid_ratios.empty:

        print(
            f"Median forecast/recent ratio : "
            f"{valid_ratios.median():.4f}"
        )

        print(
            f"Mean forecast/recent ratio   : "
            f"{valid_ratios.mean():.4f}"
        )

        print(
            f"Minimum ratio                : "
            f"{valid_ratios.min():.4f}"
        )

        print(
            f"Maximum ratio                : "
            f"{valid_ratios.max():.4f}"
        )

    print()

    # ------------------------------------------------------------------------
    # MODEL COUNTS IN SUCCESSFUL AUDIT
    # ------------------------------------------------------------------------

    print("=" * 100)
    print("SUCCESSFUL MODEL COUNTS")
    print("=" * 100)

    print(
        audit[
            "Selected_Model"
        ]
        .value_counts()
        .to_string()
    )

    print()

    # ------------------------------------------------------------------------
    # SAVE AUDIT
    # ------------------------------------------------------------------------

    audit.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    print("=" * 100)
    print("AUDIT ARTIFACT")
    print("=" * 100)

    print(
        OUTPUT_PATH
    )

    print()

    print("=" * 100)
    print("AUDIT COMPLETE")
    print("=" * 100)

    print(
        "The audit uses the frozen production routing table "
        "and does not perform model selection."
    )


if __name__ == "__main__":
    main()