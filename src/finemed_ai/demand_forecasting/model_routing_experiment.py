from pathlib import Path

import numpy as np
import pandas as pd


CHRONOS_PATH = Path(
    "data/05_gold/demand_forecasting/monthly_experiment/"
    "chronos_monthly_backtest.parquet"
)

CLASSICAL_PATH = Path(
    "data/05_gold/demand_forecasting/monthly_experiment/"
    "classical_monthly_backtest.parquet"
)

REGIME_PATH = Path(
    "data/05_gold/demand_forecasting/regime_analysis/"
    "medicine_regimes.parquet"
)

OUTPUT_DIR = Path(
    "data/05_gold/demand_forecasting/model_routing"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_RESULTS = (
    OUTPUT_DIR / "model_routing_results.parquet"
)

OUTPUT_ROUTING = (
    OUTPUT_DIR / "model_routing_rules.parquet"
)


VALIDATION_CUTOFFS = pd.to_datetime(
    [
        "2025-11-01",
        "2025-12-01",
        "2026-01-01",
        "2026-02-01",
    ]
)

HOLDOUT_CUTOFFS = pd.to_datetime(
    [
        "2026-03-01",
        "2026-04-01",
    ]
)


def evaluate(df):
    grouped = (
        df.groupby("Model")
        .agg(
            Medicines=("Medicine_ID", "nunique"),
            Actual=("Actual", "sum"),
            Predicted=("Predicted", "sum"),
            Absolute_Error=("Absolute_Error", "sum"),
        )
    )

    grouped["WAPE"] = (
        grouped["Absolute_Error"]
        / grouped["Actual"]
        * 100
    )

    grouped["Ratio"] = (
        grouped["Predicted"]
        / grouped["Actual"]
    )

    grouped["MBE"] = (
        grouped["Predicted"]
        - grouped["Actual"]
    )

    return grouped.sort_values("WAPE")


def main():

    print("=" * 80)
    print("MODEL ROUTING EXPERIMENT")
    print("=" * 80)

    chronos = pd.read_parquet(
        CHRONOS_PATH
    )

    classical = pd.read_parquet(
        CLASSICAL_PATH
    )

    regimes = pd.read_parquet(
        REGIME_PATH
    )

    print()
    print("Chronos rows:", len(chronos))
    print("Classical rows:", len(classical))
    print("Regime medicines:", len(regimes))

    # ------------------------------------------------------------------
    # Select models
    # ------------------------------------------------------------------

    chronos = chronos[
        chronos["Model"] == "chronos-2-P50"
    ].copy()

    tsb = classical[
        classical["Model"] == "tsb"
    ].copy()

    chronos["Cutoff_Date"] = pd.to_datetime(
        chronos["Cutoff_Date"]
    )

    tsb["Cutoff_Date"] = pd.to_datetime(
        tsb["Cutoff_Date"]
    )

    chronos["Medicine_ID"] = (
        chronos["Medicine_ID"]
        .astype(str)
    )

    tsb["Medicine_ID"] = (
        tsb["Medicine_ID"]
        .astype(str)
    )

    regimes["Medicine_ID"] = (
        regimes["Medicine_ID"]
        .astype(str)
    )

    # ------------------------------------------------------------------
    # Merge model predictions
    # ------------------------------------------------------------------

    chronos = chronos[
        [
            "Cutoff_Date",
            "Forecast_Month",
            "Medicine_ID",
            "Actual",
            "Predicted",
            "Absolute_Error",
        ]
    ].rename(
        columns={
            "Predicted": "Chronos_P50",
            "Absolute_Error": "Chronos_AE",
        }
    )

    tsb = tsb[
        [
            "Cutoff_Date",
            "Forecast_Month",
            "Medicine_ID",
            "Actual",
            "Predicted",
            "Absolute_Error",
        ]
    ].rename(
        columns={
            "Predicted": "TSB",
            "Absolute_Error": "TSB_AE",
        }
    )

    merged = chronos.merge(
        tsb,
        on=[
            "Cutoff_Date",
            "Forecast_Month",
            "Medicine_ID",
        ],
        how="inner",
        suffixes=("", "_TSB"),
    )

    merged["Actual"] = merged["Actual"].fillna(
        merged["Actual_TSB"]
    )

    merged = merged.drop(
        columns=["Actual_TSB"]
    )

    merged = merged.merge(
        regimes[
            [
                "Medicine_ID",
                "Regime",
                "ADI",
                "CV2",
                "Total_Demand",
            ]
        ],
        on="Medicine_ID",
        how="left",
    )

    print()
    print("Merged comparison rows:", len(merged))
    print(
        "Medicines:",
        merged["Medicine_ID"].nunique(),
    )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    validation = merged[
        merged["Cutoff_Date"].isin(
            VALIDATION_CUTOFFS
        )
    ].copy()

    print()
    print("=" * 80)
    print("VALIDATION")
    print("=" * 80)

    # Model performance by regime
    regime_validation = (
        validation
        .groupby(["Regime", "Medicine_ID"])
        .agg(
            Actual=("Actual", "sum"),
            Chronos_AE=("Chronos_AE", "sum"),
            TSB_AE=("TSB_AE", "sum"),
        )
        .reset_index()
    )

    regime_summary = (
        regime_validation
        .groupby("Regime")
        .agg(
            Medicines=("Medicine_ID", "nunique"),
            Actual=("Actual", "sum"),
            Chronos_AE=("Chronos_AE", "sum"),
            TSB_AE=("TSB_AE", "sum"),
        )
    )

    regime_summary["Chronos_WAPE"] = (
        regime_summary["Chronos_AE"]
        / regime_summary["Actual"]
        * 100
    )

    regime_summary["TSB_WAPE"] = (
        regime_summary["TSB_AE"]
        / regime_summary["Actual"]
        * 100
    )

    regime_summary["Winner"] = np.where(
        regime_summary["Chronos_WAPE"]
        < regime_summary["TSB_WAPE"],
        "chronos-2-P50",
        "tsb",
    )

    print()
    print("Validation performance by regime:")
    print(
        regime_summary.round(3).to_string()
    )

    # ------------------------------------------------------------------
    # Freeze routing rule
    # ------------------------------------------------------------------

    routing_rules = (
        regime_summary
        .reset_index()[
            [
                "Regime",
                "Winner",
                "Medicines",
                "Chronos_WAPE",
                "TSB_WAPE",
            ]
        ]
        .rename(
            columns={
                "Winner": "Selected_Model"
            }
        )
    )

    print()
    print("=" * 80)
    print("FROZEN ROUTING RULE")
    print("=" * 80)

    print(
        routing_rules.to_string(
            index=False
        )
    )

    # ------------------------------------------------------------------
    # Apply frozen routing to all periods
    # ------------------------------------------------------------------

    merged = merged.merge(
        routing_rules[
            [
                "Regime",
                "Selected_Model",
            ]
        ],
        on="Regime",
        how="left",
    )

    merged["Routed_Predicted"] = np.where(
        merged["Selected_Model"]
        == "chronos-2-P50",
        merged["Chronos_P50"],
        merged["TSB"],
    )

    merged["Routed_Absolute_Error"] = (
        abs(
            merged["Actual"]
            - merged["Routed_Predicted"]
        )
    )

    # ------------------------------------------------------------------
    # Validation routing performance
    # ------------------------------------------------------------------

    validation_routed = merged[
        merged["Cutoff_Date"].isin(
            VALIDATION_CUTOFFS
        )
    ].copy()

    validation_routed["Model"] = "routed_hybrid"

    validation_tsb = validation_routed.copy()
    validation_tsb["Predicted"] = (
        validation_tsb["TSB"]
    )
    validation_tsb["Absolute_Error"] = (
        validation_tsb["TSB_AE"]
    )
    validation_tsb["Model"] = "tsb"

    validation_chronos = validation_routed.copy()
    validation_chronos["Predicted"] = (
        validation_chronos["Chronos_P50"]
    )
    validation_chronos["Absolute_Error"] = (
        validation_chronos["Chronos_AE"]
    )
    validation_chronos["Model"] = "chronos-2-P50"

    validation_hybrid = validation_routed.copy()
    validation_hybrid["Predicted"] = (
        validation_hybrid["Routed_Predicted"]
    )
    validation_hybrid["Absolute_Error"] = (
        validation_hybrid["Routed_Absolute_Error"]
    )
    validation_hybrid["Model"] = "routed_hybrid"

    validation_comparison = pd.concat(
        [
            validation_tsb,
            validation_chronos,
            validation_hybrid,
        ],
        ignore_index=True,
    )

    print()
    print("=" * 80)
    print("VALIDATION MODEL COMPARISON")
    print("=" * 80)

    print(
        evaluate(
            validation_comparison
        ).round(3).to_string()
    )

    # ------------------------------------------------------------------
    # Holdout
    # ------------------------------------------------------------------

    holdout = merged[
        merged["Cutoff_Date"].isin(
            HOLDOUT_CUTOFFS
        )
    ].copy()

    holdout_tsb = holdout.copy()
    holdout_tsb["Predicted"] = (
        holdout_tsb["TSB"]
    )
    holdout_tsb["Absolute_Error"] = (
        holdout_tsb["TSB_AE"]
    )
    holdout_tsb["Model"] = "tsb"

    holdout_chronos = holdout.copy()
    holdout_chronos["Predicted"] = (
        holdout_chronos["Chronos_P50"]
    )
    holdout_chronos["Absolute_Error"] = (
        holdout_chronos["Chronos_AE"]
    )
    holdout_chronos["Model"] = "chronos-2-P50"

    holdout_hybrid = holdout.copy()
    holdout_hybrid["Predicted"] = (
        holdout_hybrid["Routed_Predicted"]
    )
    holdout_hybrid["Absolute_Error"] = (
        holdout_hybrid["Routed_Absolute_Error"]
    )
    holdout_hybrid["Model"] = "routed_hybrid"

    holdout_comparison = pd.concat(
        [
            holdout_tsb,
            holdout_chronos,
            holdout_hybrid,
        ],
        ignore_index=True,
    )

    print()
    print("=" * 80)
    print("HOLDOUT MODEL COMPARISON")
    print("=" * 80)

    holdout_metrics = evaluate(
        holdout_comparison
    )

    print(
        holdout_metrics.round(3).to_string()
    )

    # ------------------------------------------------------------------
    # Medicine-level holdout comparison
    # ------------------------------------------------------------------

    medicine_holdout = (
        holdout
        .groupby("Medicine_ID")
        .agg(
            Actual=("Actual", "sum"),
            TSB_AE=("TSB_AE", "sum"),
            Chronos_AE=("Chronos_AE", "sum"),
            Routed_AE=(
                "Routed_Absolute_Error",
                "sum",
            ),
        )
        .reset_index()
    )

    medicine_holdout["Routed_Better_Than_TSB"] = (
        medicine_holdout["Routed_AE"]
        < medicine_holdout["TSB_AE"]
    )

    medicine_holdout["Routed_Better_Than_Chronos"] = (
        medicine_holdout["Routed_AE"]
        < medicine_holdout["Chronos_AE"]
    )

    print()
    print("=" * 80)
    print("HOLDOUT MEDICINE-LEVEL RESULTS")
    print("=" * 80)

    print(
        "Hybrid better than TSB:",
        int(
            medicine_holdout[
                "Routed_Better_Than_TSB"
            ].sum()
        ),
        "/",
        len(medicine_holdout),
    )

    print(
        "Hybrid better than Chronos:",
        int(
            medicine_holdout[
                "Routed_Better_Than_Chronos"
            ].sum()
        ),
        "/",
        len(medicine_holdout),
    )

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    merged.to_parquet(
        OUTPUT_RESULTS,
        index=False,
    )

    routing_rules.to_parquet(
        OUTPUT_ROUTING,
        index=False,
    )

    print()
    print("=" * 80)
    print("SAVED")
    print("=" * 80)

    print(OUTPUT_RESULTS)
    print(OUTPUT_ROUTING)


if __name__ == "__main__":
    main()