from pathlib import Path

import numpy as np
import pandas as pd

from finemed_ai.demand_forecasting.production_forecast_router import (
    ProductionForecastRouter,
    ROUTING_RULE_NAME,
    VALIDATION_ADVANTAGE_THRESHOLD,
)


# ============================================================================
# PATHS
# ============================================================================

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
    "data/05_gold/demand_forecasting/medicine_robustness"
)

OUTPUT_MEDICINE = (
    OUTPUT_DIR / "medicine_model_robustness.parquet"
)

OUTPUT_SUMMARY = (
    OUTPUT_DIR / "medicine_model_robustness_summary.parquet"
)

OUTPUT_ROUTING = (
    OUTPUT_DIR / "production_routing_table.parquet"
)


# ============================================================================
# EVALUATION CUT-OFFS
# ============================================================================

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


# ============================================================================
# DATA LOADING
# ============================================================================


def load_data():
    """Load Chronos, TSB and regime data."""

    chronos = pd.read_parquet(
        CHRONOS_PATH
    )

    tsb = pd.read_parquet(
        CLASSICAL_PATH
    )

    regimes = pd.read_parquet(
        REGIME_PATH
    )

    # ------------------------------------------------------------------
    # Select required models
    # ------------------------------------------------------------------

    chronos = chronos[
        chronos["Model"].astype(str).str.lower()
        == "chronos-2-p50"
    ].copy()

    tsb = tsb[
        tsb["Model"].astype(str).str.lower()
        == "tsb"
    ].copy()

    # ------------------------------------------------------------------
    # Normalize dates
    # ------------------------------------------------------------------

    chronos["Cutoff_Date"] = pd.to_datetime(
        chronos["Cutoff_Date"]
    )

    tsb["Cutoff_Date"] = pd.to_datetime(
        tsb["Cutoff_Date"]
    )

    # ------------------------------------------------------------------
    # Normalize medicine IDs
    # ------------------------------------------------------------------

    chronos["Medicine_ID"] = (
        chronos["Medicine_ID"]
        .astype(str)
        .str.strip()
    )

    tsb["Medicine_ID"] = (
        tsb["Medicine_ID"]
        .astype(str)
        .str.strip()
    )

    regimes["Medicine_ID"] = (
        regimes["Medicine_ID"]
        .astype(str)
        .str.strip()
    )

    return chronos, tsb, regimes


# ============================================================================
# SPLIT ASSIGNMENT
# ============================================================================


def add_split(df):
    """Assign validation / holdout split based on cutoff date."""

    result = df.copy()

    result["Split"] = "unknown"

    result.loc[
        result["Cutoff_Date"].isin(
            VALIDATION_CUTOFFS
        ),
        "Split",
    ] = "validation"

    result.loc[
        result["Cutoff_Date"].isin(
            HOLDOUT_CUTOFFS
        ),
        "Split",
    ] = "holdout"

    return result


# ============================================================================
# MODEL ERROR AGGREGATION
# ============================================================================


def aggregate_model_errors(df, model_name):
    """
    Aggregate forecast errors by split and medicine.

    The aggregation is performed over all monthly evaluation
    cutoffs belonging to the split.
    """

    df = add_split(df)

    df = df[
        df["Split"].isin(
            [
                "validation",
                "holdout",
            ]
        )
    ].copy()

    if df.empty:
        return pd.DataFrame(
            columns=[
                "Split",
                "Medicine_ID",
                "Actual",
                f"{model_name}_Predicted",
                f"{model_name}_AE",
                f"{model_name}_WAPE",
            ]
        )

    result = (
        df.groupby(
            [
                "Split",
                "Medicine_ID",
            ],
            as_index=False,
        )
        .agg(
            Actual=(
                "Actual",
                "sum",
            ),
            Predicted=(
                "Predicted",
                "sum",
            ),
            Absolute_Error=(
                "Absolute_Error",
                "sum",
            ),
        )
    )

    # Safe WAPE calculation.
    result[f"{model_name}_WAPE"] = np.where(
        result["Actual"] != 0,
        (
            result["Absolute_Error"]
            / result["Actual"]
            * 100.0
        ),
        np.nan,
    )

    result = result.rename(
        columns={
            "Predicted": (
                f"{model_name}_Predicted"
            ),
            "Absolute_Error": (
                f"{model_name}_AE"
            ),
        }
    )

    return result


# ============================================================================
# MEDICINE COMPARISON
# ============================================================================


def build_medicine_comparison(
    chronos,
    tsb,
    regimes,
):
    """Build medicine-level Chronos vs TSB comparison."""

    chronos_result = aggregate_model_errors(
        chronos,
        "Chronos",
    )

    tsb_result = aggregate_model_errors(
        tsb,
        "TSB",
    )

    if chronos_result.empty:
        raise ValueError(
            "Chronos aggregation produced no validation/holdout rows."
        )

    if tsb_result.empty:
        raise ValueError(
            "TSB aggregation produced no validation/holdout rows."
        )

    # ------------------------------------------------------------------
    # IMPORTANT:
    #
    # Both tables contain an `Actual` column.
    # Do NOT assume merge suffixes will create Actual_Chronos.
    # Instead explicitly rename Actual before merging.
    # ------------------------------------------------------------------

    chronos_result = chronos_result.rename(
        columns={
            "Actual": "Actual_Chronos",
        }
    )

    tsb_result = tsb_result.rename(
        columns={
            "Actual": "Actual_TSB",
        }
    )

    comparison = chronos_result.merge(
        tsb_result,
        on=[
            "Split",
            "Medicine_ID",
        ],
        how="inner",
    )

    if comparison.empty:
        raise ValueError(
            "Chronos and TSB have no overlapping "
            "medicine/split combinations."
        )

    # ------------------------------------------------------------------
    # Actual demand
    #
    # Chronos and TSB should have identical actual demand.
    # Use Chronos as canonical actual and verify consistency.
    # ------------------------------------------------------------------

    actual_difference = (
        comparison["Actual_Chronos"]
        - comparison["Actual_TSB"]
    ).abs()

    inconsistent_actuals = (
        actual_difference > 1e-9
    )

    if inconsistent_actuals.any():
        count = int(
            inconsistent_actuals.sum()
        )

        raise ValueError(
            "Chronos and TSB actual demand differs "
            f"for {count} comparison rows."
        )

    comparison["Actual"] = (
        comparison["Actual_Chronos"]
    )

    # ------------------------------------------------------------------
    # Winner
    # ------------------------------------------------------------------

    comparison["Chronos_Better"] = (
        comparison["Chronos_AE"]
        < comparison["TSB_AE"]
    )

    comparison["Error_Difference"] = (
        comparison["TSB_AE"]
        - comparison["Chronos_AE"]
    )

    comparison["Winner"] = np.where(
        comparison["Chronos_Better"],
        "chronos-2-P50",
        "tsb",
    )

    # ------------------------------------------------------------------
    # Merge regime information
    # ------------------------------------------------------------------

    regime_columns = [
        "Medicine_ID",
        "Regime",
        "Days",
        "NonZero_Days",
        "ADI",
        "CV2",
        "Total_Demand",
    ]

    missing_regime_columns = (
        set(regime_columns)
        - set(regimes.columns)
    )

    if missing_regime_columns:
        raise ValueError(
            "Regime data is missing required columns: "
            f"{sorted(missing_regime_columns)}"
        )

    regime_data = (
        regimes[regime_columns]
        .drop_duplicates(
            subset=["Medicine_ID"]
        )
    )

    comparison = comparison.merge(
        regime_data,
        on="Medicine_ID",
        how="left",
    )

    return comparison


# ============================================================================
# STABILITY TABLE
# ============================================================================


def build_stability_table(comparison):
    """
    Build one row per medicine comparing validation winner
    against holdout winner.
    """

    validation = comparison[
        comparison["Split"] == "validation"
    ].copy()

    holdout = comparison[
        comparison["Split"] == "holdout"
    ].copy()

    if validation.empty:
        raise ValueError(
            "No validation rows available."
        )

    if holdout.empty:
        raise ValueError(
            "No holdout rows available."
        )

    # ------------------------------------------------------------------
    # Validation is aggregated across all validation cutoffs.
    # Therefore there should already be one row per medicine.
    # ------------------------------------------------------------------

    validation = validation[
        [
            "Medicine_ID",
            "Regime",
            "Days",
            "NonZero_Days",
            "ADI",
            "CV2",
            "Total_Demand",
            "Chronos_WAPE",
            "TSB_WAPE",
            "Chronos_AE",
            "TSB_AE",
            "Winner",
        ]
    ].rename(
        columns={
            "Chronos_WAPE": (
                "Validation_Chronos_WAPE"
            ),
            "TSB_WAPE": (
                "Validation_TSB_WAPE"
            ),
            "Chronos_AE": (
                "Validation_Chronos_AE"
            ),
            "TSB_AE": (
                "Validation_TSB_AE"
            ),
            "Winner": (
                "Validation_Winner"
            ),
        }
    )

    holdout = holdout[
        [
            "Medicine_ID",
            "Chronos_WAPE",
            "TSB_WAPE",
            "Chronos_AE",
            "TSB_AE",
            "Winner",
        ]
    ].rename(
        columns={
            "Chronos_WAPE": (
                "Holdout_Chronos_WAPE"
            ),
            "TSB_WAPE": (
                "Holdout_TSB_WAPE"
            ),
            "Chronos_AE": (
                "Holdout_Chronos_AE"
            ),
            "TSB_AE": (
                "Holdout_TSB_AE"
            ),
            "Winner": (
                "Holdout_Winner"
            ),
        }
    )

    # ------------------------------------------------------------------
    # Ensure one row per medicine in each split.
    # ------------------------------------------------------------------

    if validation["Medicine_ID"].duplicated().any():
        raise ValueError(
            "Validation table contains duplicate Medicine_ID values."
        )

    if holdout["Medicine_ID"].duplicated().any():
        raise ValueError(
            "Holdout table contains duplicate Medicine_ID values."
        )

    stability = validation.merge(
        holdout,
        on="Medicine_ID",
        how="inner",
    )

    if stability.empty:
        raise ValueError(
            "No medicines are comparable between validation and holdout."
        )

    # ------------------------------------------------------------------
    # Winner stability
    # ------------------------------------------------------------------

    stability["Winner_Stable"] = (
        stability["Validation_Winner"]
        == stability["Holdout_Winner"]
    )

    stability["Transition"] = (
        stability["Validation_Winner"]
        + " -> "
        + stability["Holdout_Winner"]
    )

    # ------------------------------------------------------------------
    # Advantage
    #
    # Positive = Chronos better.
    # Negative = TSB better.
    # ------------------------------------------------------------------

    stability["Validation_Advantage"] = (
        stability["Validation_TSB_AE"]
        - stability["Validation_Chronos_AE"]
    )

    stability["Holdout_Advantage"] = (
        stability["Holdout_TSB_AE"]
        - stability["Holdout_Chronos_AE"]
    )

    stability["Absolute_Validation_Advantage"] = (
        stability["Validation_Advantage"]
        .abs()
    )

    return stability


# ============================================================================
# PRINT FUNCTIONS
# ============================================================================


def print_split_summary(
    comparison,
    split,
):
    """Print aggregate model comparison for a split."""

    data = comparison[
        comparison["Split"] == split
    ].copy()

    if data.empty:
        return

    actual = data["Actual"].sum()

    chronos_error = (
        data["Chronos_AE"].sum()
    )

    tsb_error = (
        data["TSB_AE"].sum()
    )

    chronos_wape = (
        chronos_error / actual * 100.0
        if actual != 0
        else np.nan
    )

    tsb_wape = (
        tsb_error / actual * 100.0
        if actual != 0
        else np.nan
    )

    print()
    print(
        f"{split.upper()} MODEL COMPARISON"
    )
    print("-" * 80)

    print(
        f"Medicines: "
        f"{data['Medicine_ID'].nunique()}"
    )

    print(
        f"Chronos WAPE: "
        f"{chronos_wape:.3f}"
    )

    print(
        f"TSB WAPE: "
        f"{tsb_wape:.3f}"
    )

    print(
        "Chronos better: "
        f"{int(data['Chronos_Better'].sum())}"
    )

    print(
        "TSB better: "
        f"{int((~data['Chronos_Better']).sum())}"
    )


def print_stability_summary(
    stability,
):
    """Print overall winner stability."""

    print()
    print("=" * 80)
    print("MEDICINE-LEVEL WINNER STABILITY")
    print("=" * 80)

    total = len(stability)

    stable = int(
        stability["Winner_Stable"].sum()
    )

    stability_rate = (
        stable / total * 100.0
        if total
        else 0.0
    )

    print()
    print(
        f"Comparable medicines: {total}"
    )

    print(
        f"Stable winners: {stable}"
    )

    print(
        f"Winner stability: "
        f"{stability_rate:.2f}%"
    )

    print()
    print("Winner transitions:")

    print(
        stability["Transition"]
        .value_counts()
        .to_string()
    )

    print()
    print("Stable validation winners:")

    stable_data = stability[
        stability["Winner_Stable"]
    ]

    if stable_data.empty:
        print("None")
    else:
        print(
            stable_data["Validation_Winner"]
            .value_counts()
            .to_string()
        )


def print_regime_stability(
    stability,
):
    """Print winner stability by demand regime."""

    print()
    print("=" * 80)
    print("STABILITY BY REGIME")
    print("=" * 80)

    result = (
        stability
        .groupby("Regime", dropna=False)
        .agg(
            Medicines=(
                "Medicine_ID",
                "nunique",
            ),
            Stable=(
                "Winner_Stable",
                "sum",
            ),
        )
    )

    result["Stability_Percentage"] = np.where(
        result["Medicines"] != 0,
        result["Stable"]
        / result["Medicines"]
        * 100.0,
        np.nan,
    )

    print(
        result
        .round(2)
        .to_string()
    )


def print_volume_analysis(
    stability,
):
    """Print winner stability by demand-volume quartile."""

    print()
    print("=" * 80)
    print("DEMAND VOLUME ANALYSIS")
    print("=" * 80)

    data = stability.copy()

    if data["Total_Demand"].nunique() < 2:
        print(
            "Insufficient demand variation for quartile analysis."
        )
        return

    try:
        data["Demand_Bucket"] = pd.qcut(
            data["Total_Demand"],
            q=4,
            labels=[
                "Q1_Low",
                "Q2",
                "Q3",
                "Q4_High",
            ],
            duplicates="drop",
        )
    except ValueError:
        print(
            "Unable to construct demand quartiles."
        )
        return

    result = (
        data
        .groupby(
            "Demand_Bucket",
            observed=True,
        )
        .agg(
            Medicines=(
                "Medicine_ID",
                "nunique",
            ),
            Stable=(
                "Winner_Stable",
                "sum",
            ),
            Avg_Demand=(
                "Total_Demand",
                "mean",
            ),
        )
    )

    result["Stability_Percentage"] = np.where(
        result["Medicines"] != 0,
        result["Stable"]
        / result["Medicines"]
        * 100.0,
        np.nan,
    )

    print(
        result
        .round(2)
        .to_string()
    )


def print_largest_advantages(
    stability,
):
    """Print largest stable advantages for both models."""

    print()
    print("=" * 80)
    print("LARGEST STABLE CHRONOS ADVANTAGES")
    print("=" * 80)

    chronos = stability[
        (
            stability["Validation_Winner"]
            == "chronos-2-P50"
        )
        & (
            stability["Holdout_Winner"]
            == "chronos-2-P50"
        )
    ].copy()

    chronos = chronos.sort_values(
        "Holdout_Advantage",
        ascending=False,
    )

    columns = [
        "Medicine_ID",
        "Regime",
        "Total_Demand",
        "ADI",
        "CV2",
        "Validation_Advantage",
        "Holdout_Advantage",
    ]

    if chronos.empty:
        print("None")
    else:
        print(
            chronos[columns]
            .head(20)
            .round(3)
            .to_string(index=False)
        )

    print()
    print("=" * 80)
    print("LARGEST STABLE TSB ADVANTAGES")
    print("=" * 80)

    tsb = stability[
        (
            stability["Validation_Winner"]
            == "tsb"
        )
        & (
            stability["Holdout_Winner"]
            == "tsb"
        )
    ].copy()

    tsb = tsb.sort_values(
        "Holdout_Advantage",
        ascending=True,
    )

    if tsb.empty:
        print("None")
    else:
        print(
            tsb[columns]
            .head(20)
            .round(3)
            .to_string(index=False)
        )


# ============================================================================
# PRODUCTION ROUTING
# ============================================================================


def build_production_routing(
    comparison,
):
    """
    Build the production routing artifact.

    CRITICAL:
    Only validation performance is used for model selection.

    Holdout performance is NEVER used here.
    """

    validation = comparison[
        comparison["Split"] == "validation"
    ].copy()

    if validation.empty:
        raise ValueError(
            "No validation rows available for production routing."
        )

    # ------------------------------------------------------------------
    # We aggregated validation across all validation cutoffs.
    # Therefore exactly one row per medicine is expected.
    # ------------------------------------------------------------------

    if validation["Medicine_ID"].duplicated().any():
        duplicates = (
            validation.loc[
                validation["Medicine_ID"].duplicated(
                    keep=False
                ),
                "Medicine_ID",
            ]
            .astype(str)
            .unique()
            .tolist()
        )

        raise ValueError(
            "Validation results contain duplicate "
            f"Medicine_ID values: {duplicates[:20]}"
        )

    required_columns = {
        "Medicine_ID",
        "Chronos_AE",
        "TSB_AE",
    }

    missing = (
        required_columns
        - set(validation.columns)
    )

    if missing:
        raise ValueError(
            "Validation results are missing required "
            f"columns: {sorted(missing)}"
        )

    routing = validation[
        [
            "Medicine_ID",
            "Chronos_AE",
            "TSB_AE",
        ]
    ].copy()

    routing = routing.rename(
        columns={
            "Chronos_AE": (
                "Validation_Chronos_AE"
            ),
            "TSB_AE": (
                "Validation_TSB_AE"
            ),
        }
    )

    # ------------------------------------------------------------------
    # Chronos advantage over TSB.
    #
    # Positive:
    #   Chronos has lower error.
    #
    # Negative:
    #   TSB has lower error.
    # ------------------------------------------------------------------

    routing["Validation_Advantage_Pct"] = np.where(
        routing["Validation_TSB_AE"] != 0,
        (
            (
                routing["Validation_TSB_AE"]
                - routing["Validation_Chronos_AE"]
            )
            / routing["Validation_TSB_AE"]
            * 100.0
        ),
        np.nan,
    )

    # ------------------------------------------------------------------
    # Production model selection.
    #
    # ONLY validation advantage is passed to the router.
    # ------------------------------------------------------------------

    routing["Selected_Model"] = (
        routing["Validation_Advantage_Pct"]
        .apply(
            ProductionForecastRouter.select_model
        )
    )

    routing["Routing_Rule"] = (
        ROUTING_RULE_NAME
    )

    routing["Threshold"] = (
        VALIDATION_ADVANTAGE_THRESHOLD
    )

    # ------------------------------------------------------------------
    # Remove invalid records.
    # ------------------------------------------------------------------

    routing = routing.dropna(
        subset=[
            "Validation_Chronos_AE",
            "Validation_TSB_AE",
            "Validation_Advantage_Pct",
            "Selected_Model",
        ]
    ).copy()

    # ------------------------------------------------------------------
    # Safety checks.
    # ------------------------------------------------------------------

    if routing.empty:
        raise ValueError(
            "Production routing table is empty after "
            "removing invalid validation records."
        )

    if routing["Medicine_ID"].duplicated().any():
        raise ValueError(
            "Production routing table contains duplicate "
            "Medicine_ID values."
        )

    routing = (
        routing
        .sort_values("Medicine_ID")
        .reset_index(drop=True)
    )

    return routing


# ============================================================================
# SAVE STABILITY SUMMARY
# ============================================================================


def build_stability_summary(
    stability,
):
    """Build stability summary for audit."""

    return (
        stability
        .groupby(
            "Transition",
            as_index=False,
        )
        .agg(
            Medicines=(
                "Medicine_ID",
                "nunique",
            ),
            Mean_Validation_Advantage=(
                "Validation_Advantage",
                "mean",
            ),
            Mean_Holdout_Advantage=(
                "Holdout_Advantage",
                "mean",
            ),
        )
    )


# ============================================================================
# MAIN
# ============================================================================


def main():
    """Run the complete medicine robustness analysis."""

    print("=" * 80)
    print("MEDICINE MODEL ROBUSTNESS ANALYSIS")
    print("=" * 80)

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    chronos, tsb, regimes = load_data()

    print()
    print(
        "Chronos rows:",
        len(chronos),
    )

    print(
        "TSB rows:",
        len(tsb),
    )

    print(
        "Regime medicines:",
        len(regimes),
    )

    # ------------------------------------------------------------------
    # Compare models
    # ------------------------------------------------------------------

    comparison = build_medicine_comparison(
        chronos,
        tsb,
        regimes,
    )

    print()
    print(
        "Comparison rows:",
        len(comparison),
    )

    print(
        "Medicines:",
        comparison["Medicine_ID"].nunique(),
    )

    # ------------------------------------------------------------------
    # Split summaries
    # ------------------------------------------------------------------

    print_split_summary(
        comparison,
        "validation",
    )

    print_split_summary(
        comparison,
        "holdout",
    )

    # ------------------------------------------------------------------
    # Stability
    # ------------------------------------------------------------------

    stability = build_stability_table(
        comparison
    )

    print_stability_summary(
        stability
    )

    print_regime_stability(
        stability
    )

    print_volume_analysis(
        stability
    )

    print_largest_advantages(
        stability
    )

    # ------------------------------------------------------------------
    # Output directory
    # ------------------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------------------
    # Save complete medicine comparison.
    # ------------------------------------------------------------------

    comparison.to_parquet(
        OUTPUT_MEDICINE,
        index=False,
    )

    # ------------------------------------------------------------------
    # Build production routing.
    #
    # IMPORTANT:
    # This function uses validation ONLY.
    # ------------------------------------------------------------------

    routing = build_production_routing(
        comparison
    )

    routing.to_parquet(
        OUTPUT_ROUTING,
        index=False,
    )

    # ------------------------------------------------------------------
    # Save stability summary.
    #
    # This is an evaluation/audit artifact.
    # It does NOT control production model selection.
    # ------------------------------------------------------------------

    stability_summary = (
        build_stability_summary(
            stability
        )
    )

    stability_summary.to_parquet(
        OUTPUT_SUMMARY,
        index=False,
    )

    # ------------------------------------------------------------------
    # Production routing report
    # ------------------------------------------------------------------

    chronos_routes = int(
        (
            routing["Selected_Model"]
            == "chronos-2-P50"
        ).sum()
    )

    tsb_routes = int(
        (
            routing["Selected_Model"]
            == "tsb"
        ).sum()
    )

    invalid_advantage = int(
        routing[
            "Validation_Advantage_Pct"
        ]
        .isna()
        .sum()
    )

    print()
    print("=" * 80)
    print("PRODUCTION ROUTING TABLE")
    print("=" * 80)

    print(
        f"Routing medicines: "
        f"{len(routing)}"
    )

    print(
        f"Chronos routes: "
        f"{chronos_routes}"
    )

    print(
        f"TSB routes: "
        f"{tsb_routes}"
    )

    print(
        f"Invalid/missing advantage: "
        f"{invalid_advantage}"
    )

    print()
    print("Routing rule:")
    print(
        f"  {ROUTING_RULE_NAME}"
    )

    print(
        "Validation advantage threshold:"
    )
    print(
        f"  {VALIDATION_ADVANTAGE_THRESHOLD}"
    )

    print()
    print("Saved:")
    print(
        f"  {OUTPUT_MEDICINE}"
    )
    print(
        f"  {OUTPUT_ROUTING}"
    )
    print(
        f"  {OUTPUT_SUMMARY}"
    )

    print()
    print("=" * 80)
    print("MEDICINE MODEL ROBUSTNESS ANALYSIS COMPLETE")
    print("=" * 80)


# ============================================================================
# ENTRY POINT
# ============================================================================


if __name__ == "__main__":
    main()