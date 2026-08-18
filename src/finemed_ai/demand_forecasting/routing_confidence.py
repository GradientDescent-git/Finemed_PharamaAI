import pandas as pd
from pathlib import Path


INPUT = Path(
    "data/05_gold/demand_forecasting/"
    "medicine_robustness/"
    "medicine_model_robustness.parquet"
)

OUTPUT_DIR = Path(
    "data/05_gold/demand_forecasting/"
    "routing_confidence"
)

OUTPUT = OUTPUT_DIR / "routing_confidence.parquet"


def main():

    print("=" * 80)
    print("ROUTING CONFIDENCE ANALYSIS")
    print("=" * 80)

    df = pd.read_parquet(INPUT)

    print()
    print("Input rows:", len(df))
    print("Medicines:", df["Medicine_ID"].nunique())

    # ------------------------------------------------------------------
    # Split into validation and holdout
    # ------------------------------------------------------------------

    validation = df[
        df["Split"] == "validation"
    ].copy()

    holdout = df[
        df["Split"] == "holdout"
    ].copy()

    print("Validation rows:", len(validation))
    print("Holdout rows:", len(holdout))

    # ------------------------------------------------------------------
    # Keep one medicine-level record for each split
    # ------------------------------------------------------------------

    validation = validation[
        [
            "Medicine_ID",
            "Winner",
            "Chronos_WAPE",
            "TSB_WAPE",
            "Error_Difference",
        ]
    ].rename(
        columns={
            "Winner": "Validation_Winner",
            "Chronos_WAPE": "Validation_Chronos_WAPE",
            "TSB_WAPE": "Validation_TSB_WAPE",
            "Error_Difference": "Validation_Advantage",
        }
    )

    holdout = holdout[
        [
            "Medicine_ID",
            "Winner",
            "Chronos_WAPE",
            "TSB_WAPE",
            "Error_Difference",
        ]
    ].rename(
        columns={
            "Winner": "Holdout_Winner",
            "Chronos_WAPE": "Holdout_Chronos_WAPE",
            "TSB_WAPE": "Holdout_TSB_WAPE",
            "Error_Difference": "Holdout_Advantage",
        }
    )

    # ------------------------------------------------------------------
    # Merge validation and holdout
    # ------------------------------------------------------------------

    comparison = validation.merge(
        holdout,
        on="Medicine_ID",
        how="inner",
    )

    # ------------------------------------------------------------------
    # Add medicine characteristics
    # ------------------------------------------------------------------

    medicine_features = (
        df[
            [
                "Medicine_ID",
                "Regime",
                "Days",
                "NonZero_Days",
                "ADI",
                "CV2",
                "Total_Demand",
            ]
        ]
        .drop_duplicates(
            "Medicine_ID"
        )
    )

    comparison = comparison.merge(
        medicine_features,
        on="Medicine_ID",
        how="left",
    )

    print()
    print("Comparable medicines:", len(comparison))

    # ------------------------------------------------------------------
    # Winner stability
    # ------------------------------------------------------------------

    comparison["Winner_Stable"] = (
        comparison["Validation_Winner"]
        ==
        comparison["Holdout_Winner"]
    )

    comparison["Winner_Transition"] = (
        comparison["Validation_Winner"]
        + " -> "
        + comparison["Holdout_Winner"]
    )

    # ------------------------------------------------------------------
    # Winner margins
    #
    # Positive advantage = Chronos better
    # Negative advantage = TSB better
    # ------------------------------------------------------------------

    comparison["Validation_Winner_Margin"] = (
        comparison[
            "Validation_Advantage"
        ].abs()
    )

    comparison["Holdout_Winner_Margin"] = (
        comparison[
            "Holdout_Advantage"
        ].abs()
    )

    # ------------------------------------------------------------------
    # Confidence
    #
    # HIGH:
    #   Same winner in validation and holdout
    #   AND validation margin >= 100 WAPE points
    #
    # MEDIUM:
    #   Same winner but smaller margin
    #
    # LOW:
    #   Winner changes between validation and holdout
    # ------------------------------------------------------------------

    def confidence(row):

        if not row["Winner_Stable"]:
            return "LOW"

        if row["Validation_Winner_Margin"] >= 100:
            return "HIGH"

        return "MEDIUM"

    comparison["Confidence"] = comparison.apply(
        confidence,
        axis=1,
    )

    # ------------------------------------------------------------------
    # Regime-based routing rule
    # ------------------------------------------------------------------

    def regime_model(regime):

        if regime == "Intermittent":
            return "tsb"

        if regime == "Lumpy":
            return "chronos-2-P50"

        return "fallback"

    comparison["Regime_Model"] = (
        comparison["Regime"]
        .apply(regime_model)
    )

    comparison["Regime_Rule_Correct"] = (
        comparison["Regime_Model"]
        ==
        comparison["Holdout_Winner"]
    )

    # ------------------------------------------------------------------
    # Evidence-based production recommendation
    # ------------------------------------------------------------------

    def production_model(row):

        if row["Confidence"] == "LOW":
            return "fallback"

        if row["Validation_Winner"] == "chronos-2-P50":
            return "chronos-2-P50"

        return "tsb"

    comparison["Production_Recommendation"] = (
        comparison.apply(
            production_model,
            axis=1,
        )
    )

    # ------------------------------------------------------------------
    # Confidence distribution
    # ------------------------------------------------------------------

    print()
    print("=" * 80)
    print("CONFIDENCE DISTRIBUTION")
    print("=" * 80)

    print(
        comparison[
            "Confidence"
        ]
        .value_counts()
        .to_string()
    )

    # ------------------------------------------------------------------
    # Production recommendations
    # ------------------------------------------------------------------

    print()
    print("=" * 80)
    print("PRODUCTION RECOMMENDATIONS")
    print("=" * 80)

    print(
        comparison[
            "Production_Recommendation"
        ]
        .value_counts()
        .to_string()
    )

    # ------------------------------------------------------------------
    # Winner stability
    # ------------------------------------------------------------------

    print()
    print("=" * 80)
    print("WINNER STABILITY")
    print("=" * 80)

    stable = comparison[
        "Winner_Stable"
    ].sum()

    total = len(comparison)

    print("Comparable medicines:", total)
    print("Stable winners:", stable)

    if total:
        print(
            "Winner stability:",
            round(
                stable / total * 100,
                2,
            ),
            "%",
        )

    # ------------------------------------------------------------------
    # Winner transitions
    # ------------------------------------------------------------------

    print()
    print("=" * 80)
    print("WINNER TRANSITIONS")
    print("=" * 80)

    print(
        comparison[
            "Winner_Transition"
        ]
        .value_counts()
        .to_string()
    )

    # ------------------------------------------------------------------
    # Stability by regime
    # ------------------------------------------------------------------

    print()
    print("=" * 80)
    print("STABILITY BY REGIME")
    print("=" * 80)

    regime_summary = (
        comparison
        .groupby("Regime")
        .agg(
            Medicines=(
                "Medicine_ID",
                "nunique",
            ),
            Stable=(
                "Winner_Stable",
                "sum",
            ),
            Regime_Rule_Correct=(
                "Regime_Rule_Correct",
                "sum",
            ),
        )
    )

    regime_summary[
        "Stability_Percentage"
    ] = (
        regime_summary["Stable"]
        /
        regime_summary["Medicines"]
        * 100
    )

    regime_summary[
        "Regime_Rule_Accuracy"
    ] = (
        regime_summary[
            "Regime_Rule_Correct"
        ]
        /
        regime_summary["Medicines"]
        * 100
    )

    print(
        regime_summary
        .round(2)
        .to_string()
    )

    # ------------------------------------------------------------------
    # Confidence by regime
    # ------------------------------------------------------------------

    print()
    print("=" * 80)
    print("CONFIDENCE BY REGIME")
    print("=" * 80)

    confidence_regime = (
        comparison
        .groupby(
            [
                "Regime",
                "Confidence",
            ]
        )
        .agg(
            Medicines=(
                "Medicine_ID",
                "nunique",
            )
        )
        .reset_index()
    )

    print(
        confidence_regime
        .to_string(index=False)
    )

    # ------------------------------------------------------------------
    # High confidence medicines
    # ------------------------------------------------------------------

    print()
    print("=" * 80)
    print("HIGH-CONFIDENCE MEDICINES")
    print("=" * 80)

    high = comparison[
        comparison["Confidence"] == "HIGH"
    ].copy()

    print(
        high[
            [
                "Medicine_ID",
                "Regime",
                "Total_Demand",
                "ADI",
                "CV2",
                "Validation_Winner",
                "Holdout_Winner",
                "Validation_Winner_Margin",
                "Holdout_Winner_Margin",
                "Production_Recommendation",
            ]
        ]
        .sort_values(
            "Total_Demand",
            ascending=False,
        )
        .round(3)
        .to_string(index=False)
    )

    # ------------------------------------------------------------------
    # Low confidence medicines
    # ------------------------------------------------------------------

    print()
    print("=" * 80)
    print("LOW-CONFIDENCE MEDICINES")
    print("=" * 80)

    low = comparison[
        comparison["Confidence"] == "LOW"
    ].copy()

    print(
        low[
            [
                "Medicine_ID",
                "Regime",
                "Total_Demand",
                "ADI",
                "CV2",
                "Validation_Winner",
                "Holdout_Winner",
                "Validation_Winner_Margin",
                "Holdout_Winner_Margin",
                "Production_Recommendation",
            ]
        ]
        .sort_values(
            "Total_Demand",
            ascending=False,
        )
        .round(3)
        .to_string(index=False)
    )

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    comparison.to_parquet(
        OUTPUT,
        index=False,
    )

    print()
    print("=" * 80)
    print("SAVED")
    print("=" * 80)

    print(OUTPUT)


if __name__ == "__main__":
    main()