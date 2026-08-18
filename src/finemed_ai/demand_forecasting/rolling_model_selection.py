import pandas as pd
import numpy as np
from pathlib import Path


INPUT = Path(
    "data/05_gold/demand_forecasting/"
    "medicine_robustness/medicine_model_robustness.parquet"
)

OUTPUT_DIR = Path(
    "data/05_gold/demand_forecasting/rolling_model_selection"
)

RESULTS_OUTPUT = OUTPUT_DIR / "rolling_model_selection.parquet"
SUMMARY_OUTPUT = OUTPUT_DIR / "rolling_model_selection_summary.parquet"


def wape(actual, predicted):
    denominator = np.abs(actual).sum()

    if denominator == 0:
        return np.nan

    return np.abs(actual - predicted).sum() / denominator * 100


def main():

    print("=" * 80)
    print("ROLLING MODEL SELECTION")
    print("=" * 80)

    df = pd.read_parquet(INPUT)

    print()
    print("Input rows:", len(df))
    print("Medicines:", df["Medicine_ID"].nunique())

    # ------------------------------------------------------------------
    # The robustness dataset contains validation and holdout results.
    # We use the medicine-level winner transitions to test whether
    # historical model choice remains stable.
    # ------------------------------------------------------------------

    validation = df[df["Split"] == "validation"].copy()
    holdout = df[df["Split"] == "holdout"].copy()

    validation = validation[
        [
            "Medicine_ID",
            "Chronos_AE",
            "TSB_AE",
            "Actual",
            "Regime",
            "Total_Demand",
            "ADI",
            "CV2",
        ]
    ].copy()

    holdout = holdout[
        [
            "Medicine_ID",
            "Chronos_AE",
            "TSB_AE",
            "Actual",
            "Regime",
            "Total_Demand",
            "ADI",
            "CV2",
        ]
    ].copy()

    validation["Validation_Winner"] = np.where(
        validation["Chronos_AE"] <= validation["TSB_AE"],
        "chronos-2-P50",
        "tsb",
    )

    holdout["Holdout_Winner"] = np.where(
        holdout["Chronos_AE"] <= holdout["TSB_AE"],
        "chronos-2-P50",
        "tsb",
    )

    merged = validation.merge(
        holdout,
        on="Medicine_ID",
        suffixes=("_Validation", "_Holdout"),
    )

    print()
    print("Comparable medicines:", len(merged))

    # ------------------------------------------------------------------
    # Rolling-style decision:
    #
    # The validation winner is treated as the model selected from
    # historical information.
    #
    # We then evaluate that decision on the unseen holdout.
    # ------------------------------------------------------------------

    merged["Rolling_Selected_Model"] = merged["Validation_Winner"]

    merged["Rolling_Prediction_AE"] = np.where(
        merged["Rolling_Selected_Model"] == "chronos-2-P50",
        merged["Chronos_AE_Holdout"],
        merged["TSB_AE_Holdout"],
    )

    merged["Chronos_Holdout_AE"] = merged["Chronos_AE_Holdout"]
    merged["TSB_Holdout_AE"] = merged["TSB_AE_Holdout"]

    merged["Ensemble_Holdout_AE"] = (
        merged["Chronos_Holdout_AE"]
        + merged["TSB_Holdout_AE"]
    ) / 2

    merged["Rolling_Better_Than_Chronos"] = (
        merged["Rolling_Prediction_AE"]
        < merged["Chronos_Holdout_AE"]
    )

    merged["Rolling_Better_Than_TSB"] = (
        merged["Rolling_Prediction_AE"]
        < merged["TSB_Holdout_AE"]
    )

    merged["Rolling_Better_Than_Ensemble"] = (
        merged["Rolling_Prediction_AE"]
        < merged["Ensemble_Holdout_AE"]
    )

    # ------------------------------------------------------------------
    # Overall results
    # ------------------------------------------------------------------

    total_actual = merged["Actual_Holdout"].sum()

    rolling_wape = (
        merged["Rolling_Prediction_AE"].sum()
        / total_actual
        * 100
    )

    chronos_wape = (
        merged["Chronos_Holdout_AE"].sum()
        / total_actual
        * 100
    )

    tsb_wape = (
        merged["TSB_Holdout_AE"].sum()
        / total_actual
        * 100
    )

    ensemble_wape = (
        merged["Ensemble_Holdout_AE"].sum()
        / total_actual
        * 100
    )

    print()
    print("=" * 80)
    print("HOLDOUT PERFORMANCE")
    print("=" * 80)

    print(
        f"Rolling selection WAPE: {rolling_wape:.3f}"
    )

    print(
        f"Chronos WAPE:           {chronos_wape:.3f}"
    )

    print(
        f"TSB WAPE:               {tsb_wape:.3f}"
    )

    print(
        f"50/50 Ensemble WAPE:    {ensemble_wape:.3f}"
    )

    # ------------------------------------------------------------------
    # Selection distribution
    # ------------------------------------------------------------------

    print()
    print("=" * 80)
    print("ROLLING MODEL SELECTION")
    print("=" * 80)

    print(
        merged["Rolling_Selected_Model"]
        .value_counts()
        .to_string()
    )

    # ------------------------------------------------------------------
    # Medicine-level comparison
    # ------------------------------------------------------------------

    print()
    print("=" * 80)
    print("MEDICINE-LEVEL RESULTS")
    print("=" * 80)

    print(
        "Rolling better than Chronos:",
        int(merged["Rolling_Better_Than_Chronos"].sum()),
        "/",
        len(merged),
    )

    print(
        "Rolling better than TSB:",
        int(merged["Rolling_Better_Than_TSB"].sum()),
        "/",
        len(merged),
    )

    print(
        "Rolling better than Ensemble:",
        int(merged["Rolling_Better_Than_Ensemble"].sum()),
        "/",
        len(merged),
    )

    # ------------------------------------------------------------------
    # Winner transitions
    # ------------------------------------------------------------------

    print()
    print("=" * 80)
    print("WINNER TRANSITIONS")
    print("=" * 80)

    merged["Winner_Transition"] = (
        merged["Validation_Winner"]
        + " -> "
        + merged["Holdout_Winner"]
    )

    print(
        merged["Winner_Transition"]
        .value_counts()
        .to_string()
    )

    # ------------------------------------------------------------------
    # Stability
    # ------------------------------------------------------------------

    merged["Winner_Stable"] = (
        merged["Validation_Winner"]
        == merged["Holdout_Winner"]
    )

    stable = int(merged["Winner_Stable"].sum())
    total = len(merged)

    stability = (
        stable / total * 100
        if total
        else np.nan
    )

    print()
    print("=" * 80)
    print("WINNER STABILITY")
    print("=" * 80)

    print("Stable winners:", stable)
    print("Comparable medicines:", total)
    print(f"Winner stability: {stability:.2f}%")

    # ------------------------------------------------------------------
    # Regime analysis
    # ------------------------------------------------------------------

    regime_summary = (
        merged
        .groupby("Regime_Validation")
        .agg(
            Medicines=("Medicine_ID", "count"),
            Stable=("Winner_Stable", "sum"),
            Rolling_AE=("Rolling_Prediction_AE", "sum"),
            Chronos_AE=("Chronos_Holdout_AE", "sum"),
            TSB_AE=("TSB_Holdout_AE", "sum"),
            Actual=("Actual_Holdout", "sum"),
        )
        .reset_index()
    )

    regime_summary["Rolling_WAPE"] = (
        regime_summary["Rolling_AE"]
        / regime_summary["Actual"]
        * 100
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

    regime_summary["Stability_Percentage"] = (
        regime_summary["Stable"]
        / regime_summary["Medicines"]
        * 100
    )

    print()
    print("=" * 80)
    print("ROLLING PERFORMANCE BY REGIME")
    print("=" * 80)

    print(
        regime_summary[
            [
                "Regime_Validation",
                "Medicines",
                "Rolling_WAPE",
                "Chronos_WAPE",
                "TSB_WAPE",
                "Stability_Percentage",
            ]
        ]
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

    merged.to_parquet(
        RESULTS_OUTPUT,
        index=False,
    )

    regime_summary.to_parquet(
        SUMMARY_OUTPUT,
        index=False,
    )

    print()
    print("=" * 80)
    print("SAVED")
    print("=" * 80)

    print(RESULTS_OUTPUT)
    print(SUMMARY_OUTPUT)


if __name__ == "__main__":
    main()
