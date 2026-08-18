import pandas as pd
import numpy as np
from pathlib import Path


INPUT = Path(
    "data/05_gold/demand_forecasting/"
    "medicine_robustness/medicine_model_robustness.parquet"
)

OUTPUT_DIR = Path(
    "data/05_gold/demand_forecasting/"
    "routing_threshold_optimization"
)

OUTPUT = OUTPUT_DIR / "routing_threshold_results.parquet"
SUMMARY = OUTPUT_DIR / "routing_threshold_summary.parquet"


def evaluate_rule(
    validation,
    holdout,
    condition,
    rule_name,
):
    """
    condition is evaluated using validation-period information only.
    The resulting routing decision is then evaluated on untouched holdout data.
    """

    validation = validation.copy()
    holdout = holdout.copy()

    validation["Selected_Model"] = np.where(
        condition(validation),
        "chronos-2-P50",
        "tsb",
    )

    decisions = validation[
        ["Medicine_ID", "Selected_Model"]
    ].copy()

    holdout = holdout.merge(
        decisions,
        on="Medicine_ID",
        how="inner",
    )

    holdout["Selected_Predicted"] = np.where(
        holdout["Selected_Model"] == "chronos-2-P50",
        holdout["Chronos_Predicted"],
        holdout["TSB_Predicted"],
    )

    holdout["Selected_AE"] = (
        holdout["Actual"] -
        holdout["Selected_Predicted"]
    ).abs()

    total_actual = holdout["Actual"].sum()

    wape = (
        holdout["Selected_AE"].sum()
        / total_actual
        * 100
        if total_actual != 0
        else np.nan
    )

    chronos_wape = (
        holdout["Chronos_AE"].sum()
        / holdout["Actual"].sum()
        * 100
    )

    tsb_wape = (
        holdout["TSB_AE"].sum()
        / holdout["Actual"].sum()
        * 100
    )

    chronos_count = (
        holdout["Selected_Model"]
        == "chronos-2-P50"
    ).sum()

    return {
        "Rule": rule_name,
        "Medicines": len(holdout),
        "Chronos_Medicines": int(chronos_count),
        "TSB_Medicines": int(len(holdout) - chronos_count),
        "Holdout_WAPE": wape,
        "Chronos_WAPE": chronos_wape,
        "TSB_WAPE": tsb_wape,
        "Improvement_vs_TSB": tsb_wape - wape,
    }


def main():

    print("=" * 80)
    print("ROUTING THRESHOLD OPTIMIZATION")
    print("=" * 80)

    df = pd.read_parquet(INPUT)

    print()
    print("Input rows:", len(df))
    print("Medicines:", df["Medicine_ID"].nunique())

    validation = df[
        df["Split"] == "validation"
    ].copy()

    holdout = df[
        df["Split"] == "holdout"
    ].copy()

    comparable = (
        set(validation["Medicine_ID"])
        & set(holdout["Medicine_ID"])
    )

    validation = validation[
        validation["Medicine_ID"].isin(comparable)
    ].copy()

    holdout = holdout[
        holdout["Medicine_ID"].isin(comparable)
    ].copy()

    print("Validation medicines:", validation["Medicine_ID"].nunique())
    print("Holdout medicines:", holdout["Medicine_ID"].nunique())

    results = []

    # ------------------------------------------------------------------
    # Baselines
    # ------------------------------------------------------------------

    results.append(
        evaluate_rule(
            validation,
            holdout,
            lambda x: pd.Series(False, index=x.index),
            "TSB_only",
        )
    )

    results.append(
        evaluate_rule(
            validation,
            holdout,
            lambda x: pd.Series(True, index=x.index),
            "Chronos_only",
        )
    )

    # ------------------------------------------------------------------
    # Regime rules
    # ------------------------------------------------------------------

    results.append(
        evaluate_rule(
            validation,
            holdout,
            lambda x: x["Regime"] == "Lumpy",
            "Chronos_Lumpy_TSB_Intermittent",
        )
    )

    results.append(
        evaluate_rule(
            validation,
            holdout,
            lambda x: x["Regime"] == "Intermittent",
            "Chronos_Intermittent_TSB_Lumpy",
        )
    )

    # ------------------------------------------------------------------
    # ADI thresholds
    # ------------------------------------------------------------------

    for threshold in [
        2,
        3,
        4,
        5,
        6,
        8,
        10,
        12,
        15,
    ]:

        results.append(
            evaluate_rule(
                validation,
                holdout,
                lambda x, t=threshold:
                    x["ADI"] >= t,
                f"Chronos_ADI_ge_{threshold}",
            )
        )

        results.append(
            evaluate_rule(
                validation,
                holdout,
                lambda x, t=threshold:
                    x["ADI"] < t,
                f"Chronos_ADI_lt_{threshold}",
            )
        )

    # ------------------------------------------------------------------
    # CV2 thresholds
    # ------------------------------------------------------------------

    for threshold in [
        0.5,
        0.7,
        0.9,
        1.0,
        1.2,
        1.5,
        2.0,
    ]:

        results.append(
            evaluate_rule(
                validation,
                holdout,
                lambda x, t=threshold:
                    x["CV2"] >= t,
                f"Chronos_CV2_ge_{threshold}",
            )
        )

        results.append(
            evaluate_rule(
                validation,
                holdout,
                lambda x, t=threshold:
                    x["CV2"] < t,
                f"Chronos_CV2_lt_{threshold}",
            )
        )

    # ------------------------------------------------------------------
    # Demand thresholds
    # ------------------------------------------------------------------

    demand_thresholds = [
        validation["Total_Demand"].quantile(q)
        for q in [
            0.25,
            0.50,
            0.75,
        ]
    ]

    for threshold in demand_thresholds:

        results.append(
            evaluate_rule(
                validation,
                holdout,
                lambda x, t=threshold:
                    x["Total_Demand"] >= t,
                f"Chronos_Demand_ge_{threshold:.2f}",
            )
        )

        results.append(
            evaluate_rule(
                validation,
                holdout,
                lambda x, t=threshold:
                    x["Total_Demand"] < t,
                f"Chronos_Demand_lt_{threshold:.2f}",
            )
        )

        # ------------------------------------------------------------------
    # Validation winner rules
    # ------------------------------------------------------------------

    results.append(
        evaluate_rule(
            validation,
            holdout,
            lambda x:
                x["Chronos_AE"] < x["TSB_AE"],
            "Validation_Winner",
        )
    )

    results.append(
        evaluate_rule(
            validation,
            holdout,
            lambda x:
                x["Chronos_AE"] + 50 < x["TSB_AE"],
            "Validation_Chronos_Margin_gt_50",
        )
    )

    results.append(
        evaluate_rule(
            validation,
            holdout,
            lambda x:
                x["Chronos_AE"] + 100 < x["TSB_AE"],
            "Validation_Chronos_Margin_gt_100",
        )
    )

    results.append(
        evaluate_rule(
            validation,
            holdout,
            lambda x:
                x["Chronos_AE"] + 200 < x["TSB_AE"],
            "Validation_Chronos_Margin_gt_200",
        )
    )
    # ------------------------------------------------------------------
    # Validation advantage percentage
    # ------------------------------------------------------------------

    validation["Validation_Advantage_Pct"] = (
        (
            validation["TSB_AE"]
            - validation["Chronos_AE"]
        )
        / validation["TSB_AE"].replace(0, np.nan)
        * 100
    )

    for threshold in [
        0,
        5,
        10,
        20,
        30,
        50,
    ]:

        results.append(
            evaluate_rule(
                validation,
                holdout,
                lambda x, t=threshold:
                    x["Validation_Advantage_Pct"] >= t,
                f"Chronos_Validation_Advantage_ge_{threshold}pct",
            )
        )

    result_df = pd.DataFrame(results)

    # ------------------------------------------------------------------
    # Sort by holdout WAPE
    # ------------------------------------------------------------------

    result_df = result_df.sort_values(
        "Holdout_WAPE"
    ).reset_index(drop=True)

    print()
    print("=" * 80)
    print("ROUTING RULE PERFORMANCE")
    print("=" * 80)

    print(
        result_df[
            [
                "Rule",
                "Chronos_Medicines",
                "TSB_Medicines",
                "Holdout_WAPE",
                "Chronos_WAPE",
                "TSB_WAPE",
                "Improvement_vs_TSB",
            ]
        ]
        .round(3)
        .to_string(index=False)
    )

    # ------------------------------------------------------------------
    # Best rules
    # ------------------------------------------------------------------

    print()
    print("=" * 80)
    print("TOP 10 ROUTING RULES")
    print("=" * 80)

    print(
        result_df.head(10)[
            [
                "Rule",
                "Chronos_Medicines",
                "TSB_Medicines",
                "Holdout_WAPE",
                "Improvement_vs_TSB",
            ]
        ]
        .round(3)
        .to_string(index=False)
    )

    # ------------------------------------------------------------------
    # Rules that actually beat TSB
    # ------------------------------------------------------------------

    better = result_df[
        result_df["Improvement_vs_TSB"] > 0
    ]

    print()
    print("=" * 80)
    print("RULES BEATING TSB")
    print("=" * 80)

    if better.empty:
        print("NO ROUTING RULE BEATS TSB.")
    else:
        print(
            better[
                [
                    "Rule",
                    "Chronos_Medicines",
                    "TSB_Medicines",
                    "Holdout_WAPE",
                    "Improvement_vs_TSB",
                ]
            ]
            .round(3)
            .to_string(index=False)
        )

    # ------------------------------------------------------------------
    # Best explainable rule
    # ------------------------------------------------------------------

    non_baseline = result_df[
        ~result_df["Rule"].isin(
            [
                "TSB_only",
                "Chronos_only",
            ]
        )
    ]

    best = non_baseline.iloc[0]

    print()
    print("=" * 80)
    print("BEST EXPLAINABLE ROUTING RULE")
    print("=" * 80)

    print("Rule:", best["Rule"])
    print("Chronos medicines:", int(best["Chronos_Medicines"]))
    print("TSB medicines:", int(best["TSB_Medicines"]))
    print("Holdout WAPE:", round(best["Holdout_WAPE"], 3))
    print(
        "Improvement vs TSB:",
        round(best["Improvement_vs_TSB"], 3),
    )

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_df.to_parquet(
        OUTPUT,
        index=False,
    )

    result_df.head(10).to_parquet(
        SUMMARY,
        index=False,
    )

    print()
    print("=" * 80)
    print("SAVED")
    print("=" * 80)

    print(OUTPUT)
    print(SUMMARY)


if __name__ == "__main__":
    main()
