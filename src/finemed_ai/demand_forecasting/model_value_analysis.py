import pandas as pd
import numpy as np
from pathlib import Path


INPUT = Path(
    "data/05_gold/demand_forecasting/medicine_robustness/"
    "medicine_model_robustness.parquet"
)

OUTPUT_DIR = Path(
    "data/05_gold/demand_forecasting/model_value_analysis"
)

OUTPUT = OUTPUT_DIR / "model_value_analysis.parquet"
SUMMARY = OUTPUT_DIR / "model_value_analysis_summary.parquet"


def main():

    print("=" * 80)
    print("MODEL VALUE ANALYSIS")
    print("=" * 80)

    df = pd.read_parquet(INPUT)

    print()
    print("Input rows:", len(df))
    print("Medicines:", df["Medicine_ID"].nunique())

    # ------------------------------------------------------------------
    # Keep only comparable medicines
    # ------------------------------------------------------------------

    comparable = (
        df.groupby("Medicine_ID")["Split"]
        .nunique()
    )

    comparable_ids = comparable[comparable == 2].index

    df = df[df["Medicine_ID"].isin(comparable_ids)].copy()

    print("Comparable medicines:", df["Medicine_ID"].nunique())

    # ------------------------------------------------------------------
    # Pivot validation / holdout
    # ------------------------------------------------------------------

    records = []

    for medicine_id, group in df.groupby("Medicine_ID"):

        validation = group[group["Split"] == "validation"]
        holdout = group[group["Split"] == "holdout"]

        if validation.empty or holdout.empty:
            continue

        v = validation.iloc[0]
        h = holdout.iloc[0]

        chronos_v = float(v["Chronos_AE"])
        tsb_v = float(v["TSB_AE"])

        chronos_h = float(h["Chronos_AE"])
        tsb_h = float(h["TSB_AE"])

        validation_advantage = tsb_v - chronos_v
        holdout_advantage = tsb_h - chronos_h

        records.append(
            {
                "Medicine_ID": medicine_id,
                "Regime": v["Regime"],
                "Days": v["Days"],
                "NonZero_Days": v["NonZero_Days"],
                "ADI": v["ADI"],
                "CV2": v["CV2"],
                "Total_Demand": v["Total_Demand"],

                "Validation_Chronos_AE": chronos_v,
                "Validation_TSB_AE": tsb_v,
                "Validation_Advantage": validation_advantage,

                "Holdout_Chronos_AE": chronos_h,
                "Holdout_TSB_AE": tsb_h,
                "Holdout_Advantage": holdout_advantage,

                "Validation_Winner": (
                    "chronos-2-P50"
                    if chronos_v < tsb_v
                    else "tsb"
                ),

                "Holdout_Winner": (
                    "chronos-2-P50"
                    if chronos_h < tsb_h
                    else "tsb"
                ),

                "Winner_Stable": (
                    (chronos_v < tsb_v and chronos_h < tsb_h)
                    or
                    (tsb_v <= chronos_v and tsb_h <= chronos_h)
                ),
            }
        )

    result = pd.DataFrame(records)

    print()
    print("Analysis medicines:", len(result))

    # ------------------------------------------------------------------
    # Demand-volume buckets
    # ------------------------------------------------------------------

    result["Demand_Bucket"] = pd.qcut(
        result["Total_Demand"],
        q=4,
        labels=[
            "Q1_Low",
            "Q2",
            "Q3",
            "Q4_High",
        ],
        duplicates="drop",
    )

    # ------------------------------------------------------------------
    # Stable winners
    # ------------------------------------------------------------------

    result["Stable_Winner"] = np.where(
        result["Winner_Stable"],
        result["Holdout_Winner"],
        "unstable",
    )

    # ------------------------------------------------------------------
    # Value analysis
    # ------------------------------------------------------------------

    print()
    print("=" * 80)
    print("OVERALL MODEL VALUE")
    print("=" * 80)

    print()
    print(
        "Stable medicines:",
        int(result["Winner_Stable"].sum()),
        "/",
        len(result),
    )

    print(
        "Chronos stable winners:",
        int(
            (
                result["Winner_Stable"]
                & (result["Stable_Winner"] == "chronos-2-P50")
            ).sum()
        ),
    )

    print(
        "TSB stable winners:",
        int(
            (
                result["Winner_Stable"]
                & (result["Stable_Winner"] == "tsb")
            ).sum()
        ),
    )

    # ------------------------------------------------------------------
    # Stability by regime
    # ------------------------------------------------------------------

    print()
    print("=" * 80)
    print("VALUE BY REGIME")
    print("=" * 80)

    regime_summary = (
        result.groupby("Regime")
        .agg(
            Medicines=("Medicine_ID", "count"),
            Stable_Medicines=("Winner_Stable", "sum"),
            Avg_Demand=("Total_Demand", "mean"),
            Total_Demand=("Total_Demand", "sum"),
            Mean_Validation_Advantage=("Validation_Advantage", "mean"),
            Mean_Holdout_Advantage=("Holdout_Advantage", "mean"),
        )
        .reset_index()
    )

    regime_summary["Stability_Percentage"] = (
        regime_summary["Stable_Medicines"]
        / regime_summary["Medicines"]
        * 100
    )

    print(
        regime_summary.round(3).to_string(index=False)
    )

    # ------------------------------------------------------------------
    # Demand volume
    # ------------------------------------------------------------------

    print()
    print("=" * 80)
    print("VALUE BY DEMAND VOLUME")
    print("=" * 80)

    volume_summary = (
        result.groupby("Demand_Bucket", observed=True)
        .agg(
            Medicines=("Medicine_ID", "count"),
            Stable_Medicines=("Winner_Stable", "sum"),
            Avg_Demand=("Total_Demand", "mean"),
            Total_Demand=("Total_Demand", "sum"),
            Mean_Validation_Advantage=("Validation_Advantage", "mean"),
            Mean_Holdout_Advantage=("Holdout_Advantage", "mean"),
        )
        .reset_index()
    )

    volume_summary["Stability_Percentage"] = (
        volume_summary["Stable_Medicines"]
        / volume_summary["Medicines"]
        * 100
    )

    print(
        volume_summary.round(3).to_string(index=False)
    )

    # ------------------------------------------------------------------
    # High-value medicines
    # ------------------------------------------------------------------

    print()
    print("=" * 80)
    print("HIGH-VALUE MEDICINES")
    print("=" * 80)

    high_value = (
        result.sort_values(
            "Total_Demand",
            ascending=False,
        )
        .head(20)
    )

    print(
        high_value[
            [
                "Medicine_ID",
                "Regime",
                "Total_Demand",
                "ADI",
                "CV2",
                "Validation_Winner",
                "Holdout_Winner",
                "Winner_Stable",
                "Validation_Advantage",
                "Holdout_Advantage",
            ]
        ]
        .round(3)
        .to_string(index=False)
    )

    # ------------------------------------------------------------------
    # Stable Chronos value
    # ------------------------------------------------------------------

    print()
    print("=" * 80)
    print("STABLE CHRONOS VALUE")
    print("=" * 80)

    stable_chronos = result[
        result["Winner_Stable"]
        & (result["Stable_Winner"] == "chronos-2-P50")
    ].copy()

    stable_chronos = stable_chronos.sort_values(
        "Holdout_Advantage",
        ascending=False,
    )

    print(
        stable_chronos[
            [
                "Medicine_ID",
                "Regime",
                "Total_Demand",
                "ADI",
                "CV2",
                "Validation_Advantage",
                "Holdout_Advantage",
            ]
        ]
        .round(3)
        .to_string(index=False)
    )

    # ------------------------------------------------------------------
    # Stable TSB value
    # ------------------------------------------------------------------

    print()
    print("=" * 80)
    print("STABLE TSB VALUE")
    print("=" * 80)

    stable_tsb = result[
        result["Winner_Stable"]
        & (result["Stable_Winner"] == "tsb")
    ].copy()

    stable_tsb = stable_tsb.sort_values(
        "Holdout_Advantage",
        ascending=True,
    )

    print(
        stable_tsb[
            [
                "Medicine_ID",
                "Regime",
                "Total_Demand",
                "ADI",
                "CV2",
                "Validation_Advantage",
                "Holdout_Advantage",
            ]
        ]
        .round(3)
        .to_string(index=False)
    )

    # ------------------------------------------------------------------
    # Correlation analysis
    # ------------------------------------------------------------------

    print()
    print("=" * 80)
    print("FEATURE / MODEL-VALUE CORRELATION")
    print("=" * 80)

    numeric_cols = [
        "Total_Demand",
        "ADI",
        "CV2",
        "Validation_Advantage",
        "Holdout_Advantage",
    ]

    print(
        result[numeric_cols]
        .corr()
        .round(3)
        .to_string()
    )

    # ------------------------------------------------------------------
    # Production candidates
    # ------------------------------------------------------------------

    result["Production_Candidate"] = np.where(
        result["Winner_Stable"],
        result["Stable_Winner"],
        "tsb",
    )

    print()
    print("=" * 80)
    print("PRODUCTION CANDIDATE DISTRIBUTION")
    print("=" * 80)

    print(
        result["Production_Candidate"]
        .value_counts()
        .to_string()
    )

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_parquet(
        OUTPUT,
        index=False,
    )

    summary_frames = []

    regime_summary["Analysis"] = "regime"
    volume_summary["Analysis"] = "demand_volume"

    summary_frames.append(regime_summary)
    summary_frames.append(volume_summary)

    summary = pd.concat(
        summary_frames,
        ignore_index=True,
    )

    summary.to_parquet(
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
