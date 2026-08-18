from pathlib import Path

import numpy as np
import pandas as pd
import torch

from chronos import Chronos2Pipeline


INPUT = "data/05_gold/demand_forecasting/monthly_experiment/monthly_demand.parquet"
OUTPUT = Path("data/05_gold/demand_forecasting/monthly_experiment")
OUTPUT.mkdir(parents=True, exist_ok=True)

MODEL_ID = "amazon/chronos-2"

CUTOFFS = pd.to_datetime([
    "2025-11-01",
    "2025-12-01",
    "2026-01-01",
    "2026-02-01",
    "2026-03-01",
    "2026-04-01",
])

VALIDATION = set(CUTOFFS[:4])
HOLDOUT = set(CUTOFFS[4:])

MIN_HISTORY = 12

QUANTILES = (
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
)


def main():
    print("=" * 80)
    print("MONTHLY CHRONOS-2 BACKTEST")
    print("=" * 80)

    df = pd.read_parquet(INPUT)

    df["Month"] = pd.to_datetime(df["Month"])
    df["Medicine_ID"] = df["Medicine_ID"].astype(str)

    print("Rows:", len(df))
    print("Medicines:", df["Medicine_ID"].nunique())
    print(
        "Date range:",
        df["Month"].min().date(),
        "->",
        df["Month"].max().date(),
    )
    print("Minimum history:", MIN_HISTORY)
    print()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Device:", device)
    print("Loading Chronos-2...")

    pipeline = Chronos2Pipeline.from_pretrained(
        MODEL_ID,
        device_map=device,
        torch_dtype=(
            torch.bfloat16
            if device == "cuda"
            else torch.float32
        ),
    )

    print("Chronos-2 loaded.")
    print()

    rows = []

    for cutoff in CUTOFFS:

        forecast_month = cutoff + pd.offsets.MonthBegin(1)

        train = df[df["Month"] <= cutoff].copy()

        actual = df[
            df["Month"] == forecast_month
        ][
            ["Medicine_ID", "Actual"]
        ].copy()

        if actual.empty:
            print(
                "WARNING: no actual data for",
                forecast_month.date(),
            )
            continue

        actual_map = dict(
            zip(
                actual["Medicine_ID"],
                actual["Actual"],
            )
        )

        eligible = []

        for medicine_id, group in train.groupby("Medicine_ID"):

            group = group.sort_values("Month")

            if len(group) < MIN_HISTORY:
                continue

            if medicine_id not in actual_map:
                continue

            eligible.append(medicine_id)

        print(
            f"Cutoff {cutoff.date()} | "
            f"Forecast {forecast_month.date()} | "
            f"Eligible medicines: {len(eligible)}"
        )

        if not eligible:
            continue

        context = train[
            train["Medicine_ID"].isin(eligible)
        ][
            ["Medicine_ID", "Month", "Actual"]
        ].copy()

        context = context.rename(
            columns={
                "Medicine_ID": "item_id",
                "Month": "timestamp",
                "Actual": "target",
            }
        )

        context = context.sort_values(
            ["item_id", "timestamp"]
        )

        raw = pipeline.predict_df(
            context,
            prediction_length=1,
            quantile_levels=list(QUANTILES),
            id_column="item_id",
            timestamp_column="timestamp",
            target="target",
        )

        raw["item_id"] = raw["item_id"].astype(str)

        for _, row in raw.iterrows():

            medicine_id = str(row["item_id"])

            if medicine_id not in actual_map:
                continue

            actual_value = float(
                actual_map[medicine_id]
            )

            for q in QUANTILES:

                column = str(q)

                prediction = max(
                    float(row[column]),
                    0.0,
                )

                rows.append(
                    {
                        "Cutoff_Date": cutoff,
                        "Forecast_Month": forecast_month,
                        "Medicine_ID": medicine_id,
                        "Model": f"chronos-2-P{int(q * 100)}",
                        "Quantile": q,
                        "Actual": actual_value,
                        "Predicted": prediction,
                        "Absolute_Error": abs(
                            actual_value - prediction
                        ),
                    }
                )

    results = pd.DataFrame(rows)

    if results.empty:
        raise RuntimeError(
            "No Chronos results were generated."
        )

    summary = (
        results
        .groupby(
            ["Model", "Cutoff_Date"],
            as_index=False,
        )
        .agg(
            Medicines=("Medicine_ID", "nunique"),
            Actual=("Actual", "sum"),
            Predicted=("Predicted", "sum"),
            Absolute_Error=("Absolute_Error", "sum"),
        )
    )

    summary["WAPE"] = (
        summary["Absolute_Error"]
        / summary["Actual"]
        * 100
    )

    summary["Ratio"] = (
        summary["Predicted"]
        / summary["Actual"]
    )

    summary["MBE"] = (
        summary["Predicted"]
        - summary["Actual"]
    )

    summary["Split"] = np.where(
        summary["Cutoff_Date"].isin(VALIDATION),
        "validation",
        "holdout",
    )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    print()
    print("=" * 80)
    print("VALIDATION PERFORMANCE")
    print("=" * 80)

    validation = summary[
        summary["Split"] == "validation"
    ]

    vg = (
        validation
        .groupby("Model")
        .agg(
            Medicines=("Medicines", "mean"),
            Actual=("Actual", "sum"),
            Predicted=("Predicted", "sum"),
            Absolute_Error=("Absolute_Error", "sum"),
        )
    )

    vg["WAPE"] = (
        vg["Absolute_Error"]
        / vg["Actual"]
        * 100
    )

    vg["Ratio"] = (
        vg["Predicted"]
        / vg["Actual"]
    )

    print(
        vg.sort_values("WAPE")
        .to_string()
    )

    # ------------------------------------------------------------------
    # Holdout
    # ------------------------------------------------------------------

    print()
    print("=" * 80)
    print("HOLDOUT PERFORMANCE")
    print("=" * 80)

    holdout = summary[
        summary["Split"] == "holdout"
    ]

    hg = (
        holdout
        .groupby("Model")
        .agg(
            Medicines=("Medicines", "mean"),
            Actual=("Actual", "sum"),
            Predicted=("Predicted", "sum"),
            Absolute_Error=("Absolute_Error", "sum"),
        )
    )

    hg["WAPE"] = (
        hg["Absolute_Error"]
        / hg["Actual"]
        * 100
    )

    hg["Ratio"] = (
        hg["Predicted"]
        / hg["Actual"]
    )

    print(
        hg.sort_values("WAPE")
        .to_string()
    )

    # ------------------------------------------------------------------
    # Best validation quantile
    # ------------------------------------------------------------------

    print()
    print("=" * 80)
    print("BEST QUANTILE")
    print("=" * 80)

    best_model = (
        vg["WAPE"]
        .sort_values()
        .index[0]
    )

    best_validation_wape = float(
        vg.loc[best_model, "WAPE"]
    )

    print(
        "Validation winner:",
        best_model,
    )

    print(
        "Validation WAPE:",
        round(best_validation_wape, 3),
    )

    if best_model in hg.index:

        holdout_wape = float(
            hg.loc[best_model, "WAPE"]
        )

        print(
            "Holdout WAPE:",
            round(holdout_wape, 3),
        )

        print(
            "Holdout ratio:",
            round(
                float(hg.loc[best_model, "Ratio"]),
                4,
            ),
        )

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    results_path = (
        OUTPUT
        / "chronos_monthly_backtest.parquet"
    )

    summary_path = (
        OUTPUT
        / "chronos_monthly_summary.parquet"
    )

    results.to_parquet(
        results_path,
        index=False,
    )

    summary.to_parquet(
        summary_path,
        index=False,
    )

    print()
    print("=" * 80)
    print("SAVED")
    print("=" * 80)
    print(results_path)
    print(summary_path)


if __name__ == "__main__":
    main()
