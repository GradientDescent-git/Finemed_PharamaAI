from pathlib import Path

import numpy as np
import pandas as pd


INPUT = Path(
    "data/05_gold/demand_forecasting/monthly_experiment/"
    "chronos_monthly_backtest.parquet"
)

OUTPUT = Path(
    "data/05_gold/demand_forecasting/monthly_experiment/"
)

MODEL = "chronos-2-P50"

VALIDATION_CUTOFFS = pd.to_datetime([
    "2025-11-01",
    "2025-12-01",
    "2026-01-01",
    "2026-02-01",
])

HOLDOUT_CUTOFFS = pd.to_datetime([
    "2026-03-01",
    "2026-04-01",
])


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

def calculate_metrics(df):
    actual = df["Actual"].sum()
    predicted = df["Calibrated_Predicted"].sum()
    absolute_error = df["Absolute_Error"].sum()

    return {
        "Actual": actual,
        "Predicted": predicted,
        "Absolute_Error": absolute_error,
        "WAPE": (
            absolute_error / actual * 100
            if actual != 0
            else np.nan
        ),
        "Ratio": (
            predicted / actual
            if actual != 0
            else np.nan
        ),
        "MBE": predicted - actual,
    }


def evaluate(
    predictions,
    method,
    parameter,
):
    x = predictions.copy()

    x["Calibrated_Predicted"] = (
        x["Predicted"] * x["Calibration_Factor"]
    )

    x["Calibrated_Predicted"] = (
        x["Calibrated_Predicted"].clip(lower=0)
    )

    x["Absolute_Error"] = (
        x["Actual"] - x["Calibrated_Predicted"]
    ).abs()

    metrics = calculate_metrics(x)

    metrics["Method"] = method
    metrics["Parameter"] = parameter

    return metrics


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    print("=" * 80)
    print("CHRONOS-2 P50 CALIBRATION EXPERIMENT")
    print("=" * 80)

    df = pd.read_parquet(INPUT)

    df["Cutoff_Date"] = pd.to_datetime(
        df["Cutoff_Date"]
    )

    df["Medicine_ID"] = (
        df["Medicine_ID"].astype(str)
    )

    df = df[
        df["Model"] == MODEL
    ].copy()

    validation = df[
        df["Cutoff_Date"].isin(
            VALIDATION_CUTOFFS
        )
    ].copy()

    holdout = df[
        df["Cutoff_Date"].isin(
            HOLDOUT_CUTOFFS
        )
    ].copy()

    print()
    print("Model:", MODEL)
    print("Validation rows:", len(validation))
    print("Holdout rows:", len(holdout))

    # -----------------------------------------------------------------
    # BASELINE
    # -----------------------------------------------------------------

    validation["Calibration_Factor"] = 1.0

    baseline_validation = evaluate(
        validation,
        "raw_p50",
        1.0,
    )

    holdout["Calibration_Factor"] = 1.0

    baseline_holdout = evaluate(
        holdout,
        "raw_p50",
        1.0,
    )

    print()
    print("=" * 80)
    print("RAW P50 BASELINE")
    print("=" * 80)

    print(
        pd.DataFrame(
            [baseline_validation, baseline_holdout]
        ).to_string(index=False)
    )

    # -----------------------------------------------------------------
    # GLOBAL CALIBRATION
    #
    # IMPORTANT:
    # learned ONLY from validation.
    # -----------------------------------------------------------------

    global_factor = (
        validation["Actual"].sum()
        / validation["Predicted"].sum()
    )

    print()
    print("=" * 80)
    print("GLOBAL CALIBRATION")
    print("=" * 80)

    print(
        f"Validation-derived factor: "
        f"{global_factor:.6f}"
    )

    validation["Calibration_Factor"] = (
        global_factor
    )

    holdout["Calibration_Factor"] = (
        global_factor
    )

    global_validation = evaluate(
        validation,
        "global",
        global_factor,
    )

    global_holdout = evaluate(
        holdout,
        "global",
        global_factor,
    )

    # -----------------------------------------------------------------
    # MEDICINE-LEVEL FACTORS
    #
    # Factor is estimated ONLY from validation.
    # -----------------------------------------------------------------

    medicine_stats = (
        validation
        .groupby("Medicine_ID")
        .agg(
            Validation_Actual=(
                "Actual",
                "sum",
            ),
            Validation_Predicted=(
                "Predicted",
                "sum",
            ),
            Validation_Samples=(
                "Predicted",
                "count",
            ),
        )
        .reset_index()
    )

    medicine_stats["Raw_Factor"] = (
        medicine_stats["Validation_Actual"]
        / medicine_stats["Validation_Predicted"]
        .replace(0, np.nan)
    )

    # Replace invalid factors with global factor.
    medicine_stats["Raw_Factor"] = (
        medicine_stats["Raw_Factor"]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .fillna(global_factor)
    )

    # Prevent extreme corrections.
    medicine_stats["Raw_Factor"] = (
        medicine_stats["Raw_Factor"]
        .clip(
            lower=0.50,
            upper=1.50,
        )
    )

    factor_map = dict(
        zip(
            medicine_stats["Medicine_ID"],
            medicine_stats["Raw_Factor"],
        )
    )

    # -----------------------------------------------------------------
    # SHRINKAGE EXPERIMENT
    #
    # weight = n / (n + lambda)
    #
    # With only four validation observations per medicine,
    # aggressive medicine-level calibration would overfit.
    # -----------------------------------------------------------------

    results = [
        baseline_validation,
        baseline_holdout,
        global_validation,
        global_holdout,
    ]

    for lambda_value in [
        0.5,
        1.0,
        2.0,
        4.0,
        8.0,
        16.0,
    ]:

        for dataset_name, dataset in [
            ("validation", validation),
            ("holdout", holdout),
        ]:

            x = dataset.copy()

            factors = []

            for _, row in x.iterrows():

                medicine_id = (
                    str(row["Medicine_ID"])
                )

                medicine_factor = factor_map.get(
                    medicine_id,
                    global_factor,
                )

                n = medicine_stats.loc[
                    medicine_stats["Medicine_ID"]
                    == medicine_id,
                    "Validation_Samples",
                ]

                n = (
                    float(n.iloc[0])
                    if len(n)
                    else 0.0
                )

                weight = (
                    n / (n + lambda_value)
                )

                factor = (
                    global_factor
                    + weight
                    * (
                        medicine_factor
                        - global_factor
                    )
                )

                factors.append(factor)

            x["Calibration_Factor"] = factors

            result = evaluate(
                x,
                "medicine_shrinkage",
                lambda_value,
            )

            result["Dataset"] = dataset_name

            results.append(result)

    # -----------------------------------------------------------------
    # RESULTS TABLE
    # -----------------------------------------------------------------

    result_df = pd.DataFrame(results)

    print()
    print("=" * 80)
    print("CALIBRATION RESULTS")
    print("=" * 80)

    print(
        result_df[
            [
                "Method",
                "Parameter",
                "Actual",
                "Predicted",
                "Absolute_Error",
                "WAPE",
                "Ratio",
                "MBE",
            ]
        ]
        .to_string(index=False)
    )

    # -----------------------------------------------------------------
    # Separate validation / holdout result table
    # -----------------------------------------------------------------

    output_path = (
        OUTPUT
        / "chronos_p50_calibration_results.parquet"
    )

    result_df.to_parquet(
        output_path,
        index=False,
    )

    # -----------------------------------------------------------------
    # Save medicine factors
    # -----------------------------------------------------------------

    factors_path = (
        OUTPUT
        / "chronos_p50_medicine_factors.parquet"
    )

    medicine_stats.to_parquet(
        factors_path,
        index=False,
    )

    # -----------------------------------------------------------------
    # Determine best shrinkage using VALIDATION ONLY
    # -----------------------------------------------------------------

    shrinkage_validation = (
        result_df[
            (
                result_df["Method"]
                == "medicine_shrinkage"
            )
            & (
                result_df["Dataset"]
                == "validation"
            )
        ]
        .sort_values("WAPE")
    )

    best = (
        shrinkage_validation
        .iloc[0]
    )

    print()
    print("=" * 80)
    print("BEST VALIDATION SHRINKAGE")
    print("=" * 80)

    print(
        "Lambda:",
        best["Parameter"],
    )

    print(
        "Validation WAPE:",
        round(best["WAPE"], 3),
    )

    print()
    print("Saved:")
    print(output_path)
    print(factors_path)


if __name__ == "__main__":
    main()