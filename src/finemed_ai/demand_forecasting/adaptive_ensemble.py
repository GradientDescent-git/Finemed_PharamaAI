from pathlib import Path

import numpy as np
import pandas as pd


INPUT = Path(
    "data/05_gold/demand_forecasting/"
    "medicine_robustness/medicine_model_robustness.parquet"
)

OUTPUT_DIR = Path(
    "data/05_gold/demand_forecasting/adaptive_ensemble"
)

RESULTS_OUTPUT = OUTPUT_DIR / "adaptive_ensemble_results.parquet"
WEIGHTS_OUTPUT = OUTPUT_DIR / "adaptive_ensemble_weights.parquet"


WEIGHTS = np.round(np.arange(0.0, 1.01, 0.1), 1)


def wape(actual, predicted):
    denominator = np.abs(actual).sum()

    if denominator == 0:
        return np.nan

    return (
        np.abs(actual - predicted).sum()
        / denominator
        * 100
    )


def evaluate_model(df, prediction_column):
    return {
        "Actual": df["Actual"].sum(),
        "Predicted": df[prediction_column].sum(),
        "Absolute_Error": (
            np.abs(
                df["Actual"]
                - df[prediction_column]
            ).sum()
        ),
        "WAPE": wape(
            df["Actual"].to_numpy(),
            df[prediction_column].to_numpy(),
        ),
    }


def main():

    print("=" * 80)
    print("ADAPTIVE ENSEMBLE EXPERIMENT")
    print("=" * 80)

    df = pd.read_parquet(INPUT)

    print()
    print("Input rows:", len(df))
    print(
        "Medicines:",
        df["Medicine_ID"].nunique(),
    )

    required = {
        "Split",
        "Medicine_ID",
        "Actual",
        "Chronos_Predicted",
        "TSB_Predicted",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    # ------------------------------------------------------------------
    # Validation / holdout separation
    # ------------------------------------------------------------------

    validation = df[
        df["Split"] == "validation"
    ].copy()

    holdout = df[
        df["Split"] == "holdout"
    ].copy()

    print(
        "Validation rows:",
        len(validation),
    )

    print(
        "Holdout rows:",
        len(holdout),
    )

    # ------------------------------------------------------------------
    # Build equal-weight ensemble
    # ------------------------------------------------------------------

    validation["Ensemble_0.5"] = (
        0.5 * validation["Chronos_Predicted"]
        + 0.5 * validation["TSB_Predicted"]
    )

    holdout["Ensemble_0.5"] = (
        0.5 * holdout["Chronos_Predicted"]
        + 0.5 * holdout["TSB_Predicted"]
    )

    # ------------------------------------------------------------------
    # Global validation weight selection
    #
    # IMPORTANT:
    # Weight is selected ONLY from validation.
    # Holdout is never used for selecting the weight.
    # ------------------------------------------------------------------

    weight_results = []

    for weight in WEIGHTS:

        predicted = (
            weight * validation["Chronos_Predicted"]
            + (1.0 - weight)
            * validation["TSB_Predicted"]
        )

        score = wape(
            validation["Actual"].to_numpy(),
            predicted.to_numpy(),
        )

        weight_results.append(
            {
                "Chronos_Weight": weight,
                "TSB_Weight": round(
                    1.0 - weight,
                    1,
                ),
                "Validation_WAPE": score,
            }
        )

    weight_table = pd.DataFrame(
        weight_results
    )

    best_row = weight_table.loc[
        weight_table["Validation_WAPE"].idxmin()
    ]

    best_chronos_weight = float(
        best_row["Chronos_Weight"]
    )

    best_tsb_weight = float(
        best_row["TSB_Weight"]
    )

    print()
    print("=" * 80)
    print("VALIDATION WEIGHT SEARCH")
    print("=" * 80)

    print(
        weight_table
        .round(3)
        .to_string(index=False)
    )

    print()
    print(
        "Selected Chronos weight:",
        best_chronos_weight,
    )

    print(
        "Selected TSB weight:",
        best_tsb_weight,
    )

    print(
        "Selected validation WAPE:",
        round(
            float(best_row["Validation_WAPE"]),
            3,
        ),
    )

    # ------------------------------------------------------------------
    # Apply frozen weight
    # ------------------------------------------------------------------

    validation["Adaptive_Ensemble"] = (
        best_chronos_weight
        * validation["Chronos_Predicted"]
        + best_tsb_weight
        * validation["TSB_Predicted"]
    )

    holdout["Adaptive_Ensemble"] = (
        best_chronos_weight
        * holdout["Chronos_Predicted"]
        + best_tsb_weight
        * holdout["TSB_Predicted"]
    )

    # ------------------------------------------------------------------
    # Model-level comparison
    # ------------------------------------------------------------------

    models = {
        "chronos-2-P50": "Chronos_Predicted",
        "tsb": "TSB_Predicted",
        "simple_ensemble": "Ensemble_0.5",
        "adaptive_ensemble": "Adaptive_Ensemble",
    }

    summary_rows = []

    for split_name, split_df in [
        ("validation", validation),
        ("holdout", holdout),
    ]:

        for model_name, prediction_column in models.items():

            metrics = evaluate_model(
                split_df,
                prediction_column,
            )

            summary_rows.append(
                {
                    "Split": split_name,
                    "Model": model_name,
                    "Medicines": split_df[
                        "Medicine_ID"
                    ].nunique(),
                    **metrics,
                }
            )

    summary = pd.DataFrame(
        summary_rows
    )

    # ------------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------------

    print()
    print("=" * 80)
    print("VALIDATION MODEL COMPARISON")
    print("=" * 80)

    print(
        summary[
            summary["Split"] == "validation"
        ][
            [
                "Model",
                "Medicines",
                "Actual",
                "Predicted",
                "Absolute_Error",
                "WAPE",
            ]
        ]
        .round(3)
        .to_string(index=False)
    )

    print()
    print("=" * 80)
    print("HOLDOUT MODEL COMPARISON")
    print("=" * 80)

    print(
        summary[
            summary["Split"] == "holdout"
        ][
            [
                "Model",
                "Medicines",
                "Actual",
                "Predicted",
                "Absolute_Error",
                "WAPE",
            ]
        ]
        .round(3)
        .to_string(index=False)
    )

    # ------------------------------------------------------------------
    # Medicine-level holdout comparison
    # ------------------------------------------------------------------

    holdout_medicine = (
        holdout
        .groupby("Medicine_ID")
        .agg(
            Actual=("Actual", "sum"),
            Chronos_Predicted=(
                "Chronos_Predicted",
                "sum",
            ),
            TSB_Predicted=(
                "TSB_Predicted",
                "sum",
            ),
            Simple_Ensemble_Predicted=(
                "Ensemble_0.5",
                "sum",
            ),
            Adaptive_Ensemble_Predicted=(
                "Adaptive_Ensemble",
                "sum",
            ),
        )
        .reset_index()
    )

    holdout_medicine[
        "Chronos_AE"
    ] = np.abs(
        holdout_medicine["Actual"]
        - holdout_medicine["Chronos_Predicted"]
    )

    holdout_medicine[
        "TSB_AE"
    ] = np.abs(
        holdout_medicine["Actual"]
        - holdout_medicine["TSB_Predicted"]
    )

    holdout_medicine[
        "Simple_Ensemble_AE"
    ] = np.abs(
        holdout_medicine["Actual"]
        - holdout_medicine[
            "Simple_Ensemble_Predicted"
        ]
    )

    holdout_medicine[
        "Adaptive_Ensemble_AE"
    ] = np.abs(
        holdout_medicine["Actual"]
        - holdout_medicine[
            "Adaptive_Ensemble_Predicted"
        ]
    )

    ae_columns = [
        "Chronos_AE",
        "TSB_AE",
        "Simple_Ensemble_AE",
        "Adaptive_Ensemble_AE",
    ]

    holdout_medicine[
        "Adaptive_Better_Than_Chronos"
    ] = (
        holdout_medicine["Adaptive_Ensemble_AE"]
        < holdout_medicine["Chronos_AE"]
    )

    holdout_medicine[
        "Adaptive_Better_Than_TSB"
    ] = (
        holdout_medicine["Adaptive_Ensemble_AE"]
        < holdout_medicine["TSB_AE"]
    )

    holdout_medicine[
        "Adaptive_Better_Than_Both"
    ] = (
        holdout_medicine[
            "Adaptive_Better_Than_Chronos"
        ]
        & holdout_medicine[
            "Adaptive_Better_Than_TSB"
        ]
    )

    print()
    print("=" * 80)
    print("HOLDOUT MEDICINE-LEVEL RESULTS")
    print("=" * 80)

    print(
        "Medicines:",
        len(holdout_medicine),
    )

    print(
        "Adaptive better than Chronos:",
        int(
            holdout_medicine[
                "Adaptive_Better_Than_Chronos"
            ].sum()
        ),
        "/",
        len(holdout_medicine),
    )

    print(
        "Adaptive better than TSB:",
        int(
            holdout_medicine[
                "Adaptive_Better_Than_TSB"
            ].sum()
        ),
        "/",
        len(holdout_medicine),
    )

    print(
        "Adaptive better than BOTH:",
        int(
            holdout_medicine[
                "Adaptive_Better_Than_Both"
            ].sum()
        ),
        "/",
        len(holdout_medicine),
    )

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary.to_parquet(
        RESULTS_OUTPUT,
        index=False,
    )

    weight_table.to_parquet(
        WEIGHTS_OUTPUT,
        index=False,
    )

    print()
    print("=" * 80)
    print("SAVED")
    print("=" * 80)

    print(RESULTS_OUTPUT)
    print(WEIGHTS_OUTPUT)


if __name__ == "__main__":
    main()