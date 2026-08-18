"""
Frozen Routing Rule Backtest
============================

Purpose
-------
Validate the production candidate routing rule discovered during
routing threshold optimization:

    If Chronos has >= 30% validation error advantage over TSB:
        use Chronos-2-P50
    else:
        use TSB

IMPORTANT
---------
The routing decision is calculated using VALIDATION information only.

The HOLDOUT period remains untouched for evaluation.

This script deliberately does NOT re-optimize the 30% threshold.
The threshold is frozen at 30% before evaluating holdout performance.

Current input artifact contains one validation -> holdout split.
Therefore this script reports a frozen validation/holdout backtest,
not a multi-window rolling backtest.

Once multiple temporal folds are available, this same framework can
be extended to evaluate the frozen rule across all folds.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# PATHS
# =============================================================================

INPUT = Path(
    "data/05_gold/demand_forecasting/"
    "medicine_robustness/medicine_model_robustness.parquet"
)

OUTPUT_DIR = Path(
    "data/05_gold/demand_forecasting/"
    "routing_rule_backtest"
)

OUTPUT = OUTPUT_DIR / "routing_rule_backtest.parquet"
SUMMARY = OUTPUT_DIR / "routing_rule_backtest_summary.parquet"
REGIME_SUMMARY = OUTPUT_DIR / "routing_rule_backtest_regime_summary.parquet"


# =============================================================================
# CONFIGURATION
# =============================================================================

# FROZEN production candidate discovered previously.
ADVANTAGE_THRESHOLD = 30.0

CHRONOS = "chronos-2-P50"
TSB = "tsb"


# =============================================================================
# VALIDATION
# =============================================================================

REQUIRED_COLUMNS = [
    "Split",
    "Medicine_ID",
    "Actual",
    "Chronos_Predicted",
    "Chronos_AE",
    "TSB_Predicted",
    "TSB_AE",
    "Regime",
    "ADI",
    "CV2",
    "Total_Demand",
]


def validate_input(df: pd.DataFrame) -> None:
    """Validate that the input artifact has the required schema."""

    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Input parquet is missing required columns:\n"
            f"{missing}\n\n"
            f"Available columns:\n{df.columns.tolist()}"
        )

    splits = set(df["Split"].dropna().unique())

    if "validation" not in splits:
        raise ValueError(
            "Expected a 'validation' split in the input parquet."
        )

    if "holdout" not in splits:
        raise ValueError(
            "Expected a 'holdout' split in the input parquet."
        )


# =============================================================================
# HELPERS
# =============================================================================

def calculate_wape(
    absolute_error: pd.Series,
    actual: pd.Series,
) -> float:
    """Calculate WAPE safely."""

    total_actual = actual.sum()

    if total_actual == 0:
        return np.nan

    return (
        absolute_error.sum()
        / total_actual
        * 100
    )


def calculate_model_metrics(
    df: pd.DataFrame,
    prediction_column: str,
) -> dict:
    """Calculate aggregate model metrics."""

    predicted = df[prediction_column]

    actual = df["Actual"]

    ae = (actual - predicted).abs()

    return {
        "Medicines": df["Medicine_ID"].nunique(),
        "Actual": actual.sum(),
        "Predicted": predicted.sum(),
        "Absolute_Error": ae.sum(),
        "WAPE": calculate_wape(ae, actual),
    }


def calculate_medicine_metrics(
    df: pd.DataFrame,
    prediction_column: str,
    model_name: str,
) -> pd.DataFrame:
    """Calculate medicine-level metrics."""

    result = (
        df.groupby("Medicine_ID", as_index=False)
        .agg(
            Actual=("Actual", "sum"),
            Predicted=(prediction_column, "sum"),
            Absolute_Error=("Actual", lambda x: 0.0),
        )
    )

    # Recalculate absolute error correctly at row level.
    row_error = (
        df.assign(
            _AE=(
                df["Actual"]
                - df[prediction_column]
            ).abs()
        )
        .groupby("Medicine_ID")["_AE"]
        .sum()
    )

    result["Absolute_Error"] = (
        result["Medicine_ID"]
        .map(row_error)
    )

    result["WAPE"] = np.where(
        result["Actual"] != 0,
        result["Absolute_Error"]
        / result["Actual"]
        * 100,
        np.nan,
    )

    result["Model"] = model_name

    return result


# =============================================================================
# ROUTING RULE
# =============================================================================

def apply_frozen_routing_rule(
    validation: pd.DataFrame,
    holdout: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply the frozen 30% validation-advantage routing rule.

    Validation_Advantage_Pct:

        (TSB_AE - Chronos_AE) / TSB_AE * 100

    Positive value means Chronos has lower validation error.

    Rule:

        >= 30% -> Chronos
        < 30%  -> TSB
    """

    validation = validation.copy()
    holdout = holdout.copy()

    validation["Validation_Advantage_Pct"] = np.where(
        validation["TSB_AE"] != 0,
        (
            (
                validation["TSB_AE"]
                - validation["Chronos_AE"]
            )
            / validation["TSB_AE"]
            * 100
        ),
        np.nan,
    )

    validation["Selected_Model"] = np.where(
        validation["Validation_Advantage_Pct"]
        >= ADVANTAGE_THRESHOLD,
        CHRONOS,
        TSB,
    )

    decisions = validation[
        [
            "Medicine_ID",
            "Validation_Advantage_Pct",
            "Selected_Model",
            "Regime",
            "ADI",
            "CV2",
            "Total_Demand",
        ]
    ].copy()

    # One routing decision per medicine is required.
    if decisions["Medicine_ID"].duplicated().any():
        duplicate_ids = (
            decisions.loc[
                decisions["Medicine_ID"].duplicated(),
                "Medicine_ID",
            ]
            .unique()
            .tolist()
        )

        raise ValueError(
            "Validation contains multiple routing rows per medicine: "
            f"{duplicate_ids}"
        )

    holdout = holdout.merge(
        decisions,
        on="Medicine_ID",
        how="inner",
        suffixes=("", "_Validation"),
    )

    if holdout.empty:
        raise ValueError(
            "No validation/holdout medicines overlap."
        )

    holdout["Selected_Predicted"] = np.where(
        holdout["Selected_Model"] == CHRONOS,
        holdout["Chronos_Predicted"],
        holdout["TSB_Predicted"],
    )

    holdout["Selected_AE"] = (
        holdout["Actual"]
        - holdout["Selected_Predicted"]
    ).abs()

    return validation, holdout


# =============================================================================
# MODEL COMPARISON
# =============================================================================

def build_model_comparison(
    holdout: pd.DataFrame,
) -> pd.DataFrame:
    """Compare routing against fixed baselines."""

    actual = holdout["Actual"]

    chronos_ae = (
        actual
        - holdout["Chronos_Predicted"]
    ).abs()

    tsb_ae = (
        actual
        - holdout["TSB_Predicted"]
    ).abs()

    ensemble_predicted = (
        holdout["Chronos_Predicted"]
        + holdout["TSB_Predicted"]
    ) / 2

    ensemble_ae = (
        actual
        - ensemble_predicted
    ).abs()

    routing_ae = holdout["Selected_AE"]

    rows = [
        {
            "Model": CHRONOS,
            "Medicines": holdout["Medicine_ID"].nunique(),
            "Actual": actual.sum(),
            "Predicted": holdout["Chronos_Predicted"].sum(),
            "Absolute_Error": chronos_ae.sum(),
            "WAPE": calculate_wape(
                chronos_ae,
                actual,
            ),
        },
        {
            "Model": TSB,
            "Medicines": holdout["Medicine_ID"].nunique(),
            "Actual": actual.sum(),
            "Predicted": holdout["TSB_Predicted"].sum(),
            "Absolute_Error": tsb_ae.sum(),
            "WAPE": calculate_wape(
                tsb_ae,
                actual,
            ),
        },
        {
            "Model": "simple_ensemble",
            "Medicines": holdout["Medicine_ID"].nunique(),
            "Actual": actual.sum(),
            "Predicted": ensemble_predicted.sum(),
            "Absolute_Error": ensemble_ae.sum(),
            "WAPE": calculate_wape(
                ensemble_ae,
                actual,
            ),
        },
        {
            "Model": "frozen_30pct_routing",
            "Medicines": holdout["Medicine_ID"].nunique(),
            "Actual": actual.sum(),
            "Predicted": holdout["Selected_Predicted"].sum(),
            "Absolute_Error": routing_ae.sum(),
            "WAPE": calculate_wape(
                routing_ae,
                actual,
            ),
        },
    ]

    result = pd.DataFrame(rows)

    tsb_wape = result.loc[
        result["Model"] == TSB,
        "WAPE",
    ].iloc[0]

    chronos_wape = result.loc[
        result["Model"] == CHRONOS,
        "WAPE",
    ].iloc[0]

    ensemble_wape = result.loc[
        result["Model"] == "simple_ensemble",
        "WAPE",
    ].iloc[0]

    routing_wape = result.loc[
        result["Model"] == "frozen_30pct_routing",
        "WAPE",
    ].iloc[0]

    result["Improvement_vs_TSB"] = (
        tsb_wape - result["WAPE"]
    )

    result["Improvement_vs_Chronos"] = (
        chronos_wape - result["WAPE"]
    )

    result["Improvement_vs_Ensemble"] = (
        ensemble_wape - result["WAPE"]
    )

    return result


# =============================================================================
# MEDICINE LEVEL EVALUATION
# =============================================================================

def build_medicine_results(
    holdout: pd.DataFrame,
) -> pd.DataFrame:
    """Build medicine-level routing evaluation."""

    result = holdout[
        [
            "Medicine_ID",
            "Regime",
            "ADI_Validation",
            "CV2_Validation",
            "Total_Demand_Validation",
            "Validation_Advantage_Pct",
            "Selected_Model",
        ]
    ].drop_duplicates(
        subset=["Medicine_ID"]
    ).copy()

    actual = (
        holdout.groupby("Medicine_ID")["Actual"]
        .sum()
    )

    chronos_error = (
        holdout.assign(
            _AE=(
                holdout["Actual"]
                - holdout["Chronos_Predicted"]
            ).abs()
        )
        .groupby("Medicine_ID")["_AE"]
        .sum()
    )

    tsb_error = (
        holdout.assign(
            _AE=(
                holdout["Actual"]
                - holdout["TSB_Predicted"]
            ).abs()
        )
        .groupby("Medicine_ID")["_AE"]
        .sum()
    )

    routing_error = (
        holdout.groupby("Medicine_ID")["Selected_AE"]
        .sum()
    )

    result["Holdout_Actual"] = (
        result["Medicine_ID"]
        .map(actual)
    )

    result["Chronos_Holdout_AE"] = (
        result["Medicine_ID"]
        .map(chronos_error)
    )

    result["TSB_Holdout_AE"] = (
        result["Medicine_ID"]
        .map(tsb_error)
    )

    result["Routing_Holdout_AE"] = (
        result["Medicine_ID"]
        .map(routing_error)
    )

    result["Chronos_Holdout_WAPE"] = np.where(
        result["Holdout_Actual"] != 0,
        result["Chronos_Holdout_AE"]
        / result["Holdout_Actual"]
        * 100,
        np.nan,
    )

    result["TSB_Holdout_WAPE"] = np.where(
        result["Holdout_Actual"] != 0,
        result["TSB_Holdout_AE"]
        / result["Holdout_Actual"]
        * 100,
        np.nan,
    )

    result["Routing_Holdout_WAPE"] = np.where(
        result["Holdout_Actual"] != 0,
        result["Routing_Holdout_AE"]
        / result["Holdout_Actual"]
        * 100,
        np.nan,
    )

    result["Routing_Better_Than_Chronos"] = (
        result["Routing_Holdout_AE"]
        < result["Chronos_Holdout_AE"]
    )

    result["Routing_Better_Than_TSB"] = (
        result["Routing_Holdout_AE"]
        < result["TSB_Holdout_AE"]
    )

    result["Routing_Better_Than_Both"] = (
        result["Routing_Better_Than_Chronos"]
        & result["Routing_Better_Than_TSB"]
    )

    return result


# =============================================================================
# REGIME ANALYSIS
# =============================================================================

def build_regime_summary(
    medicine_results: pd.DataFrame,
) -> pd.DataFrame:
    """Evaluate routing by demand regime."""

    rows = []

    for regime, group in medicine_results.groupby("Regime"):

        rows.append(
            {
                "Regime": regime,
                "Medicines": len(group),
                "Routing_WAPE": (
                    group["Routing_Holdout_AE"].sum()
                    / group["Holdout_Actual"].sum()
                    * 100
                    if group["Holdout_Actual"].sum() != 0
                    else np.nan
                ),
                "Chronos_WAPE": (
                    group["Chronos_Holdout_AE"].sum()
                    / group["Holdout_Actual"].sum()
                    * 100
                    if group["Holdout_Actual"].sum() != 0
                    else np.nan
                ),
                "TSB_WAPE": (
                    group["TSB_Holdout_AE"].sum()
                    / group["Holdout_Actual"].sum()
                    * 100
                    if group["Holdout_Actual"].sum() != 0
                    else np.nan
                ),
                "Chronos_Selected": int(
                    (
                        group["Selected_Model"]
                        == CHRONOS
                    ).sum()
                ),
                "TSB_Selected": int(
                    (
                        group["Selected_Model"]
                        == TSB
                    ).sum()
                ),
                "Routing_Better_Than_TSB": int(
                    group["Routing_Better_Than_TSB"].sum()
                ),
                "Routing_Better_Than_Chronos": int(
                    group["Routing_Better_Than_Chronos"].sum()
                ),
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# SUMMARY
# =============================================================================

def build_summary(
    validation: pd.DataFrame,
    holdout: pd.DataFrame,
    model_comparison: pd.DataFrame,
    medicine_results: pd.DataFrame,
) -> pd.DataFrame:
    """Build one-row summary artifact."""

    routing_row = model_comparison.loc[
        model_comparison["Model"]
        == "frozen_30pct_routing"
    ].iloc[0]

    tsb_row = model_comparison.loc[
        model_comparison["Model"] == TSB
    ].iloc[0]

    chronos_row = model_comparison.loc[
        model_comparison["Model"] == CHRONOS
    ].iloc[0]

    ensemble_row = model_comparison.loc[
        model_comparison["Model"]
        == "simple_ensemble"
    ].iloc[0]

    chronos_selected = (
        medicine_results["Selected_Model"]
        == CHRONOS
    ).sum()

    tsb_selected = (
        medicine_results["Selected_Model"]
        == TSB
    ).sum()

    return pd.DataFrame(
        [
            {
                "Backtest_Type": (
                    "Frozen validation-to-holdout"
                ),
                "Advantage_Threshold_Pct": (
                    ADVANTAGE_THRESHOLD
                ),
                "Validation_Medicines": (
                    validation["Medicine_ID"]
                    .nunique()
                ),
                "Holdout_Medicines": (
                    holdout["Medicine_ID"]
                    .nunique()
                ),
                "Chronos_Selected": int(
                    chronos_selected
                ),
                "TSB_Selected": int(
                    tsb_selected
                ),
                "Chronos_Selection_Pct": (
                    chronos_selected
                    / len(medicine_results)
                    * 100
                ),
                "TSB_Selection_Pct": (
                    tsb_selected
                    / len(medicine_results)
                    * 100
                ),
                "Routing_WAPE": routing_row["WAPE"],
                "TSB_WAPE": tsb_row["WAPE"],
                "Chronos_WAPE": chronos_row["WAPE"],
                "Ensemble_WAPE": ensemble_row["WAPE"],
                "Routing_Improvement_vs_TSB": (
                    routing_row[
                        "Improvement_vs_TSB"
                    ]
                ),
                "Routing_Improvement_vs_Chronos": (
                    routing_row[
                        "Improvement_vs_Chronos"
                    ]
                ),
                "Routing_Improvement_vs_Ensemble": (
                    routing_row[
                        "Improvement_vs_Ensemble"
                    ]
                ),
                "Routing_Better_Than_TSB": int(
                    medicine_results[
                        "Routing_Better_Than_TSB"
                    ].sum()
                ),
                "Routing_Better_Than_Chronos": int(
                    medicine_results[
                        "Routing_Better_Than_Chronos"
                    ].sum()
                ),
                "Routing_Better_Than_Both": int(
                    medicine_results[
                        "Routing_Better_Than_Both"
                    ].sum()
                ),
            }
        ]
    )


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 80)
    print("FROZEN ROUTING RULE BACKTEST")
    print("=" * 80)

    print()
    print(
        "Frozen rule:"
    )
    print(
        f"Chronos if validation advantage >= "
        f"{ADVANTAGE_THRESHOLD:.0f}%"
    )
    print("Otherwise: TSB")

    print()
    print("Input:", INPUT)

    # -------------------------------------------------------------------------
    # Load
    # -------------------------------------------------------------------------

    if not INPUT.exists():
        raise FileNotFoundError(
            f"Input file does not exist:\n{INPUT}"
        )

    df = pd.read_parquet(INPUT)

    print()
    print("Input rows:", len(df))
    print(
        "Medicines:",
        df["Medicine_ID"].nunique(),
    )

    validate_input(df)

    # -------------------------------------------------------------------------
    # Split
    # -------------------------------------------------------------------------

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
        validation["Medicine_ID"]
        .isin(comparable)
    ].copy()

    holdout = holdout[
        holdout["Medicine_ID"]
        .isin(comparable)
    ].copy()

    print(
        "Validation medicines:",
        validation["Medicine_ID"].nunique(),
    )

    print(
        "Holdout medicines:",
        holdout["Medicine_ID"].nunique(),
    )

    print(
        "Comparable medicines:",
        len(comparable),
    )

    # -------------------------------------------------------------------------
    # Apply frozen rule
    # -------------------------------------------------------------------------

    validation, routed_holdout = (
        apply_frozen_routing_rule(
            validation,
            holdout,
        )
    )

    # -------------------------------------------------------------------------
    # Routing distribution
    # -------------------------------------------------------------------------

    decisions = (
        validation[
            [
                "Medicine_ID",
                "Selected_Model",
                "Validation_Advantage_Pct",
            ]
        ]
        .drop_duplicates("Medicine_ID")
    )

    chronos_selected = (
        decisions["Selected_Model"]
        == CHRONOS
    ).sum()

    tsb_selected = (
        decisions["Selected_Model"]
        == TSB
    ).sum()

    print()
    print("=" * 80)
    print("ROUTING DISTRIBUTION")
    print("=" * 80)

    print(
        decisions["Selected_Model"]
        .value_counts()
        .to_string()
    )

    print()
    print(
        "Chronos selected:",
        int(chronos_selected),
    )

    print(
        "TSB selected:",
        int(tsb_selected),
    )

    print(
        "Chronos selection:",
        round(
            chronos_selected
            / len(decisions)
            * 100,
            2,
        ),
        "%",
    )

    # -------------------------------------------------------------------------
    # Model comparison
    # -------------------------------------------------------------------------

    model_comparison = build_model_comparison(
        routed_holdout
    )

    print()
    print("=" * 80)
    print("HOLDOUT MODEL COMPARISON")
    print("=" * 80)

    print(
        model_comparison[
            [
                "Model",
                "Medicines",
                "Actual",
                "Predicted",
                "Absolute_Error",
                "WAPE",
                "Improvement_vs_TSB",
                "Improvement_vs_Chronos",
                "Improvement_vs_Ensemble",
            ]
        ]
        .round(3)
        .to_string(index=False)
    )

    # -------------------------------------------------------------------------
    # Medicine-level
    # -------------------------------------------------------------------------

    medicine_results = build_medicine_results(
        routed_holdout
    )

    print()
    print("=" * 80)
    print("MEDICINE-LEVEL RESULTS")
    print("=" * 80)

    print(
        "Routing better than Chronos:",
        int(
            medicine_results[
                "Routing_Better_Than_Chronos"
            ].sum()
        ),
        "/",
        len(medicine_results),
    )

    print(
        "Routing better than TSB:",
        int(
            medicine_results[
                "Routing_Better_Than_TSB"
            ].sum()
        ),
        "/",
        len(medicine_results),
    )

    print(
        "Routing better than BOTH:",
        int(
            medicine_results[
                "Routing_Better_Than_Both"
            ].sum()
        ),
        "/",
        len(medicine_results),
    )

    # -------------------------------------------------------------------------
    # Regime
    # -------------------------------------------------------------------------

    regime_summary = build_regime_summary(
        medicine_results
    )

    print()
    print("=" * 80)
    print("PERFORMANCE BY REGIME")
    print("=" * 80)

    print(
        regime_summary.round(3)
        .to_string(index=False)
    )

    # -------------------------------------------------------------------------
    # Validation rule diagnostics
    # -------------------------------------------------------------------------

    print()
    print("=" * 80)
    print("VALIDATION ADVANTAGE DISTRIBUTION")
    print("=" * 80)

    advantage = validation[
        "Validation_Advantage_Pct"
    ].dropna()

    print(
        "Minimum:",
        round(advantage.min(), 3),
    )

    print(
        "Median:",
        round(advantage.median(), 3),
    )

    print(
        "Mean:",
        round(advantage.mean(), 3),
    )

    print(
        "Maximum:",
        round(advantage.max(), 3),
    )

    print(
        "Medicines >= threshold:",
        int(
            (
                advantage
                >= ADVANTAGE_THRESHOLD
            ).sum()
        ),
    )

    # -------------------------------------------------------------------------
    # Final verdict
    # -------------------------------------------------------------------------

    summary = build_summary(
        validation,
        routed_holdout,
        model_comparison,
        medicine_results,
    )

    routing_wape = summary.loc[
        0,
        "Routing_WAPE",
    ]

    tsb_wape = summary.loc[
        0,
        "TSB_WAPE",
    ]

    chronos_wape = summary.loc[
        0,
        "Chronos_WAPE",
    ]

    ensemble_wape = summary.loc[
        0,
        "Ensemble_WAPE",
    ]

    print()
    print("=" * 80)
    print("FINAL VERDICT")
    print("=" * 80)

    if routing_wape < tsb_wape:
        print(
            "PASS: Frozen routing beats TSB on holdout."
        )
    else:
        print(
            "FAIL: Frozen routing does not beat TSB."
        )

    if routing_wape < chronos_wape:
        print(
            "PASS: Frozen routing beats Chronos."
        )
    else:
        print(
            "NOTE: Frozen routing does not beat Chronos."
        )

    if routing_wape < ensemble_wape:
        print(
            "PASS: Frozen routing beats 50/50 ensemble."
        )
    else:
        print(
            "NOTE: Frozen routing does not beat "
            "50/50 ensemble."
        )

    print()
    print(
        "Routing WAPE:",
        round(routing_wape, 3),
    )

    print(
        "TSB WAPE:",
        round(tsb_wape, 3),
    )

    print(
        "Chronos WAPE:",
        round(chronos_wape, 3),
    )

    print(
        "50/50 Ensemble WAPE:",
        round(ensemble_wape, 3),
    )

    print()
    print(
        "Improvement vs TSB:",
        round(
            tsb_wape - routing_wape,
            3,
        ),
    )

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Main medicine-level artifact.
    medicine_results.to_parquet(
        OUTPUT,
        index=False,
    )

    # One-row overall summary.
    summary.to_parquet(
        SUMMARY,
        index=False,
    )

    # Regime-level artifact.
    regime_summary.to_parquet(
        REGIME_SUMMARY,
        index=False,
    )

    print()
    print("=" * 80)
    print("SAVED")
    print("=" * 80)

    print(OUTPUT)
    print(SUMMARY)
    print(REGIME_SUMMARY)


if __name__ == "__main__":
    main()