import pandas as pd
from pathlib import Path


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

ROUTING_PATH = Path(
    "data/05_gold/demand_forecasting/model_routing/"
    "model_routing_rules.parquet"
)

OUTPUT_DIR = Path(
    "data/05_gold/demand_forecasting/model_selection"
)

OUTPUT_RULES = OUTPUT_DIR / "production_model_rules.parquet"
OUTPUT_SUMMARY = OUTPUT_DIR / "model_selection_summary.parquet"


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


def load_data():
    chronos = pd.read_parquet(CHRONOS_PATH)
    classical = pd.read_parquet(CLASSICAL_PATH)
    regimes = pd.read_parquet(REGIME_PATH)

    chronos = chronos[
        chronos["Model"] == "chronos-2-P50"
    ].copy()

    classical = classical[
        classical["Model"] == "tsb"
    ].copy()

    chronos["Cutoff_Date"] = pd.to_datetime(
        chronos["Cutoff_Date"]
    )

    classical["Cutoff_Date"] = pd.to_datetime(
        classical["Cutoff_Date"]
    )

    regimes["Medicine_ID"] = regimes["Medicine_ID"].astype(str)

    return chronos, classical, regimes


def add_split(df):
    df = df.copy()

    df["Split"] = "unknown"

    df.loc[
        df["Cutoff_Date"].isin(VALIDATION_CUTOFFS),
        "Split"
    ] = "validation"

    df.loc[
        df["Cutoff_Date"].isin(HOLDOUT_CUTOFFS),
        "Split"
    ] = "holdout"

    return df


def calculate_wape(actual, absolute_error):
    actual_total = actual.sum()
    error_total = absolute_error.sum()

    if actual_total == 0:
        return float("inf")

    return error_total / actual_total * 100


def build_comparison(chronos, classical, regimes):
    chronos = add_split(chronos)
    classical = add_split(classical)

    c = (
        chronos
        .groupby(
            ["Split", "Medicine_ID"],
            as_index=False,
        )
        .agg(
            Actual=("Actual", "sum"),
            Chronos_Predicted=("Predicted", "sum"),
            Chronos_AE=("Absolute_Error", "sum"),
        )
    )

    t = (
        classical
        .groupby(
            ["Split", "Medicine_ID"],
            as_index=False,
        )
        .agg(
            TSB_Predicted=("Predicted", "sum"),
            TSB_AE=("Absolute_Error", "sum"),
        )
    )

    comparison = c.merge(
        t,
        on=["Split", "Medicine_ID"],
        how="inner",
    )

    comparison = comparison.merge(
        regimes[
            [
                "Medicine_ID",
                "Regime",
                "ADI",
                "CV2",
            ]
        ],
        on="Medicine_ID",
        how="left",
    )

    comparison["Chronos_WAPE"] = (
        comparison["Chronos_AE"]
        / comparison["Actual"]
        * 100
    )

    comparison["TSB_WAPE"] = (
        comparison["TSB_AE"]
        / comparison["Actual"]
        * 100
    )

    comparison["Validation_Better_Model"] = comparison.apply(
        lambda row:
        "chronos-2-P50"
        if row["Chronos_AE"] < row["TSB_AE"]
        else "tsb",
        axis=1,
    )

    return comparison


def build_validation_rules(comparison):
    validation = comparison[
        comparison["Split"] == "validation"
    ].copy()

    rules = []

    for regime, group in validation.groupby("Regime"):
        chronos_error = group["Chronos_AE"].sum()
        tsb_error = group["TSB_AE"].sum()

        actual = group["Actual"].sum()

        chronos_wape = (
            chronos_error / actual * 100
            if actual > 0
            else float("inf")
        )

        tsb_wape = (
            tsb_error / actual * 100
            if actual > 0
            else float("inf")
        )

        if chronos_wape < tsb_wape:
            selected_model = "chronos-2-P50"
        else:
            selected_model = "tsb"

        rules.append(
            {
                "Regime": regime,
                "Selected_Model": selected_model,
                "Medicines": group["Medicine_ID"].nunique(),
                "Chronos_WAPE": chronos_wape,
                "TSB_WAPE": tsb_wape,
                "Selection_Basis": "validation_only",
            }
        )

    return pd.DataFrame(rules)


def build_overall_summary(comparison, rules):
    rows = []

    for split in ["validation", "holdout"]:
        data = comparison[
            comparison["Split"] == split
        ].copy()

        if data.empty:
            continue

        chronos_actual = data["Actual"].sum()
        chronos_error = data["Chronos_AE"].sum()

        tsb_error = data["TSB_AE"].sum()

        routed_error = 0.0
        routed_prediction = 0.0

        for _, row in data.iterrows():
            rule = rules[
                rules["Regime"] == row["Regime"]
            ]

            if rule.empty:
                selected = "tsb"
            else:
                selected = rule.iloc[0]["Selected_Model"]

            if selected == "chronos-2-P50":
                routed_error += row["Chronos_AE"]
                routed_prediction += row["Chronos_Predicted"]
            else:
                routed_error += row["TSB_AE"]
                routed_prediction += row["TSB_Predicted"]

        rows.extend(
            [
                {
                    "Split": split,
                    "Model": "chronos-2-P50",
                    "Medicines": data["Medicine_ID"].nunique(),
                    "Actual": chronos_actual,
                    "Absolute_Error": chronos_error,
                    "WAPE": (
                        chronos_error
                        / chronos_actual
                        * 100
                    ),
                },
                {
                    "Split": split,
                    "Model": "tsb",
                    "Medicines": data["Medicine_ID"].nunique(),
                    "Actual": chronos_actual,
                    "Absolute_Error": tsb_error,
                    "WAPE": (
                        tsb_error
                        / chronos_actual
                        * 100
                    ),
                },
                {
                    "Split": split,
                    "Model": "routed_hybrid",
                    "Medicines": data["Medicine_ID"].nunique(),
                    "Actual": chronos_actual,
                    "Absolute_Error": routed_error,
                    "WAPE": (
                        routed_error
                        / chronos_actual
                        * 100
                    ),
                },
            ]
        )

    return pd.DataFrame(rows)


def main():
    print("=" * 80)
    print("PRODUCTION MODEL SELECTION")
    print("=" * 80)

    chronos, classical, regimes = load_data()

    print()
    print("Chronos rows:", len(chronos))
    print("TSB rows:", len(classical))
    print("Regime medicines:", len(regimes))

    comparison = build_comparison(
        chronos,
        classical,
        regimes,
    )

    print()
    print("Comparison rows:", len(comparison))
    print(
        "Medicines:",
        comparison["Medicine_ID"].nunique(),
    )

    rules = build_validation_rules(
        comparison
    )

    print()
    print("=" * 80)
    print("VALIDATION-DERIVED PRODUCTION RULES")
    print("=" * 80)

    print(
        rules.to_string(index=False)
    )

    summary = build_overall_summary(
        comparison,
        rules,
    )

    print()
    print("=" * 80)
    print("MODEL PERFORMANCE")
    print("=" * 80)

    print(
        summary.to_string(index=False)
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    rules.to_parquet(
        OUTPUT_RULES,
        index=False,
    )

    summary.to_parquet(
        OUTPUT_SUMMARY,
        index=False,
    )

    print()
    print("=" * 80)
    print("SAVED")
    print("=" * 80)

    print(OUTPUT_RULES)
    print(OUTPUT_SUMMARY)


if __name__ == "__main__":
    main()