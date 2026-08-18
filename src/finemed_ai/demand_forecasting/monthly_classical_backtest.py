import pandas as pd
import numpy as np
from pathlib import Path

INPUT = "data/05_gold/demand_forecasting/monthly_experiment/monthly_demand.parquet"
OUTPUT = Path("data/05_gold/demand_forecasting/monthly_experiment")
OUTPUT.mkdir(parents=True, exist_ok=True)

df = pd.read_parquet(INPUT)
df["Month"] = pd.to_datetime(df["Month"])
df["Medicine_ID"] = df["Medicine_ID"].astype(str)

# Locked evaluation windows
cutoffs = pd.to_datetime([
    "2025-11-01",
    "2025-12-01",
    "2026-01-01",
    "2026-02-01",
    "2026-03-01",
    "2026-04-01",
])

VALIDATION = set(cutoffs[:4])
HOLDOUT = set(cutoffs[4:])

MIN_HISTORY = 12


def naive(history):
    return float(history.iloc[-1])


def seasonal_naive(history):
    if len(history) < 12:
        return naive(history)
    return float(history.iloc[-12])


def croston(history):
    y = np.asarray(history, dtype=float)
    nz = np.flatnonzero(y > 0)

    if len(nz) == 0:
        return 0.0

    sizes = y[nz]

    if len(nz) == 1:
        intervals = np.array([len(y)])
    else:
        intervals = np.diff(np.r_[-1, nz])

    return float(sizes.mean() / intervals.mean())


def sba(history):
    return 0.95 * croston(history)


def tsb(history, alpha=0.1, beta=0.1):
    y = np.asarray(history, dtype=float)

    if len(y) == 0:
        return 0.0

    p = 0.5
    z = y[y > 0]

    if len(z) == 0:
        return 0.0

    demand = float(z[0])

    for value in y:
        occurrence = 1.0 if value > 0 else 0.0

        p = p + beta * (occurrence - p)

        if value > 0:
            demand = demand + alpha * (value - demand)

    return float(p * demand)


MODELS = {
    "naive": naive,
    "seasonal_naive": seasonal_naive,
    "croston": croston,
    "sba": sba,
    "tsb": tsb,
}

rows = []

for cutoff in cutoffs:

    target_month = cutoff + pd.offsets.MonthBegin(1)

    train = df[df["Month"] <= cutoff]
    actual = df[df["Month"] == target_month][
        ["Medicine_ID", "Actual"]
    ]

    if actual.empty:
        continue

    for medicine_id, g in train.groupby("Medicine_ID"):

        g = g.sort_values("Month")

        # Require at least 12 months of history
        if len(g) < MIN_HISTORY:
            continue

        history = g["Actual"].reset_index(drop=True)

        actual_row = actual[
            actual["Medicine_ID"] == medicine_id
        ]

        if actual_row.empty:
            continue

        actual_value = float(actual_row["Actual"].iloc[0])

        for model_name, model_fn in MODELS.items():

            try:
                prediction = max(float(model_fn(history)), 0.0)
            except Exception:
                prediction = np.nan

            if np.isnan(prediction):
                continue

            error = abs(actual_value - prediction)

            rows.append({
                "Cutoff_Date": cutoff,
                "Forecast_Month": target_month,
                "Medicine_ID": medicine_id,
                "Model": model_name,
                "Actual": actual_value,
                "Predicted": prediction,
                "Absolute_Error": error,
            })


results = pd.DataFrame(rows)

# Metrics
summary = (
    results
    .groupby(["Model", "Cutoff_Date"])
    .agg(
        Medicines=("Medicine_ID", "nunique"),
        Actual=("Actual", "sum"),
        Predicted=("Predicted", "sum"),
        Absolute_Error=("Absolute_Error", "sum"),
        MAE=("Absolute_Error", "mean"),
    )
    .reset_index()
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

print("=" * 80)
print("MONTHLY CLASSICAL FORECASTING BENCHMARK")
print("=" * 80)

print()
print("Validation cutoffs:")
for x in sorted(VALIDATION):
    print(" ", x.date())

print()
print("Holdout cutoffs:")
for x in sorted(HOLDOUT):
    print(" ", x.date())

print()
print("Minimum history:", MIN_HISTORY, "months")

print()
print("=== VALIDATION PERFORMANCE ===")

v = summary[summary["Split"] == "validation"].copy()

vg = (
    v.groupby("Model")
    .agg(
        Medicines=("Medicines", "mean"),
        Actual=("Actual", "sum"),
        Predicted=("Predicted", "sum"),
        Absolute_Error=("Absolute_Error", "sum"),
        MAE=("MAE", "mean"),
    )
)

vg["WAPE"] = vg["Absolute_Error"] / vg["Actual"] * 100
vg["Ratio"] = vg["Predicted"] / vg["Actual"]

print(
    vg.sort_values("WAPE")
    .to_string()
)

print()
print("=== HOLDOUT PERFORMANCE ===")

h = summary[summary["Split"] == "holdout"].copy()

hg = (
    h.groupby("Model")
    .agg(
        Medicines=("Medicines", "mean"),
        Actual=("Actual", "sum"),
        Predicted=("Predicted", "sum"),
        Absolute_Error=("Absolute_Error", "sum"),
        MAE=("MAE", "mean"),
    )
)

hg["WAPE"] = hg["Absolute_Error"] / hg["Actual"] * 100
hg["Ratio"] = hg["Predicted"] / hg["Actual"]

print(
    hg.sort_values("WAPE")
    .to_string()
)

results.to_parquet(
    OUTPUT / "classical_monthly_backtest.parquet",
    index=False,
)

summary.to_parquet(
    OUTPUT / "classical_monthly_summary.parquet",
    index=False,
)

print()
print("Saved:")
print(OUTPUT / "classical_monthly_backtest.parquet")
print(OUTPUT / "classical_monthly_summary.parquet")
