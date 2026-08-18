import pandas as pd
import numpy as np
from pathlib import Path

INPUT = "data/04_silver/demand_forecasting/chronos_series.parquet"
OUTPUT = Path("data/05_gold/demand_forecasting/monthly_experiment")
OUTPUT.mkdir(parents=True, exist_ok=True)

df = pd.read_parquet(INPUT)

df["INVDT"] = pd.to_datetime(df["INVDT"])
df["MDCODE"] = df["MDCODE"].astype(str)

# Aggregate daily demand into monthly demand
monthly = (
    df.assign(Month=df["INVDT"].dt.to_period("M").dt.to_timestamp())
      .groupby(["MDCODE", "Month"], as_index=False)["Demand_Qty"]
      .sum()
      .rename(columns={"MDCODE": "Medicine_ID", "Demand_Qty": "Actual"})
)

# Create complete monthly calendar per medicine
all_rows = []

for medicine_id, g in monthly.groupby("Medicine_ID"):
    g = g.sort_values("Month")

    full_months = pd.date_range(
        g["Month"].min(),
        g["Month"].max(),
        freq="MS"
    )

    tmp = (
        g.set_index("Month")
         .reindex(full_months, fill_value=0)
         .rename_axis("Month")
         .reset_index()
    )

    tmp["Medicine_ID"] = medicine_id
    all_rows.append(tmp)

monthly = pd.concat(all_rows, ignore_index=True)

monthly = monthly[
    ["Medicine_ID", "Month", "Actual"]
].sort_values(["Medicine_ID", "Month"])

print("=" * 80)
print("MONTHLY DEMAND DATASET")
print("=" * 80)
print("Medicines:", monthly["Medicine_ID"].nunique())
print("Rows:", len(monthly))
print("Date range:", monthly["Month"].min(), "->", monthly["Month"].max())
print("Total demand:", monthly["Actual"].sum())
print()

# Basic monthly statistics
stats = (
    monthly.groupby("Medicine_ID")["Actual"]
    .agg(["count", "mean", "std", "sum"])
)

stats["CV2"] = (
    stats["std"] / stats["mean"]
).replace([np.inf, -np.inf], np.nan) ** 2

stats["Zero_Month_Pct"] = (
    monthly.assign(Zero=monthly["Actual"].eq(0))
    .groupby("Medicine_ID")["Zero"]
    .mean()
    * 100
)

print("Median CV2:", round(stats["CV2"].median(), 4))
print("Mean CV2:", round(stats["CV2"].mean(), 4))
print("Median zero-month %:", round(stats["Zero_Month_Pct"].median(), 2))
print("Mean zero-month %:", round(stats["Zero_Month_Pct"].mean(), 2))
print()

# Save monthly dataset
out = OUTPUT / "monthly_demand.parquet"
monthly.to_parquet(out, index=False)

print("Saved:", out)
