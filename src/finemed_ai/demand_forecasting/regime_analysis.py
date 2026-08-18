import pandas as pd
import numpy as np
from pathlib import Path

INPUT = "data/04_silver/demand_forecasting/chronos_series.parquet"
OUTPUT_DIR = Path("data/05_gold/demand_forecasting/regime_analysis")
OUTPUT = OUTPUT_DIR / "medicine_regimes.parquet"

df = pd.read_parquet(INPUT)

df["INVDT"] = pd.to_datetime(df["INVDT"])

df = df.sort_values(
    ["MDCODE", "INVDT"]
)

rows = []

for medicine_id, group in df.groupby(
    "MDCODE",
    sort=True,
):
    y = (
        group["Demand_Qty"]
        .astype(float)
        .to_numpy()
    )

    non_zero = y[y > 0]

    if len(non_zero) == 0:
        adi = np.inf
        cv2 = 0.0
    else:
        adi = len(y) / len(non_zero)

        if len(non_zero) > 1 and np.mean(non_zero) > 0:
            cv2 = (
                np.var(
                    non_zero,
                    ddof=1,
                )
                / np.mean(non_zero) ** 2
            )
        else:
            cv2 = 0.0

    if adi >= 1.32 and cv2 < 0.49:
        regime = "Intermittent"

    elif adi >= 1.32 and cv2 >= 0.49:
        regime = "Lumpy"

    elif adi < 1.32 and cv2 < 0.49:
        regime = "Smooth"

    else:
        regime = "Erratic"

    rows.append(
        {
            "Medicine_ID": str(medicine_id),
            "Days": len(y),
            "NonZero_Days": len(non_zero),
            "ADI": adi,
            "CV2": cv2,
            "Total_Demand": float(y.sum()),
            "Regime": regime,
        }
    )

regimes = pd.DataFrame(rows)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

regimes.to_parquet(
    OUTPUT,
    index=False,
)

print("=" * 80)
print("MEDICINE DEMAND REGIME ANALYSIS")
print("=" * 80)

print()
print("Medicines:", len(regimes))

print()
print("Regime counts:")
print(
    regimes["Regime"]
    .value_counts()
    .to_string()
)

print()
print("Regime summary:")
print(
    regimes
    .groupby("Regime")
    .agg(
        Medicines=("Medicine_ID", "count"),
        Avg_ADI=("ADI", "mean"),
        Avg_CV2=("CV2", "mean"),
        Total_Demand=("Total_Demand", "sum"),
    )
    .round(3)
    .to_string()
)

print()
print("Medicine classifications:")
print(
    regimes.to_string(index=False)
)

print()
print("Saved:", OUTPUT)
