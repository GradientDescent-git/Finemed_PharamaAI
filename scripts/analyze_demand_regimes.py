import pandas as pd
import numpy as np

path = "data/04_silver/demand_forecasting/chronos_series.parquet"

d = pd.read_parquet(path)

rows = []

for mid, x in d.groupby("MDCODE"):
    x = x.sort_values("INVDT")

    positive = x[x["Demand_Qty"] > 0]

    n = len(x)
    nz = len(positive)

    adi = n / nz if nz else np.inf

    mean = positive["Demand_Qty"].mean() if nz else 0
    std = positive["Demand_Qty"].std(ddof=0) if nz else 0

    cv2 = (std / mean) ** 2 if mean > 0 else np.inf

    rows.append({
        "MDCODE": mid,
        "Days": n,
        "NonZero_Days": nz,
        "ADI": adi,
        "CV2": cv2,
        "Total_Demand": x["Demand_Qty"].sum(),
    })

r = pd.DataFrame(rows)

r["Regime"] = np.select(
    [
        (r["ADI"] < 1.32) & (r["CV2"] < 0.49),
        (r["ADI"] < 1.32) & (r["CV2"] >= 0.49),
        (r["ADI"] >= 1.32) & (r["CV2"] < 0.49),
        (r["ADI"] >= 1.32) & (r["CV2"] >= 0.49),
    ],
    [
        "Smooth",
        "Erratic",
        "Intermittent",
        "Lumpy",
    ],
    default="Unknown",
)

print("=" * 80)
print("FINEMED DEMAND REGIME ANALYSIS")
print("=" * 80)

print()
print("REGIME COUNTS")
print(r["Regime"].value_counts().to_string())

print()
print("REGIME SUMMARY")
print(
    r.groupby("Regime")[
        ["Days", "NonZero_Days", "ADI", "CV2", "Total_Demand"]
    ]
    .mean()
    .to_string()
)

print()
print("MEDICINE-LEVEL RESULTS")
print(
    r.sort_values("Total_Demand", ascending=False)
    .to_string(index=False)
)
