from __future__ import annotations

from pathlib import Path

import pandas as pd


DEFAULT_INPUT = Path(
    "data/04_silver/demand_forecasting/daily_demand.parquet"
)

DEFAULT_OUTPUT = Path(
    "data/04_silver/demand_forecasting/series_quality.parquet"
)


def analyze_series_quality(
    input_path: str | Path = DEFAULT_INPUT,
    output_path: str | Path = DEFAULT_OUTPUT,
) -> pd.DataFrame:
    """
    Analyze demand-series quality for every medicine.

    The input is the observed daily-demand dataset produced by
    data_preparation.py.

    This function does NOT fill missing dates with zero.
    Missing observations must be interpreted before doing that.
    """

    input_path = Path(input_path)
    output_path = Path(output_path)

    df = pd.read_parquet(input_path)

    required_columns = {"MDCODE", "INVDT", "Demand_Qty"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    df = df.copy()

    df["INVDT"] = pd.to_datetime(df["INVDT"])
    df["MDCODE"] = df["MDCODE"].astype(str)
    df["Demand_Qty"] = pd.to_numeric(df["Demand_Qty"])

    # One row per medicine/date should already be guaranteed by
    # data_preparation.py.
    duplicates = df.duplicated(["MDCODE", "INVDT"]).sum()

    if duplicates:
        raise ValueError(
            f"Found {duplicates} duplicate MDCODE/INVDT rows."
        )

    records = []

    for mdcode, group in df.groupby("MDCODE", sort=True):
        group = group.sort_values("INVDT")

        first_date = group["INVDT"].min()
        last_date = group["INVDT"].max()

        calendar_days = (
            last_date - first_date
        ).days + 1

        observation_count = len(group)

        missing_days = calendar_days - observation_count

        active_days = int((group["Demand_Qty"] > 0).sum())

        zero_days_observed = int(
            (group["Demand_Qty"] == 0).sum()
        )

        total_demand = group["Demand_Qty"].sum()

        mean_demand = group["Demand_Qty"].mean()

        median_demand = group["Demand_Qty"].median()

        std_demand = group["Demand_Qty"].std()

        max_demand = group["Demand_Qty"].max()

        activity_rate = (
            observation_count / calendar_days
            if calendar_days > 0
            else 0.0
        )

        records.append(
            {
                "MDCODE": mdcode,
                "first_sale_date": first_date,
                "last_sale_date": last_date,
                "observation_count": observation_count,
                "calendar_days": calendar_days,
                "missing_days": missing_days,
                "active_days": active_days,
                "zero_days_observed": zero_days_observed,
                "activity_rate": activity_rate,
                "total_demand": total_demand,
                "mean_demand": mean_demand,
                "median_demand": median_demand,
                "std_demand": std_demand,
                "max_demand": max_demand,
            }
        )

    quality_df = pd.DataFrame(records)

    # ------------------------------------------------------------------
    # Initial eligibility classification
    #
    # These thresholds are intentionally conservative.
    # They are NOT the final business rules.
    # ------------------------------------------------------------------

        # ------------------------------------------------------------------
    # Demand-pattern classification
    #
    # activity_rate = proportion of calendar days with actual demand.
    # This is different from observation density.
    # ------------------------------------------------------------------

    quality_df["demand_activity_rate"] = (
        quality_df["active_days"]
        / quality_df["calendar_days"]
    )

    def classify(row: pd.Series) -> str:

        observations = row["observation_count"]
        demand_activity = row["demand_activity_rate"]

        if observations < 30:
            return "INSUFFICIENT_HISTORY"

        if observations < 90:
            return "SPARSE_HISTORY"

        if demand_activity < 0.05:
            return "HIGHLY_INTERMITTENT"

        if demand_activity < 0.20:
            return "INTERMITTENT"

        return "REGULAR"

    quality_df["series_class"] = quality_df.apply(
        classify,
        axis=1,
    )

    # Chronos candidate is deliberately separate from classification.
    quality_df["forecast_candidate"] = (
        (quality_df["observation_count"] >= 90)
        & (quality_df["calendar_days"] >= 180))

    quality_df = quality_df.sort_values(
        ["forecast_candidate", "observation_count"],
        ascending=[False, False],
    ).reset_index(drop=True)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    quality_df.to_parquet(
        output_path,
        index=False,
    )

    return quality_df


if __name__ == "__main__":
    result = analyze_series_quality()

    print("=" * 70)
    print("DEMAND SERIES QUALITY REPORT")
    print("=" * 70)

    print(f"Medicines: {len(result)}")
    print(
        "Forecast candidates:",
        int(result["forecast_candidate"].sum()),
    )

    print("\nSeries classification:")
    print(
        result["series_class"]
        .value_counts()
        .to_string()
    )

    print("\nQuality summary:")
    print(
        result[
            [
                "observation_count",
                "calendar_days",
                "missing_days",
                "activity_rate",
                "mean_demand",
                "max_demand",
            ]
        ]
        .describe()
        .to_string()
    )

    print("\nSaved:")
    print(DEFAULT_OUTPUT)