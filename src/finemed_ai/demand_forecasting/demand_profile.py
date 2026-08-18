from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_INPUT = Path(
    "data/04_silver/demand_forecasting/chronos_series.parquet"
)

DEFAULT_OUTPUT = Path(
    "data/04_silver/demand_forecasting/demand_profile.parquet"
)


def classify_demand(adi: float, cv2: float) -> str:
    """
    Syntetos-Boylan style demand classification.

    ADI:
        Average Demand Interval.

    CV2:
        Squared coefficient of variation of non-zero demand.

    Thresholds:
        ADI < 1.32 and CV2 < 0.49 -> SMOOTH
        ADI < 1.32 and CV2 >= 0.49 -> ERRATIC
        ADI >= 1.32 and CV2 < 0.49 -> INTERMITTENT
        ADI >= 1.32 and CV2 >= 0.49 -> LUMPY
    """

    if adi < 1.32 and cv2 < 0.49:
        return "SMOOTH"

    if adi < 1.32 and cv2 >= 0.49:
        return "ERRATIC"

    if adi >= 1.32 and cv2 < 0.49:
        return "INTERMITTENT"

    return "LUMPY"


def analyze_demand_profile(
    input_path: str | Path = DEFAULT_INPUT,
    output_path: str | Path = DEFAULT_OUTPUT,
) -> pd.DataFrame:

    input_path = Path(input_path)
    output_path = Path(output_path)

    df = pd.read_parquet(input_path)

    required = {
        "MDCODE",
        "INVDT",
        "Demand_Qty",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Input dataset missing columns: {sorted(missing)}"
        )

    df = df.copy()

    df["MDCODE"] = (
        df["MDCODE"]
        .astype(str)
        .str.strip()
    )

    df["INVDT"] = pd.to_datetime(
        df["INVDT"],
        errors="coerce",
    )

    df["Demand_Qty"] = pd.to_numeric(
        df["Demand_Qty"],
        errors="coerce",
    )

    if df["INVDT"].isna().any():
        raise ValueError("Invalid INVDT values found.")

    if df["Demand_Qty"].isna().any():
        raise ValueError("Invalid Demand_Qty values found.")

    if (df["Demand_Qty"] < 0).any():
        raise ValueError("Negative demand found.")

    if df.duplicated(["MDCODE", "INVDT"]).any():
        raise ValueError(
            "Duplicate MDCODE/INVDT rows found."
        )

    records = []

    for mdcode, group in df.groupby("MDCODE", sort=True):

        group = group.sort_values("INVDT").copy()

        dates = group["INVDT"]
        demand = group["Demand_Qty"].astype(float)

        first_date = dates.min()
        last_date = dates.max()

        calendar_days = (
            last_date - first_date
        ).days + 1

        total_demand = float(demand.sum())

        positive = demand[demand > 0]

        positive_days = int(len(positive))

        zero_days = int(
            (demand == 0).sum()
        )

        if positive_days > 0:
            mean_positive = float(
                positive.mean()
            )

            median_positive = float(
                positive.median()
            )

            std_positive = float(
                positive.std(ddof=1)
            ) if positive_days > 1 else 0.0

            cv = (
                std_positive / mean_positive
                if mean_positive > 0
                else 0.0
            )

            cv2 = cv ** 2

            adi = (
                calendar_days / positive_days
            )
        else:
            mean_positive = 0.0
            median_positive = 0.0
            std_positive = 0.0
            cv = 0.0
            cv2 = 0.0
            adi = float("inf")

        zero_ratio = (
            zero_days / calendar_days
            if calendar_days > 0
            else 1.0
        )

        activity_rate = (
            positive_days / calendar_days
            if calendar_days > 0
            else 0.0
        )

        # ----------------------------------------------------------
        # Recent demand
        # ----------------------------------------------------------

        indexed = group.set_index("INVDT")["Demand_Qty"]

        recent_30 = float(
            indexed.tail(30).sum()
        )

        recent_90 = float(
            indexed.tail(90).sum()
        )

        recent_180 = float(
            indexed.tail(180).sum()
        )

        recent_365 = float(
            indexed.tail(365).sum()
        )

        # ----------------------------------------------------------
        # Average daily demand
        # ----------------------------------------------------------

        avg_daily_demand = (
            total_demand / calendar_days
            if calendar_days > 0
            else 0.0
        )

        # ----------------------------------------------------------
        # Recent vs historical demand ratio
        # ----------------------------------------------------------

        recent_365_avg = (
            recent_365 / min(365, calendar_days)
            if calendar_days > 0
            else 0.0
        )

        historical_avg = avg_daily_demand

        if historical_avg > 0:
            recent_vs_historical = (
                recent_365_avg / historical_avg
            )
        else:
            recent_vs_historical = 0.0

        # ----------------------------------------------------------
        # Simple trend indicator
        #
        # Compare first half vs second half of available history.
        # ----------------------------------------------------------

        midpoint = len(group) // 2

        if midpoint >= 2:

            first_half = float(
                group.iloc[:midpoint]["Demand_Qty"].mean()
            )

            second_half = float(
                group.iloc[midpoint:]["Demand_Qty"].mean()
            )

            if first_half > 0:
                trend_ratio = (
                    second_half / first_half
                )
            else:
                trend_ratio = 0.0

        else:
            first_half = 0.0
            second_half = 0.0
            trend_ratio = 0.0

        demand_class = classify_demand(
            adi=adi,
            cv2=cv2,
        )

        records.append(
            {
                "MDCODE": mdcode,

                "first_date": first_date,
                "last_date": last_date,

                "calendar_days": calendar_days,

                "total_demand": total_demand,

                "positive_demand_days": positive_days,
                "zero_demand_days": zero_days,

                "demand_activity_rate": activity_rate,
                "zero_demand_ratio": zero_ratio,

                "mean_positive_demand": mean_positive,
                "median_positive_demand": median_positive,
                "std_positive_demand": std_positive,

                "cv": cv,
                "cv2": cv2,
                "adi": adi,

                "avg_daily_demand": avg_daily_demand,

                "recent_30_demand": recent_30,
                "recent_90_demand": recent_90,
                "recent_180_demand": recent_180,
                "recent_365_demand": recent_365,

                "recent_vs_historical_ratio": (
                    recent_vs_historical
                ),

                "first_half_avg": first_half,
                "second_half_avg": second_half,
                "trend_ratio": trend_ratio,

                "demand_class": demand_class,
            }
        )

    profile = pd.DataFrame(records)

    # --------------------------------------------------------------
    # Validation
    # --------------------------------------------------------------

    if profile.empty:
        raise ValueError(
            "Demand profile is empty."
        )

    if profile["MDCODE"].duplicated().any():
        raise ValueError(
            "Duplicate medicine profiles detected."
        )

    numeric_columns = [
        "calendar_days",
        "total_demand",
        "positive_demand_days",
        "zero_demand_days",
        "demand_activity_rate",
        "zero_demand_ratio",
        "mean_positive_demand",
        "median_positive_demand",
        "std_positive_demand",
        "cv",
        "cv2",
        "adi",
        "avg_daily_demand",
        "recent_30_demand",
        "recent_90_demand",
        "recent_180_demand",
        "recent_365_demand",
        "recent_vs_historical_ratio",
        "trend_ratio",
    ]

    for column in numeric_columns:

        if profile[column].isna().any():
            raise ValueError(
                f"NaN values found in {column}."
            )

    profile = profile.sort_values(
        ["demand_class", "demand_activity_rate"],
        ascending=[True, False],
    ).reset_index(drop=True)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    profile.to_parquet(
        output_path,
        index=False,
    )

    # --------------------------------------------------------------
    # Report
    # --------------------------------------------------------------

    print("=" * 70)
    print("DEMAND PROFILE ANALYSIS")
    print("=" * 70)

    print(
        f"Medicines analyzed : {len(profile)}"
    )

    print("\nDemand classification:")

    print(
        profile["demand_class"]
        .value_counts()
        .to_string()
    )

    print("\nDemand activity summary:")

    print(
        profile[
            [
                "demand_activity_rate",
                "zero_demand_ratio",
                "adi",
                "cv2",
                "avg_daily_demand",
            ]
        ]
        .describe()
        .to_string()
    )

    print("\nLowest activity medicines:")

    print(
        profile[
            [
                "MDCODE",
                "demand_class",
                "calendar_days",
                "positive_demand_days",
                "zero_demand_days",
                "demand_activity_rate",
                "adi",
                "cv2",
            ]
        ]
        .sort_values("demand_activity_rate")
        .head(15)
        .to_string(index=False)
    )

    print("\nHighest activity medicines:")

    print(
        profile[
            [
                "MDCODE",
                "demand_class",
                "calendar_days",
                "positive_demand_days",
                "zero_demand_days",
                "demand_activity_rate",
                "adi",
                "cv2",
            ]
        ]
        .sort_values(
            "demand_activity_rate",
            ascending=False,
        )
        .head(15)
        .to_string(index=False)
    )

    print("\nSaved:")
    print(output_path)

    return profile


if __name__ == "__main__":
    analyze_demand_profile()