from __future__ import annotations

from pathlib import Path

import pandas as pd


DEFAULT_INPUT = Path(
    "data/04_silver/demand_forecasting/daily_demand.parquet"
)

DEFAULT_OUTPUT = Path(
    "data/04_silver/demand_forecasting/monthly_demand.parquet"
)


def build_monthly_demand(
    input_path: str | Path = DEFAULT_INPUT,
    output_path: str | Path = DEFAULT_OUTPUT,
) -> pd.DataFrame:
    """
    Build a continuous monthly demand series from daily demand.

    Rules:
    - One row per MDCODE/month.
    - First month = medicine's first observed demand month.
    - Last month = medicine's last observed demand month.
    - Missing months inside that active window = zero demand.
    - No months before first sale or after last sale.
    """

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

    # ---------------------------------------------------------
    # Aggregate observed demand to calendar month
    # ---------------------------------------------------------

    df["Month"] = (
        df["INVDT"]
        .dt.to_period("M")
        .dt.to_timestamp()
    )

    observed = (
        df.groupby(
            ["MDCODE", "Month"],
            as_index=False,
        )["Demand_Qty"]
        .sum()
    )

    # ---------------------------------------------------------
    # Build continuous monthly series
    # ---------------------------------------------------------

    series_list = []

    for mdcode, group in observed.groupby(
        "MDCODE",
        sort=True,
    ):

        first_month = group["Month"].min()
        last_month = group["Month"].max()

        calendar = pd.DataFrame(
            {
                "Month": pd.date_range(
                    start=first_month,
                    end=last_month,
                    freq="MS",
                )
            }
        )

        calendar = calendar.merge(
            group[
                [
                    "Month",
                    "Demand_Qty",
                ]
            ],
            on="Month",
            how="left",
            validate="one_to_one",
        )

        calendar["Demand_Qty"] = (
            calendar["Demand_Qty"]
            .fillna(0.0)
        )

        calendar["MDCODE"] = mdcode

        series_list.append(
            calendar[
                [
                    "MDCODE",
                    "Month",
                    "Demand_Qty",
                ]
            ]
        )

    if not series_list:
        raise ValueError(
            "No monthly demand series could be constructed."
        )

    result = pd.concat(
        series_list,
        ignore_index=True,
    )

    result = (
        result
        .sort_values(["MDCODE", "Month"])
        .reset_index(drop=True)
    )

    # ---------------------------------------------------------
    # Final validation
    # ---------------------------------------------------------

    if result.duplicated(
        ["MDCODE", "Month"]
    ).any():
        raise ValueError(
            "Duplicate MDCODE/Month rows found."
        )

    if result["Demand_Qty"].isna().any():
        raise ValueError(
            "Monthly demand contains NULL values."
        )

    if (result["Demand_Qty"] < 0).any():
        raise ValueError(
            "Monthly demand contains negative values."
        )

    # ---------------------------------------------------------
    # Conservation check
    # ---------------------------------------------------------
    daily_totals = (
        df.groupby("MDCODE")["Demand_Qty"]
        .sum()
        .sort_index())
    monthly_totals = (
        result.groupby("MDCODE")["Demand_Qty"]
        .sum()
        .sort_index())
    
    if set(daily_totals.index) != set(monthly_totals.index):
        raise ValueError(
            "Medicine sets differ between daily and monthly demand.")

    daily_totals = daily_totals.sort_index()
    monthly_totals = monthly_totals.reindex(daily_totals.index)

    differences = (
        daily_totals.astype(float)
        - monthly_totals.astype(float)).abs()

    bad_differences = differences[
        differences > 1e-8]

    if not bad_differences.empty:
        raise ValueError(
            "Daily/monthly demand conservation failed. "
            f"Differences:\n{bad_differences.head(20)}")

    print(
        "Conservation check passed: "
        "daily demand totals equal monthly demand totals.")

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_parquet(
        output_path,
        index=False,
    )

    # ---------------------------------------------------------
    # Report
    # ---------------------------------------------------------

    print("=" * 70)
    print("MONTHLY DEMAND SERIES BUILT")
    print("=" * 70)

    print(
        f"Medicines       : {result['MDCODE'].nunique()}"
    )

    print(
        f"Rows            : {len(result):,}"
    )

    print(
        f"Date range      : "
        f"{result['Month'].min()} -> "
        f"{result['Month'].max()}"
    )

    print(
        f"Positive months : "
        f"{(result['Demand_Qty'] > 0).sum():,}"
    )

    print(
        f"Zero months     : "
        f"{(result['Demand_Qty'] == 0).sum():,}"
    )

    print(
        f"Duplicate keys  : "
        f"{result.duplicated(['MDCODE', 'Month']).sum()}"
    )

    print()
    print("Monthly rows per medicine:")

    print(
        result.groupby("MDCODE")
        .size()
        .describe()
        .to_string()
    )

    print()
    print(f"Saved: {output_path}")

    return result


if __name__ == "__main__":
    build_monthly_demand()