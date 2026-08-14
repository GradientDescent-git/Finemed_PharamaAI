from __future__ import annotations

from pathlib import Path

import pandas as pd


DEFAULT_DEMAND_INPUT = Path(
    "data/04_silver/demand_forecasting/daily_demand.parquet"
)

DEFAULT_QUALITY_INPUT = Path(
    "data/04_silver/demand_forecasting/series_quality.parquet"
)

DEFAULT_OUTPUT = Path(
    "data/04_silver/demand_forecasting/chronos_series.parquet"
)


def build_forecasting_series(
    demand_path: str | Path = DEFAULT_DEMAND_INPUT,
    quality_path: str | Path = DEFAULT_QUALITY_INPUT,
    output_path: str | Path = DEFAULT_OUTPUT,
) -> pd.DataFrame:
    """
    Build continuous daily demand series for forecasting.

    Rules:
    - Only forecast_candidate medicines are included.
    - Each medicine keeps its observed historical window:
        first_sale_date -> last_sale_date
    - Missing dates INSIDE that window are treated as zero demand.
    - Dates before first sale or after last sale are not created.
    - One row per MDCODE/date is guaranteed.
    """

    demand_path = Path(demand_path)
    quality_path = Path(quality_path)
    output_path = Path(output_path)

    # ---------------------------------------------------------------
    # Load inputs
    # ---------------------------------------------------------------

    demand = pd.read_parquet(demand_path)
    quality = pd.read_parquet(quality_path)

    required_demand = {
        "MDCODE",
        "INVDT",
        "Demand_Qty",
    }

    required_quality = {
        "MDCODE",
        "first_sale_date",
        "last_sale_date",
        "forecast_candidate",
    }

    missing_demand = required_demand - set(demand.columns)
    missing_quality = required_quality - set(quality.columns)

    if missing_demand:
        raise ValueError(
            f"Demand dataset missing columns: {sorted(missing_demand)}"
        )

    if missing_quality:
        raise ValueError(
            f"Quality dataset missing columns: {sorted(missing_quality)}"
        )

    # ---------------------------------------------------------------
    # Normalize
    # ---------------------------------------------------------------

    demand = demand.copy()
    quality = quality.copy()

    demand["MDCODE"] = (
        demand["MDCODE"]
        .astype(str)
        .str.strip()
    )

    quality["MDCODE"] = (
        quality["MDCODE"]
        .astype(str)
        .str.strip()
    )

    demand["INVDT"] = pd.to_datetime(
        demand["INVDT"],
        errors="coerce",
    )

    quality["first_sale_date"] = pd.to_datetime(
        quality["first_sale_date"],
        errors="coerce",
    )

    quality["last_sale_date"] = pd.to_datetime(
        quality["last_sale_date"],
        errors="coerce",
    )

    demand["Demand_Qty"] = pd.to_numeric(
        demand["Demand_Qty"],
        errors="coerce",
    )

    if demand["INVDT"].isna().any():
        raise ValueError(
            "Demand dataset contains invalid INVDT values."
        )

    if demand["Demand_Qty"].isna().any():
        raise ValueError(
            "Demand dataset contains invalid Demand_Qty values."
        )

    # ---------------------------------------------------------------
    # Validate source uniqueness
    # ---------------------------------------------------------------

    duplicate_count = demand.duplicated(
        ["MDCODE", "INVDT"]
    ).sum()

    if duplicate_count:
        raise ValueError(
            f"Found {duplicate_count} duplicate "
            "MDCODE/INVDT rows in demand dataset."
        )

    duplicate_quality = quality.duplicated(
        ["MDCODE"]
    ).sum()

    if duplicate_quality:
        raise ValueError(
            f"Found {duplicate_quality} duplicate MDCODE "
            "rows in quality dataset."
        )

    # ---------------------------------------------------------------
    # Select forecasting candidates
    # ---------------------------------------------------------------

    candidates = quality[
        quality["forecast_candidate"] == True
    ].copy()

    if candidates.empty:
        raise ValueError(
            "No forecast candidates found."
        )

    candidate_codes = set(
        candidates["MDCODE"]
    )

    demand = demand[
        demand["MDCODE"].isin(candidate_codes)
    ].copy()

    # ---------------------------------------------------------------
    # Build continuous series
    # ---------------------------------------------------------------

    series_list = []

    for _, row in candidates.iterrows():

        mdcode = row["MDCODE"]
        first_date = row["first_sale_date"]
        last_date = row["last_sale_date"]

        if pd.isna(first_date) or pd.isna(last_date):
            continue

        if first_date > last_date:
            raise ValueError(
                f"Invalid date range for MDCODE {mdcode}: "
                f"{first_date} > {last_date}"
            )

        # Existing observations for this medicine
        medicine = demand[
            demand["MDCODE"] == mdcode
        ][
            ["INVDT", "Demand_Qty"]
        ].copy()

        # Calendar from first observed sale to last observed sale
        calendar = pd.DataFrame(
            {
                "INVDT": pd.date_range(
                    start=first_date,
                    end=last_date,
                    freq="D",
                )
            }
        )

        # Left join observed demand onto the complete calendar
        calendar = calendar.merge(
            medicine,
            on="INVDT",
            how="left",
            validate="one_to_one",
        )

        # Missing dates inside active window = zero demand
        calendar["Demand_Qty"] = (
            calendar["Demand_Qty"]
            .fillna(0.0)
        )

        calendar["MDCODE"] = mdcode

        series_list.append(
            calendar[
                [
                    "MDCODE",
                    "INVDT",
                    "Demand_Qty",
                ]
            ]
        )

    if not series_list:
        raise ValueError(
            "No forecasting series could be constructed."
        )

    result = pd.concat(
        series_list,
        ignore_index=True,
    )

    # ---------------------------------------------------------------
    # Final validation
    # ---------------------------------------------------------------

    result = result.sort_values(
        ["MDCODE", "INVDT"]
    ).reset_index(drop=True)

    duplicate_count = result.duplicated(
        ["MDCODE", "INVDT"]
    ).sum()

    if duplicate_count:
        raise ValueError(
            f"Found {duplicate_count} duplicate "
            "MDCODE/INVDT rows after series construction."
        )

    if result["Demand_Qty"].isna().any():
        raise ValueError(
            "Constructed series contains NULL Demand_Qty."
        )

    if (result["Demand_Qty"] < 0).any():
        raise ValueError(
            "Constructed series contains negative demand."
        )

    # ---------------------------------------------------------------
    # Save
    # ---------------------------------------------------------------

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_parquet(
        output_path,
        index=False,
    )

    # ---------------------------------------------------------------
    # Report
    # ---------------------------------------------------------------

    print("=" * 70)
    print("FORECASTING SERIES BUILT")
    print("=" * 70)

    print(
        f"Forecast candidates : "
        f"{result['MDCODE'].nunique()}"
    )

    print(
        f"Rows                 : "
        f"{len(result):,}"
    )

    print(
        f"Date range           : "
        f"{result['INVDT'].min()} -> "
        f"{result['INVDT'].max()}"
    )

    print(
        f"Total demand         : "
        f"{result['Demand_Qty'].sum():,.2f}"
    )

    print(
        f"Zero-demand rows     : "
        f"{int((result['Demand_Qty'] == 0).sum()):,}"
    )

    print()
    print("Rows per medicine:")
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
    build_forecasting_series()