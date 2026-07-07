from __future__ import annotations

import pandas as pd


def build_dim_salesperson(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    table_name = "SFILE.DAT"

    if table_name not in tables:
        raise KeyError(f"{table_name} not found in extracted tables.")

    source = tables[table_name].copy()

    required_columns = [
        "SCODE",
        "SNAME",
        "SOURCE_MONTH",
    ]

    available_columns = [
        col
        for col in required_columns
        if col in source.columns
    ]

    dim_salesperson = source[available_columns].copy()

    if "SCODE" not in dim_salesperson.columns:
        raise KeyError("SCODE is required to build dim_salesperson.")

    dim_salesperson = (
        dim_salesperson
        .sort_values("SOURCE_MONTH")
        .drop_duplicates(subset=["SCODE"], keep="last")
        .reset_index(drop=True)
    )

    return dim_salesperson