from __future__ import annotations

import pandas as pd


def build_dim_supplier(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    table_name = "SUPMAST.DAT"

    if table_name not in tables:
        raise KeyError(f"{table_name} not found in input tables.")

    source = tables[table_name].copy()

    required_columns = [
        "SUPNO",
        "SUPNAME",
        "SUPCODE",
        "OLDCODE",
        "RNAME",
        "RADD1",
        "RADD2",
        "RADD3",
        "RADD4",
        "RPHON",
        "SOURCE_MONTH",
    ]

    available_columns = [col for col in required_columns if col in source.columns]

    dim_supplier = source[available_columns].copy()

    if "SUPNO" not in dim_supplier.columns:
        raise KeyError("SUPNO is required to build dim_supplier.")

    dim_supplier = (
        dim_supplier
        .sort_values("SOURCE_MONTH")
        .drop_duplicates(subset=["SUPNO"], keep="last")
        .reset_index(drop=True)
    )

    return dim_supplier