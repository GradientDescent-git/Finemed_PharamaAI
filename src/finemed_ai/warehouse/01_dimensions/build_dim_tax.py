from __future__ import annotations

import pandas as pd


def build_dim_tax(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
     table_name = "TFILE.DAT"

    if table_name not in tables:
        raise KeyError(f"{table_name} not found in extracted tables.")

    source = tables[table_name].copy()

    required_columns = [
        "TCODE",
        "STATUS",
        "CST",
        "TS",
        "ST",
        "SC",
        "RST",
        "TAXLESS",
        "CSTFLAG",
        "VATFLAG",
        "COMCODE",
        "IGST",
        "CGST",
        "SGST",
        "SOURCE_MONTH",
    ]

    available_columns = [
        col
        for col in required_columns
        if col in source.columns
    ]

    dim_tax = source[available_columns].copy()

    if "TCODE" not in dim_tax.columns:
        raise KeyError("TCODE is required to build dim_tax.")

    dim_tax = (
        dim_tax
        .sort_values("SOURCE_MONTH")
        .drop_duplicates(subset=["TCODE"], keep="last")
        .reset_index(drop=True)
    )

    return dim_tax