from __future__ import annotations

import pandas as pd


def build_fact_sales_line(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    table_name = "INVDET.DAT"

    if table_name not in tables:
        raise KeyError(f"{table_name} not found in extracted tables.")

    source = tables[table_name].copy()

    required_columns = [
        "INVNO",
        "MDCODE",
        "BATCH",
        "EXP",
        "QTY",
        "FQTY",
        "RATE",
        "SERATE",
        "ACRATE",
        "PRATE",
        "MRP",
        "TCODE",
        "CANCEL_ID",
        "SUPNO",
        "LPDT",
        "RATCHG",
        "SOURCE_MONTH",
    ]

    missing_columns = [
        col for col in required_columns
        if col not in source.columns
    ]

    if missing_columns:
        raise KeyError(
            f"Missing required columns: {missing_columns}"
        )

    fact_sales_line = source[
        required_columns
    ].copy()

    fact_sales_line = (
        fact_sales_line
        .drop_duplicates(
            subset=[
                "SOURCE_MONTH",
                "INVNO",
                "MDCODE",
                "BATCH",
            ]
        )
        .sort_values(
            [
                "SOURCE_MONTH",
                "INVNO",
                "MDCODE",
            ]
        )
        .reset_index(drop=True)
    )

    return fact_sales_line