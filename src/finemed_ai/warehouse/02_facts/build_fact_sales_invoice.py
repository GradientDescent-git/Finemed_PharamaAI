from __future__ import annotations

import pandas as pd


def build_fact_sales_invoice(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
     table_name = "INVOICE.DAT"

    if table_name not in tables:
        raise KeyError(f"{table_name} not found in extracted tables.")

    source = tables[table_name].copy()

    required_columns = [
        "INVNO",
        "INVDT",
        "TIME",
        "DUEDT",
        "SOURCE_MONTH",
        "CCODE",
        "ACODE",
        "SCODE",
        "TOT_AMT",
        "DISC_PER",
        "FDISC_AMT",
        "ST_AMT",
        "NET_AMT",
        "CANCEL_ID",
        "REMARK",
    ]

    missing_columns = [
        col for col in required_columns
        if col not in source.columns
    ]

    if missing_columns:
        raise KeyError(
            f"Missing required columns: {missing_columns}"
        )

    fact_sales_invoice = source[
        required_columns
    ].copy()

    fact_sales_invoice = (
        fact_sales_invoice
        .drop_duplicates(
            subset=["SOURCE_MONTH", "INVNO"]
        )
        .sort_values(
            ["SOURCE_MONTH", "INVNO"]
        )
        .reset_index(drop=True)
    )

    return fact_sales_invoice