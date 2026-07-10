from __future__ import annotations

import pandas as pd


def build_fact_invoice_due(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    table_name = "INVOICE.DAT"

    if table_name not in tables:
        raise KeyError(f"{table_name} not found in extracted tables.")

    source = tables[table_name].copy()

    required_columns = [
        "INVNO",
        "CCODE",
        "SCODE",
        "INVDT",
        "DUEDT",
        "NET_AMT",
        "CANCEL_ID",
        "SOURCE_MONTH",
    ]

    missing_columns = [
        col
        for col in required_columns
        if col not in source.columns
    ]

    if missing_columns:
        raise KeyError(
            f"Missing required columns: {missing_columns}"
        )

    fact_invoice_due = source[
        required_columns
    ].copy()

    fact_invoice_due = (
        fact_invoice_due
        .drop_duplicates(
            subset=[
                "SOURCE_MONTH",
                "INVNO",
            ]
        )
        .sort_values(
            [
                "SOURCE_MONTH",
                "DUEDT",
                "INVNO",
            ]
        )
        .reset_index(drop=True)
    )

    return fact_invoice_due