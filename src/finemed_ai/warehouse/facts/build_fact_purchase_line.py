from __future__ import annotations

import pandas as pd


def build_fact_purchase_line(
    tables: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """ Build fact_purchase_line from PURCHASE.DAT. """

    table_name = "PURCHASE.DAT"

    if table_name not in tables:
        raise KeyError(f"{table_name} not found in extracted tables.")

    source = tables[table_name].copy()

    required_columns = [
        "PINVNO",
        "MDCODE",
        "BAT",
        "EXP",
        "QTY",
        "FQTY",
        "PRATE",
        "SRATE",
        "MRP",
        "TCODE",
        "SOH",
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

    fact_purchase_line = source[
        required_columns
    ].copy()

    fact_purchase_line = (
        fact_purchase_line
        .drop_duplicates(
            subset=[
                "SOURCE_MONTH",
                "PINVNO",
                "MDCODE",
                "BAT",
            ]
        )
        .sort_values(
            [
                "SOURCE_MONTH",
                "PINVNO",
                "MDCODE",
            ]
        )
        .reset_index(drop=True)
    )

    return fact_purchase_line