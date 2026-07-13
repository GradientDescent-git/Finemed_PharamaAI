from __future__ import annotations

import pandas as pd


def build_fact_purchase_header(
    tables: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """ Build fact_purchase_header from COMPUR.DAT. """

    table_name = "COMPUR.DAT"

    if table_name not in tables:
        raise KeyError(f"{table_name} not found in extracted tables.")

    source = tables[table_name].copy()

    required_columns = [
        "PINVNO",
        "PINVDT",
        "TRADT",
        "PACODE",
        "SUPNO",
        "CALVAL",
        "INVAMT",
        "ADDDTL",
        "ADDAMT",
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

    fact_purchase_header = source[
        required_columns
    ].copy()

    fact_purchase_header = (
        fact_purchase_header
        .drop_duplicates(
            subset=[
                "SOURCE_MONTH",
                "PINVNO",
            ]
        )
        .sort_values(
            [
                "SOURCE_MONTH",
                "PINVNO",
            ]
        )
        .reset_index(drop=True)
    )

    return fact_purchase_header