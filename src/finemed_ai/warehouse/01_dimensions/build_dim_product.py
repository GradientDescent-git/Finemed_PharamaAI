from __future__ import annotations

import pandas as pd


def build_dim_product(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    table_name = "MEDIMAST.DAT"

    if table_name not in tables:
        raise KeyError(f"{table_name} not found in input tables.")

    source = tables[table_name].copy()

    required_columns = [
        "MDCODE",
        "MDNAME",
        "PACKG",
        "DETAIL",
        "SUPNO",
        "SUPCODE",
        "TCODE",
        "HSN",
        "UQC",
        "NEWDT",
        "SMDT",
        "SOURCE_MONTH",
    ]

    available_columns = [col for col in required_columns if col in source.columns]

    dim_product = source[available_columns].copy()

    if "MDCODE" not in dim_product.columns:
        raise KeyError("MDCODE is required to build dim_product.")

    dim_product = (
        dim_product
        .sort_values("SOURCE_MONTH")
        .drop_duplicates(subset=["MDCODE"], keep="last")
        .reset_index(drop=True)
    )

    return dim_product