from __future__ import annotations

import pandas as pd

from finemed_ai.utils.logger import get_logger

logger = get_logger(__name__)


def build_dim_product(
    tables: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """ Build the Product Dimension from MEDIMAST. """

    logger.info("Building Product Dimension")

    try:

        table_name = "MEDIMAST.DAT"

        if table_name not in tables:
            raise KeyError(
                f"{table_name} not found in extracted tables."
            )

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

        available_columns = [
            column
            for column in required_columns
            if column in source.columns
        ]

        dim_product = source[
            available_columns
        ].copy()

        if "MDCODE" not in dim_product.columns:
            raise KeyError(
                "MDCODE is required to build Product Dimension."
            )

        dim_product = (
            dim_product
            .sort_values("SOURCE_MONTH")
            .drop_duplicates(
                subset=["MDCODE"],
                keep="last",
            )
            .reset_index(drop=True)
        )

        logger.info(
            "Product Dimension built successfully | rows=%s | columns=%s",
            len(dim_product),
            len(dim_product.columns),
        )

        return dim_product

    except Exception as error:

        logger.exception(
            "Failed while building Product Dimension."
        )

        raise RuntimeError(
            "Product Dimension build failed."
        ) from error