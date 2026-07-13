from __future__ import annotations

import pandas as pd

from finemed_ai.utils.logger import get_logger

logger = get_logger(__name__)


def build_dim_tax(
    tables: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """ Build the Tax Dimension from TFILE. """

    logger.info("Building Tax Dimension")

    try:

        table_name = "TFILE.DAT"

        if table_name not in tables:
            raise KeyError(
                f"{table_name} not found in extracted tables."
            )

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
            column
            for column in required_columns
            if column in source.columns
        ]

        dim_tax = source[
            available_columns
        ].copy()

        if "TCODE" not in dim_tax.columns:
            raise KeyError(
                "TCODE is required to build Tax Dimension."
            )

        dim_tax = (
            dim_tax
            .sort_values("SOURCE_MONTH")
            .drop_duplicates(
                subset=["TCODE"],
                keep="last",
            )
            .reset_index(drop=True)
        )

        logger.info(
            "Tax Dimension built successfully | rows=%s | columns=%s",
            len(dim_tax),
            len(dim_tax.columns),
        )

        return dim_tax

    except Exception as error:

        logger.exception(
            "Failed while building Tax Dimension."
        )

        raise RuntimeError(
            "Tax Dimension build failed."
        ) from error