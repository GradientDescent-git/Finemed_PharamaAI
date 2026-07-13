from __future__ import annotations

import pandas as pd

from finemed_ai.utils.logger import get_logger

logger = get_logger(__name__)


def build_fact_sales_line(
    tables: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """ Build the Sales Line Fact table from INVDET.DAT. """

    logger.info("Building Sales Line Fact")

    try:

        table_name = "INVDET.DAT"

        if table_name not in tables:
            raise KeyError(
                f"{table_name} not found in extracted tables."
            )

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
            column
            for column in required_columns
            if column not in source.columns
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

        logger.info(
            "Sales Line Fact built successfully | rows=%s | columns=%s",
            len(fact_sales_line),
            len(fact_sales_line.columns),
        )

        return fact_sales_line

    except Exception as error:

        logger.exception(
            "Failed while building Sales Line Fact."
        )

        raise RuntimeError(
            "Sales Line Fact build failed."
        ) from error