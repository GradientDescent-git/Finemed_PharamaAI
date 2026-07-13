from __future__ import annotations

import pandas as pd

from finemed_ai.utils.logger import get_logger

logger = get_logger(__name__)


def build_fact_invoice_due(
    tables: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """ Build the Invoice Due Fact table from INVOICE.DAT. """

    logger.info("Building Invoice Due Fact")

    try:

        table_name = "INVOICE.DAT"

        if table_name not in tables:
            raise KeyError(
                f"{table_name} not found in extracted tables."
            )

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
            column
            for column in required_columns
            if column not in source.columns
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

        logger.info(
            "Invoice Due Fact built successfully | rows=%s | columns=%s",
            len(fact_invoice_due),
            len(fact_invoice_due.columns),
        )

        return fact_invoice_due

    except Exception as error:

        logger.exception(
            "Failed while building Invoice Due Fact."
        )

        raise RuntimeError(
            "Invoice Due Fact build failed."
        ) from error