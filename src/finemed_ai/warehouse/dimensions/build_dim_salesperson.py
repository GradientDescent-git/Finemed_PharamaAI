from __future__ import annotations

import pandas as pd

from finemed_ai.utils.logger import get_logger

logger = get_logger(__name__)


def build_dim_salesperson(
    tables: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """ Build the Salesperson Dimension from SFILE. """

    logger.info("Building Salesperson Dimension")

    try:

        table_name = "SFILE.DAT"
        
        if table_name not in tables:
            raise KeyError(
                f"{table_name} not found in extracted tables."
            )

        source = tables[table_name].copy()

        required_columns = [
            "SCODE",
            "SNAME",
            "SOURCE_MONTH",
        ]

        available_columns = [
            column
            for column in required_columns
            if column in source.columns
        ]

        dim_salesperson = source[
            available_columns
        ].copy()

        if "SCODE" not in dim_salesperson.columns:
            raise KeyError(
                "SCODE is required to build Salesperson Dimension."
            )

        dim_salesperson = (
            dim_salesperson
            .sort_values("SOURCE_MONTH")
            .drop_duplicates(
                subset=["SCODE"],
                keep="last",
            )
            .reset_index(drop=True)
        )

        logger.info(
            "Salesperson Dimension built successfully | rows=%s | columns=%s",
            len(dim_salesperson),
            len(dim_salesperson.columns),
        )

        return dim_salesperson

    except Exception as error:

        logger.exception(
            "Failed while building Salesperson Dimension."
        )

        raise RuntimeError(
            "Salesperson Dimension build failed."
        ) from error