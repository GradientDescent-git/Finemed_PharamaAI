from __future__ import annotations

import pandas as pd

from finemed_ai.utils.logger import get_logger

logger = get_logger(__name__)


def build_dim_supplier(
    tables: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """ Build the Supplier Dimension from SUPMAST. """

    logger.info("Building Supplier Dimension")

    try:

        table_name = "SUPMAST.DAT"

        if table_name not in tables:
            raise KeyError(
                f"{table_name} not found in extracted tables."
            )

        source = tables[table_name].copy()

        required_columns = [
            "SUPNO",
            "SUPNAME",
            "SUPCODE",
            "OLDCODE",
            "RNAME",
            "RADD1",
            "RADD2",
            "RADD3",
            "RADD4",
            "RPHON",
            "SOURCE_MONTH",
        ]

        available_columns = [
            column
            for column in required_columns
            if column in source.columns
        ]

        dim_supplier = source[
            available_columns
        ].copy()

        if "SUPNO" not in dim_supplier.columns:
            raise KeyError(
                "SUPNO is required to build Supplier Dimension."
            )

        dim_supplier = (
            dim_supplier
            .sort_values("SOURCE_MONTH")
            .drop_duplicates(
                subset=["SUPNO"],
                keep="last",
            )
            .reset_index(drop=True)
        )

        logger.info(
            "Supplier Dimension built successfully | rows=%s | columns=%s",
            len(dim_supplier),
            len(dim_supplier.columns),
        )

        return dim_supplier

    except Exception as error:

        logger.exception(
            "Failed while building Supplier Dimension."
        )

        raise RuntimeError(
            "Supplier Dimension build failed."
        ) from error