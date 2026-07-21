from __future__ import annotations

import pandas as pd

from finemed_ai.utils.logger import get_logger

from finemed_ai.warehouse.dimensions.build_dim_date import build_dim_date
from finemed_ai.warehouse.dimensions.build_dim_product import build_dim_product
from finemed_ai.warehouse.dimensions.build_dim_supplier import build_dim_supplier
from finemed_ai.warehouse.dimensions.build_dim_salesperson import build_dim_salesperson
from finemed_ai.warehouse.dimensions.build_dim_tax import build_dim_tax


logger = get_logger(__name__)


def run_dimensions(
    tables: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """ Build all warehouse dimension tables. """

    logger.info("Starting Dimension Layer...")

    logger.info("Available extracted tables:")
    
    for key in tables.keys():
        logger.info(" -> %s", key)

    dimensions: dict[str, pd.DataFrame] = {}

    try:

        dimensions["dim_date"] = build_dim_date(
            start_date="2018-01-01",
            end_date="2026-05-31",
        )

        dimensions["dim_product"] = build_dim_product(
            tables
        )

        dimensions["dim_supplier"] = build_dim_supplier(
            tables
        )

        dimensions["dim_salesperson"] = build_dim_salesperson(
            tables
        )

        dimensions["dim_tax"] = build_dim_tax(
            tables
        )


        for name, df in dimensions.items():

            logger.info(
                "%s created | rows=%s columns=%s",
                name,
                df.shape[0],
                df.shape[1],
            )


        logger.info(
            "Dimension Layer Completed | total=%s",
            len(dimensions),
        )


        return dimensions


    except Exception as error:

        logger.exception(
            "Dimension Layer Failed"
        )

        raise RuntimeError(
            "Failed building dimension layer"
        ) from error