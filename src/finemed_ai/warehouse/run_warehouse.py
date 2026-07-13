from __future__ import annotations

import logging

import pandas as pd

from finemed_ai.utils.logger import get_logger

from finemed_ai.warehouse.dimensions.run_dimensions import (
    run_dimensions,
)

from finemed_ai.warehouse.facts.run_facts import (
    run_facts,
)


logger = get_logger(__name__)


def build_warehouse(
    tables: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """ Build complete warehouse layer. """

    logger.info(
        "Starting Warehouse Build..."
    )


    warehouse: dict[str, pd.DataFrame] = {}


    try:

        dimensions = run_dimensions(
            tables
        )

        warehouse.update(
            dimensions
        )


        facts = run_facts(
            tables
        )

        warehouse.update(
            facts
        )


        logger.info(
            "Warehouse Completed | total_tables=%s",
            len(warehouse),
        )


        return warehouse


    except Exception as error:

        logger.exception(
            "Warehouse Build Failed"
        )

        raise RuntimeError(
            "Warehouse layer failed"
        ) from error



if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | "
            "%(levelname)s | "
            "%(message)s"
        ),
    )


    staging_tables: dict[str, pd.DataFrame] = {}


    warehouse = build_warehouse(
        staging_tables
    )


    for name, df in warehouse.items():

        print(
            f"{name}: "
            f"{df.shape}"
        )