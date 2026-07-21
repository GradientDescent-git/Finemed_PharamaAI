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
    """
    Build complete warehouse layer.

    Parameters
    ----------
    tables : dict[str, pd.DataFrame]
        Dictionary of extracted ERP tables.

    Returns
    -------
    dict[str, pd.DataFrame]
        Dictionary containing all warehouse tables.
    """

    logger.info("=" * 80)
    logger.info("Starting Warehouse Build")
    logger.info("=" * 80)

    logger.info("Warehouse received %d extracted tables", len(tables))

    for table_name in sorted(tables.keys()):
        logger.info(" -> %s", table_name)

    warehouse: dict[str, pd.DataFrame] = {}

    try:

        logger.info("Building Dimension Layer...")

        dimensions = run_dimensions(tables)

        warehouse.update(dimensions)

        logger.info("Building Fact Layer...")

        facts = run_facts(tables)

        warehouse.update(facts)

        logger.info(
            "Warehouse Build Completed | total_tables=%d",
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
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    dummy_tables: dict[str, pd.DataFrame] = {}

    warehouse = build_warehouse(dummy_tables)

    for name, df in warehouse.items():

        print(f"{name}: {df.shape}")