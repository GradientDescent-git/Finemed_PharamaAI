from __future__ import annotations

import logging
import pandas as pd

from finemed_ai.warehouse.run_dimensions import build_all_dimensions
from finemed_ai.warehouse.run_facts import build_all_facts

logger = logging.getLogger(__name__)


def build_warehouse(
    tables: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:

    logger.info("Building Warehouse Layer...")

    warehouse = {}

    dimensions = build_all_dimensions(tables)
    warehouse.update(dimensions)

    facts = build_all_facts(tables)
    warehouse.update(facts)

    logger.info("Warehouse Layer Built Successfully.")

    return warehouse