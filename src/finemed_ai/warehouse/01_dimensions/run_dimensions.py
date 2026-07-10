from __future__ import annotations

import pandas as pd

from finemed_ai.utils.logger import get_logger

from finemed_ai.warehouse.dimensions.build_dim_date import build_dim_date
from finemed_ai.warehouse.dimensions.build_dim_product import build_dim_product
from finemed_ai.warehouse.dimensions.build_dim_supplier import build_dim_supplier
from finemed_ai.warehouse.dimensions.build_dim_salesperson import (
    build_dim_salesperson,
)
from finemed_ai.warehouse.dimensions.build_dim_tax import build_dim_tax

logger = get_logger(__name__)


def run_dimensions(tables: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    logger.info("Building warehouse dimensions...")

    dimensions = {
        "dim_date": build_dim_date(
            start_date="2017-12-01",
            end_date="2026-05-31",
        ),
        "dim_product": build_dim_product(tables),
        "dim_supplier": build_dim_supplier(tables),
        "dim_salesperson": build_dim_salesperson(tables),
        "dim_tax": build_dim_tax(tables),
    }

    logger.info(
        "Successfully built %d dimension tables.",
        len(dimensions),
    )

    return dimensions