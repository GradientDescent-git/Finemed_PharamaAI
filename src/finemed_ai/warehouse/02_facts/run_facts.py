from __future__ import annotations

import logging
import pandas as pd

from finemed_ai.warehouse.facts.build_fact_sales_invoice import (
    build_fact_sales_invoice,
)

from finemed_ai.warehouse.facts.build_fact_sales_line import (
    build_fact_sales_line,
)

from finemed_ai.warehouse.facts.build_fact_purchase_header import (
    build_fact_purchase_header,
)

from finemed_ai.warehouse.facts.build_fact_purchase_line import (
    build_fact_purchase_line,
)

from finemed_ai.warehouse.facts.build_fact_invoice_due import (
    build_fact_invoice_due,
)

logger = logging.getLogger(__name__)


def build_all_facts(tables: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    logger.info("Building Fact Tables...")

    facts = {}

    facts["fact_sales_invoice"] = build_fact_sales_invoice(tables)
    logger.info("✓ fact_sales_invoice created")

    facts["fact_sales_line"] = build_fact_sales_line(tables)
    logger.info("✓ fact_sales_line created")

    facts["fact_purchase_header"] = build_fact_purchase_header(tables)
    logger.info("✓ fact_purchase_header created")

    facts["fact_purchase_line"] = build_fact_purchase_line(tables)
    logger.info("✓ fact_purchase_line created")

    facts["fact_invoice_due"] = build_fact_invoice_due(tables)
    logger.info("✓ fact_invoice_due created")

    logger.info("All Fact Tables Created Successfully.")

    return facts