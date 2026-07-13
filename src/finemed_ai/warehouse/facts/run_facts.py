from __future__ import annotations

import pandas as pd

from finemed_ai.utils.logger import get_logger

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


logger = get_logger(__name__)


def run_facts(
    tables: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """
    Build all warehouse fact tables.
    """

    logger.info(
        "Starting Fact Layer..."
    )

    try:

        facts: dict[str, pd.DataFrame] = {}


        facts["fact_sales_invoice"] = (
            build_fact_sales_invoice(tables)
        )


        facts["fact_sales_line"] = (
            build_fact_sales_line(tables)
        )


        facts["fact_purchase_header"] = (
            build_fact_purchase_header(tables)
        )


        facts["fact_purchase_line"] = (
            build_fact_purchase_line(tables)
        )


        facts["fact_invoice_due"] = (
            build_fact_invoice_due(tables)
        )


        for name, dataframe in facts.items():

            logger.info(
                "%s built | rows=%s columns=%s",
                name,
                dataframe.shape[0],
                dataframe.shape[1],
            )


        return facts


    except Exception as error:

        logger.exception(
            "Fact Layer failed."
        )

        raise RuntimeError(
            "Fact Layer build failed."
        ) from error