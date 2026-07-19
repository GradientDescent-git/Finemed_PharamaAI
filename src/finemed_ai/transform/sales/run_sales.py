from __future__ import annotations

from pathlib import Path

from finemed_ai.transform.common.helper_functions import (
    get_logger,
    log_step,
)

from finemed_ai.transform.sales.sales_transform import (
    SalesTransformer,
)

logger = get_logger(__name__)


def main() -> None:

    try:

        log_step(logger, "=" * 70)
        log_step(logger, "Starting Sales Silver Layer Pipeline")
        log_step(logger, "=" * 70)

        # Warehouse Inputs

        fact_sales_path = Path(
            "data/03_warehouse/facts/fact_sales.parquet"
        )

        dim_customer_path = Path(
            "data/03_warehouse/dimensions/dim_customer.parquet"
        )

        dim_date_path = Path(
            "data/03_warehouse/dimensions/dim_date.parquet"
        )

        dim_medicine_path = Path(
            "data/03_warehouse/dimensions/dim_product.parquet"
        )

        # Silver Output

        output_path = Path(
            "data/04_silver/sales/sales_silver.parquet"
        )

        # Execute Pipeline

        transformer = SalesTransformer(
            fact_sales_path=fact_sales_path,
            dim_customer_path=dim_customer_path,
            dim_date_path=dim_date_path,
            dim_medicine_path=dim_medicine_path,
        )

        transformer.run(
            output_path=output_path,
        )

        log_step(logger, "=" * 70)
        log_step(logger, "Sales Silver Layer Completed Successfully")
        log_step(logger, "=" * 70)

    except Exception:

        logger.exception(
            "Sales Silver Layer Pipeline Failed."
        )

        raise


if __name__ == "__main__":
    main()