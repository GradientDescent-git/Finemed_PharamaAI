from __future__ import annotations

from pathlib import Path

from finemed_ai.transform.sales.sales_transform import (
    SalesTransformer,
)

from finemed_ai.transform.common.helper_functions import (
    get_logger,
    log_step,
)

logger = get_logger(__name__)


def main() -> None:

    log_step(logger, "=" * 70)
    log_step(logger, "Testing Sales Transform Module")
    log_step(logger, "=" * 70)

    transformer = SalesTransformer(

        fact_sales_path=Path(
            "data/03_warehouse/facts/fact_sales.parquet"
        ),

        dim_customer_path=Path(
            "data/03_warehouse/dimensions/dim_customer.parquet"
        ),

        dim_date_path=Path(
            "data/03_warehouse/dimensions/dim_date.parquet"
        ),

        dim_medicine_path=Path(
            "data/03_warehouse/dimensions/dim_product.parquet"
        ),
    )

    # Load

    log_step(logger, "Testing load_data()")

    transformer.load_data()

    logger.info("✓ load_data() Passed")

    # Join Dimensions

    log_step(logger, "Testing join_dimensions()")

    transformer.join_dimensions()

    logger.info("✓ join_dimensions() Passed")


    # Clean Data

    log_step(logger, "Testing clean_data()")

    transformer.clean_data()

    logger.info("✓ clean_data() Passed")

    # Business Transformations

    log_step(logger, "Testing business_transformations()")

    transformer.business_transformations()

    logger.info("✓ business_transformations() Passed")


    # Save
    

    log_step(logger, "Testing save()")

    output_path = Path(
        "data/04_silver/sales/sales_silver.parquet"
    )

    transformer.save(output_path)

    logger.info("✓ save() Passed")

    log_step(logger, "=" * 70)
    log_step(logger, "Sales Transform Module Test Passed")
    log_step(logger, "=" * 70)


if __name__ == "__main__":

    main()