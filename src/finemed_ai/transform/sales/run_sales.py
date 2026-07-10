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


def run_sales_pipeline() -> None:

    log_step(
        logger,
        "=" * 70,
    )

    log_step(
        logger,
        "Starting Sales Transformation Pipeline",
    )

    log_step(
        logger,
        "=" * 70,
    )

    transformer = SalesTransformer(

        fact_sales_path=Path(
            "data/warehouse/fact_sales.parquet"
        ),

        dim_customer_path=Path(
            "data/warehouse/dim_customer.parquet"
        ),

        dim_date_path=Path(
            "data/warehouse/dim_date.parquet"
        ),

        dim_medicine_path=Path(
            "data/warehouse/dim_medicine.parquet"
        ),
    )

    transformer.run(

        output_path=Path(
            "data/silver/sales/sales_silver.parquet"
        )
    )

    log_step(
        logger,
        "=" * 70,
    )

    log_step(
        logger,
        "Sales Transformation Pipeline Completed Successfully",
    )

    log_step(
        logger,
        "=" * 70,
    )


if __name__ == "__main__":

    run_sales_pipeline()
    