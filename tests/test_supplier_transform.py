from __future__ import annotations

from pathlib import Path

from finemed_ai.transform.supplier.supplier_transform import (
    SupplierTransformer,
)

from finemed_ai.transform.common.helper_functions import (
    get_logger,
    log_step,
)

logger = get_logger(__name__)


def main() -> None:

    log_step(logger, "=" * 70)
    log_step(logger, "Testing Supplier Transform Module")
    log_step(logger, "=" * 70)

    transformer = SupplierTransformer(

        supplier_dimension_path=Path(
            "data/03_warehouse/dimensions/dim_supplier.parquet"
        ),
    )

    
    # Load
    

    log_step(logger, "Testing load_data()")

    transformer.load_data()

    logger.info("✓ load_data() Passed")

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
        "data/04_silver/supplier/supplier_silver.parquet"
    )

    transformer.save(output_path)

    logger.info("✓ save() Passed")

    log_step(logger, "=" * 70)
    log_step(logger, "Supplier Transform Module Test Passed")
    log_step(logger, "=" * 70)


if __name__ == "__main__":

    main()