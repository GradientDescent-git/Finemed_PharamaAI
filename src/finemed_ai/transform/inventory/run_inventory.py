from __future__ import annotations

from pathlib import Path

from finemed_ai.transform.inventory.inventory_transform import (
    InventoryTransformer,
)

from finemed_ai.transform.common.helper_functions import (
    get_logger,
    log_step,
)

logger = get_logger(__name__)


def main() -> None:

    log_step(logger, "=" * 70)
    log_step(logger, "Starting Inventory Silver Layer Pipeline")
    log_step(logger, "=" * 70)

    
    # Warehouse Inputs

    inventory_fact_path = Path(
        "data/03_warehouse/facts/fact_inventory.parquet"
    )

    medicine_dimension_path = Path(
        "data/03_warehouse/dimensions/dim_product.parquet"
    )

    date_dimension_path = Path(
        "data/03_warehouse/dimensions/dim_date.parquet"
    )

    
    # Silver Output
    
    output_path = Path(
        "data/04_silver/inventory/inventory_silver.parquet"
    )

    
    # Validate Input Files
    

    for path in [
        inventory_fact_path,
        medicine_dimension_path,
        date_dimension_path,
    ]:

        if not path.exists():
            raise FileNotFoundError(
                f"Missing input file: {path}"
            )

    
    # Log Input / Output Paths

    logger.info(
        "Inventory Fact      : %s",
        inventory_fact_path,
    )

    logger.info(
        "Medicine Dimension  : %s",
        medicine_dimension_path,
    )

    logger.info(
        "Date Dimension      : %s",
        date_dimension_path,
    )

    logger.info(
        "Silver Output       : %s",
        output_path,
    )

    
    # Execute Inventory Silver Pipeline

    transformer = InventoryTransformer(
        inventory_fact_path=inventory_fact_path,
        medicine_dimension_path=medicine_dimension_path,
        date_dimension_path=date_dimension_path,
    )

    try:

        transformer.run(output_path)

        log_step(logger, "=" * 70)
        log_step(
            logger,
            "Inventory Silver Layer Completed Successfully",
        )
        log_step(logger, "=" * 70)

    except Exception:

        logger.exception(
            "Inventory Silver Layer Pipeline Failed."
        )

        raise


if __name__ == "__main__":
    main()