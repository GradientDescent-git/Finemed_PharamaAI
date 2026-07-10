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

    # Execute Pipeline
    transformer = InventoryTransformer(
        inventory_fact_path=inventory_fact_path,
        medicine_dimension_path=medicine_dimension_path,
        date_dimension_path=date_dimension_path,
    )

    transformer.run(output_path)

    log_step(logger, "=" * 70)
    log_step(logger, "Inventory Silver Layer Completed Successfully")
    log_step(logger, "=" * 70)


if __name__ == "__main__":
    main()