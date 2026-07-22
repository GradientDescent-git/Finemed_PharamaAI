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
    """
    Execute the Inventory Silver Layer pipeline.

    Flow
    ----
    Warehouse (PostgreSQL)
            ↓
    InventoryTransformer
            ↓
    Silver Parquet
    """

    log_step(logger, "=" * 70)
    log_step(logger, "Starting Inventory Silver Layer Pipeline")
    log_step(logger, "=" * 70)

    # Silver Output

    output_path = Path(
        "data/04_silver/inventory/inventory_silver.parquet"
    )

    logger.info(
        "Silver Output : %s",
        output_path,
    )

    # Execute Inventory Silver Pipeline

    transformer = InventoryTransformer()

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