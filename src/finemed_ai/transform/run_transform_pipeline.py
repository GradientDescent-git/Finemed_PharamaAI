from __future__ import annotations

from finemed_ai.transform.common.helper_functions import (
    get_logger,
    log_step,
)

from finemed_ai.transform.inventory.run_inventory import (
    main as run_inventory,
)

from finemed_ai.transform.medicine.run_medicine import (
    main as run_medicine,
)

from finemed_ai.transform.purchase.run_purchase import (
    main as run_purchase,
)

from finemed_ai.transform.sales.run_sales import (
    main as run_sales,
)

from finemed_ai.transform.supplier.run_supplier import (
    main as run_supplier,
)

logger = get_logger(__name__)


def main() -> None:

    try:

        log_step(logger, "=" * 80)
        log_step(logger, "Starting Silver Layer Transformation Pipeline")
        log_step(logger, "=" * 80)

        # Master Dimensions
        run_medicine()
        run_supplier()

        # Inventory
        run_inventory()

        # Transactional Data
        run_purchase()
        run_sales()

        log_step(logger, "=" * 80)
        log_step(
            logger,
            "Silver Layer Transformation Completed Successfully",
        )
        log_step(logger, "=" * 80)

    except Exception:

        logger.exception(
            "Silver Layer Transformation Pipeline Failed."
        )
        raise


if __name__ == "__main__":
    main()