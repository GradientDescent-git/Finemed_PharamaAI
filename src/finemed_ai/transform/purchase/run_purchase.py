from __future__ import annotations

from pathlib import Path

from finemed_ai.transform.purchase.purchase_transform import (
    PurchaseTransformer,
)

from finemed_ai.transform.common.helper_functions import (
    get_logger,
    log_step,
)

logger = get_logger(__name__)


def main() -> None:

    log_step(logger, "=" * 70)
    log_step(logger, "Starting Purchase Silver Layer Pipeline")
    log_step(logger, "=" * 70)

   
    # Silver Output

    output_path = Path(
        "data/04_silver/purchase/purchase_silver.parquet"
    )

    try:

        transformer = PurchaseTransformer()

        transformer.run(output_path)

        log_step(logger, "=" * 70)
        log_step(
            logger,
            "Purchase Silver Layer Completed Successfully",
        )
        log_step(logger, "=" * 70)

    except Exception:

        logger.exception(
            "Purchase Silver Layer Pipeline Failed."
        )

        raise


if __name__ == "__main__":
    main()