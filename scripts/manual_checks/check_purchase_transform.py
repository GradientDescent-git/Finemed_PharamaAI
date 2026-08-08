from __future__ import annotations

from pathlib import Path

from finemed_ai.transform.purchase.purchase_transform import PurchaseTransformer
from finemed_ai.transform.common.helper_functions import get_logger, log_step

logger = get_logger(__name__)


def main() -> None:

    log_step(logger, "=" * 70)
    log_step(logger, "Testing Purchase Transform Module")
    log_step(logger, "=" * 70)

    transformer = PurchaseTransformer()

    log_step(logger, "Testing load_data()")
    transformer.load_data()
    logger.info("✓ load_data() Passed")

    log_step(logger, "Testing join_dimensions()")
    transformer.join_dimensions()
    logger.info("✓ join_dimensions() Passed")

    log_step(logger, "Testing clean_data()")
    transformer.clean_data()
    logger.info("✓ clean_data() Passed")

    log_step(logger, "Testing business_transformations()")
    transformer.business_transformations()
    logger.info("✓ business_transformations() Passed")

    log_step(logger, "Testing save()")
    transformer.save(
        Path("data/04_silver/purchase/purchase_silver.parquet")
    )
    logger.info("✓ save() Passed")

    log_step(logger, "=" * 70)
    log_step(logger, "Purchase Transform Module Test Passed")
    log_step(logger, "=" * 70)


if __name__ == "__main__":
    main()