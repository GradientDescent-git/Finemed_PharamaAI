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

        # Silver Output

        output_path = Path(
            "data/04_silver/sales/sales_silver.parquet"
        )

        # Execute Pipeline

        transformer = SalesTransformer()

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