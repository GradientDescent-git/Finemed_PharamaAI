from __future__ import annotations

from pathlib import Path

from finemed_ai.transform.medicine.medicine_transform import (
    MedicineTransformer,
)

from finemed_ai.transform.common.helper_functions import (
    get_logger,
    log_step,
)

logger = get_logger(__name__)


def main() -> None:

    log_step(logger, "=" * 70)
    log_step(logger, "Starting Medicine Silver Layer Pipeline")
    log_step(logger, "=" * 70)

    # --------------------------------------------------
    # Silver Output
    # --------------------------------------------------

    output_path = Path(
        "data/04_silver/medicine/medicine_silver.parquet"
    )

    logger.info(
        "Silver Output : %s",
        output_path,
    )

    transformer = MedicineTransformer()

    try:

        transformer.run(output_path)

        log_step(logger, "=" * 70)
        log_step(
            logger,
            "Medicine Silver Layer Completed Successfully",
        )
        log_step(logger, "=" * 70)

    except Exception:

        logger.exception(
            "Medicine Silver Layer Pipeline Failed."
        )

        raise


if __name__ == "__main__":
    main()