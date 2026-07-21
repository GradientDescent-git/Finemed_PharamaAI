"""
Extract Layer Runner.

Reads all monthly ERP DAT files and returns them as
a dictionary of pandas DataFrames.
"""

from __future__ import annotations

import pandas as pd

from finemed_ai.config.paths import RAW_DATA_DIR
from finemed_ai.extract.read_dat import read_all_months
from finemed_ai.validation.validation_config import REQUIRED_FILES
from finemed_ai.utils.logger import get_logger

logger = get_logger(__name__)


def run_extract() -> dict[str, pd.DataFrame]:
    """
    Execute the complete Extract Layer.

    Returns
    -------
    dict[str, pd.DataFrame]
        Dictionary containing all extracted ERP tables.
    """

    logger.info("=" * 80)
    logger.info("Starting Extract Layer")
    logger.info("=" * 80)

    try:

        extracted_tables = read_all_months(
            raw_data_dir=RAW_DATA_DIR,
            required_files=REQUIRED_FILES,
        )

        logger.info(
            "Extract Layer Completed Successfully | Tables=%d",
            len(extracted_tables),
        )

        return extracted_tables

    except Exception as error:

        logger.exception(
            "Extract Layer Failed."
        )

        raise RuntimeError(
            "Extract layer execution failed."
        ) from error


if __name__ == "__main__":

    tables = run_extract()

    for table_name, dataframe in tables.items():

        print(
            f"{table_name:<20} "
            f"Rows={len(dataframe):>8} "
            f"Cols={len(dataframe.columns):>3}"
        )