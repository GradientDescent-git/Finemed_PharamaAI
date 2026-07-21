from __future__ import annotations

import pandas as pd

from finemed_ai.config.paths import RAW_DATA_DIR
from finemed_ai.utils.logger import get_logger

from finemed_ai.validation.datatype_validator import (
    validate_datatype_consistency,
)
from finemed_ai.validation.duplicate_validator import (
    validate_duplicates,
)
from finemed_ai.validation.file_validator import (
    validate_required_files,
)
from finemed_ai.validation.orphan_validator import (
    validate_orphans,
)
from finemed_ai.validation.profile_validator import (
    profile_tables,
)
from finemed_ai.validation.schema_validator import (
    validate_schema_consistency,
)
from finemed_ai.validation.validation_config import (
    REQUIRED_FILES,
    DUPLICATE_KEYS,
    ORPHAN_CHECKS,
)

logger = get_logger(__name__)


def run_all_validations(
    tables: dict[str, pd.DataFrame],
) -> dict[str, object]:
    """ Execute the complete Validation Layer. """

    logger.info("Validation layer started")

    try:

        logger.info("Running File Validation")

        file_report = validate_required_files(
            raw_data_dir=RAW_DATA_DIR,
            required_files=REQUIRED_FILES,
        )

        logger.info("Running Schema Validation")

        schema_report = validate_schema_consistency(
            raw_data_dir=RAW_DATA_DIR,
            required_files=REQUIRED_FILES,
        )

        logger.info("Running Datatype Validation")

        datatype_report = validate_datatype_consistency(
            tables,
        )

        logger.info("Running Duplicate Validation")

        duplicate_report, duplicate_samples = (
            validate_duplicates(
                tables=tables,
                duplicate_keys=DUPLICATE_KEYS,
            )
        )

        logger.info("Running Orphan Validation")

        orphan_report, orphan_samples = (
            validate_orphans(
                tables=tables,
                orphan_checks=ORPHAN_CHECKS,
            )
        )

        logger.info("Running Data Profiling")

        profile_report = profile_tables(
            tables,
        )

        reports = {
            "file_report": file_report,
            "schema_report": schema_report,
            "datatype_report": datatype_report,
            "duplicate_report": duplicate_report,
            "duplicate_samples": duplicate_samples,
            "orphan_report": orphan_report,
            "orphan_samples": orphan_samples,
            "profile_report": profile_report,
        }

        logger.info("Validation layer completed successfully")
        
        return {"tables": tables,"reports": reports}

    except Exception as error:

        logger.exception(
            "Validation layer execution failed."
        )

        raise RuntimeError(
            "Validation pipeline failed."
        ) from error