"""
Main ETL Pipeline Orchestrator.

Runs the Finemed Pharma AI pipeline in the correct order.
"""

from __future__ import annotations

from finemed_ai.utils.logger import get_logger

# Extract
from finemed_ai.extract.run_extract  import run_extract

# Validation
from finemed_ai.validation.run_validation import run_all_validations

# Warehouse
from finemed_ai.warehouse.run_warehouse import build_warehouse

# Database
from finemed_ai.database.run_database import run_database


logger = get_logger(__name__)


def run_pipeline() -> None:
    """ Execute the complete ETL Pipeline. """

    logger.info("=" * 80)
    logger.info("Starting Finemed Pharma AI Pipeline")
    logger.info("=" * 80)

    try:

        # STEP 1 : Extract
        logger.info("STEP 1 : Extract Layer")
        
        extracted_tables = run_extract()
        
        # STEP 2 : Validation
        logger.info("STEP 2 : Validation Layer")
        validated = run_all_validations(extracted_tables)
        validated_tables = validated["tables"]
        validation_reports = validated["reports"]
        
        # STEP 3 : Warehouse
        logger.info("STEP 3 : Warehouse Layer")
        warehouse_tables = build_warehouse(validated_tables)
        
        # STEP 4 : Database
        logger.info("STEP 4 : Database Layer")
        run_database(warehouse_tables=warehouse_tables)

        logger.info("=" * 80)
        logger.info("Pipeline Completed Successfully")
        logger.info("=" * 80)

    except Exception:

        logger.exception(
            "Pipeline Execution Failed."
        )

        raise


if __name__ == "__main__":

    run_pipeline()