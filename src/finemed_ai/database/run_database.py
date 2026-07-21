"""
Database Pipeline Runner

Loads warehouse tables into PostgreSQL.
"""

from __future__ import annotations

import pandas as pd

from finemed_ai.database.database_loader import (
    load_warehouse,
    verify_upload,
)

from finemed_ai.utils.logger import get_logger

logger = get_logger(__name__)


def run_database(
    warehouse_tables: dict[str, pd.DataFrame],
    schema: str = "warehouse",
    load_mode: str = "replace",
) -> None:
    """
    Load warehouse tables into PostgreSQL.

    Parameters
    ----------
    warehouse_tables : dict[str, pd.DataFrame]
        Dictionary containing warehouse DataFrames.

    schema : str, default="warehouse"
        PostgreSQL schema.

    load_mode : str, default="replace"
        Upload strategy.

        Supported values
        ----------------
        replace
            Replace existing tables.

        append
            Append new records.
    """

    logger.info("=" * 80)
    logger.info("Starting Database Pipeline")
    logger.info("=" * 80)

    try:

        if not warehouse_tables:

            raise ValueError(
                "Warehouse tables dictionary is empty."
            )

        
        # Upload Warehouse
        
        load_warehouse(
            warehouse_tables=warehouse_tables,
            schema=schema,
            load_mode=load_mode,
        )

        
        # Verify Upload
        

        logger.info(
            "Verifying uploaded warehouse tables..."
        )

        for table_name, dataframe in warehouse_tables.items():

            verify_upload(
                dataframe=dataframe,
                table_name=table_name,
                schema=schema,
            )

        logger.info("=" * 80)
        logger.info("Database Pipeline Completed Successfully")
        logger.info("=" * 80)

    except Exception:

        logger.exception(
            "Database Pipeline Failed."
        )

        raise