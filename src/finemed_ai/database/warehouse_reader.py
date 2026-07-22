"""
Warehouse Reader Module for Finemed Pharma AI.

This module is responsible for reading Warehouse tables
from PostgreSQL into Pandas DataFrames.

Responsibilities
----------------
- Read warehouse tables
- Validate table existence
- Return clean DataFrames
"""

from __future__ import annotations

import pandas as pd
from sqlalchemy import inspect
from sqlalchemy import text

from finemed_ai.database.db_connection import get_engine
from finemed_ai.utils.logger import get_logger

logger = get_logger(__name__)


def table_exists(
    table_name: str,
    schema: str = "warehouse",
) -> bool:
    """
    Check whether a warehouse table exists.

    Parameters
    ----------
    table_name : str
        Warehouse table name.

    schema : str
        PostgreSQL schema.

    Returns
    -------
    bool
        True if table exists.
    """

    engine = get_engine()

    inspector = inspect(engine)

    return inspector.has_table(
        table_name,
        schema=schema,
    )


def read_table(
    table_name: str,
    schema: str = "warehouse",
) -> pd.DataFrame:
    """
    Read a warehouse table from PostgreSQL.

    Parameters
    ----------
    table_name : str
        Warehouse table name.

    schema : str, default="warehouse"
        PostgreSQL schema.

    Returns
    -------
    pd.DataFrame
        Warehouse table.
    """

    logger.info(
        "Reading table '%s.%s'",
        schema,
        table_name,
    )

    try:

        if not table_exists(
            table_name=table_name,
            schema=schema,
        ):

            raise ValueError(
                f"Table '{schema}.{table_name}' does not exist."
            )

        engine = get_engine()

        query = text(
            f'SELECT * FROM "{schema}"."{table_name}"'
        )

        dataframe = pd.read_sql(
            query,
            engine,
        )

        logger.info(
            "Loaded '%s.%s' | Rows=%d | Columns=%d",
            schema,
            table_name,
            len(dataframe),
            len(dataframe.columns),
        )

        return dataframe

    except Exception:

        logger.exception(
            "Failed reading table '%s.%s'",
            schema,
            table_name,
        )

        raise


def read_multiple_tables(
    table_names: list[str],
    schema: str = "warehouse",
) -> dict[str, pd.DataFrame]:
    """
    Read multiple warehouse tables.

    Parameters
    ----------
    table_names : list[str]
        List of warehouse tables.

    schema : str
        PostgreSQL schema.

    Returns
    -------
    dict[str, pd.DataFrame]
    """

    logger.info(
        "Reading %d warehouse tables...",
        len(table_names),
    )

    warehouse_tables = {}

    for table in table_names:

        warehouse_tables[table] = read_table(
            table_name=table,
            schema=schema,
        )

    logger.info(
        "Successfully loaded %d warehouse tables.",
        len(warehouse_tables),
    )

    return warehouse_tables