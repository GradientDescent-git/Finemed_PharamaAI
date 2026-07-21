from __future__ import annotations

import pandas as pd

from finemed_ai.database.db_connection import get_engine
from finemed_ai.utils.logger import get_logger

from sqlalchemy import inspect


logger = get_logger(__name__)


def upload_dataframe(
    dataframe: pd.DataFrame,
    table_name: str,
    schema: str = "warehouse",
    load_mode: str = "replace",
) -> None:
    """
    Upload a warehouse DataFrame into PostgreSQL.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Warehouse DataFrame to upload.

    table_name : str
        Destination PostgreSQL table name.

    schema : str, default="warehouse"
        PostgreSQL schema.

    load_mode : str, default="replace"
        Table loading strategy.

        Supported values:
            - replace
            - append
    """

    logger.info(
        "Uploading '%s' into schema '%s' (mode=%s)",
        table_name,
        schema,
        load_mode,
    )

    # Validate load mode

    if load_mode not in ("replace", "append"):

        raise ValueError(
            "load_mode must be either "
            "'replace' or 'append'."
        )

    
    # Validate dataframe

    if dataframe.empty:

        raise ValueError(
            f"{table_name} is empty. "
            "Upload aborted."
        )

    try:

        engine = get_engine()

        dataframe.to_sql(
            name=table_name,
            con=engine,
            schema=schema,
            if_exists=load_mode,
            index=False,
            method="multi",
        )

        logger.info(
            "Successfully uploaded '%s' | rows=%d | columns=%d",
            table_name,
            len(dataframe),
            len(dataframe.columns),
        )

    except Exception:

        logger.exception(
            "Failed uploading '%s'",
            table_name,
        )

        raise


def load_warehouse(
    warehouse_tables: dict[str, pd.DataFrame],
    schema: str = "warehouse",
    load_mode: str = "replace",
) -> None:
    """
    Upload all warehouse tables into PostgreSQL.

    Parameters
    ----------
    warehouse_tables : dict[str, pd.DataFrame]
        Dictionary containing warehouse table names and DataFrames.

    schema : str, default="warehouse"
        Destination PostgreSQL schema.

    load_mode : str, default="replace"
        Table loading strategy.

        Supported values
        ----------------
        replace
            Replace the existing table.

        append
            Append records to an existing table.
    """

    logger.info(
        "Starting Warehouse Upload | Tables=%d | Schema=%s | Mode=%s",
        len(warehouse_tables),
        schema,
        load_mode,
    )

    # Validate warehouse dictionary


    if not warehouse_tables:

        raise ValueError(
            "Warehouse dictionary is empty."
        )

    try:

        for table_name, dataframe in warehouse_tables.items():

            logger.info(
                "Uploading warehouse table: %s",
                table_name,
            )

            upload_dataframe(
                dataframe=dataframe,
                table_name=table_name,
                schema=schema,
                load_mode=load_mode,
            )

        logger.info(
            "Warehouse upload completed successfully."
        )

    except Exception:

        logger.exception(
            "Warehouse upload failed."
        )

        raise

def table_exists(
    table_name: str,
    schema: str = "warehouse",
) -> bool:
    """
    Check whether a PostgreSQL table exists.

    Parameters
    ----------
    table_name : str
        Name of the PostgreSQL table.

    schema : str, default="warehouse"
        PostgreSQL schema.

    Returns
    -------
    bool
        True if table exists.
        False otherwise.
    """

    logger.info(
        "Checking if table '%s.%s' exists.",
        schema,
        table_name,
    )

    try:

        engine = get_engine()

        inspector = inspect(engine)

        exists = inspector.has_table(
            table_name,
            schema=schema,
        )

        if exists:

            logger.info(
                "Table '%s.%s' exists.",
                schema,
                table_name,
            )

        else:

            logger.warning(
                "Table '%s.%s' does not exist.",
                schema,
                table_name,
            )

        return exists

    except Exception:

        logger.exception(
            "Failed while checking table '%s.%s'.",
            schema,
            table_name,
        )

        raise


def verify_upload(
    dataframe: pd.DataFrame,
    table_name: str,
    schema: str = "warehouse",
) -> bool:
    """
    Verify that a DataFrame has been uploaded successfully.

    Verification Steps
    ------------------
    1. Check that the table exists.
    2. Compare DataFrame row count with PostgreSQL row count.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Original uploaded DataFrame.

    table_name : str
        PostgreSQL table name.

    schema : str, default="warehouse"
        PostgreSQL schema.

    Returns
    -------
    bool
        True if verification succeeds.

    Raises
    ------
    RuntimeError
        If verification fails.
    """

    logger.info(
        "Verifying upload for '%s.%s'",
        schema,
        table_name,
    )

    try:

        # Verify table exists

        if not table_exists(
            table_name=table_name,
            schema=schema,
        ):

            raise RuntimeError(
                f"Table '{schema}.{table_name}' "
                "does not exist."
            )

        engine = get_engine()

        # Read database row count

        query = (
            f'SELECT COUNT(*) AS total_rows '
            f'FROM "{schema}"."{table_name}"'
        )

        database_rows = int(
            pd.read_sql(
                query,
                engine,
            ).iloc[0]["total_rows"]
        )

        dataframe_rows = len(dataframe)

        # Compare counts

        if dataframe_rows != database_rows:

            raise RuntimeError(
                f"Upload verification failed for "
                f"{table_name}. "
                f"DataFrame rows={dataframe_rows}, "
                f"Database rows={database_rows}"
            )

        logger.info(
            "Upload verified successfully | "
            "rows=%d",
            dataframe_rows,
        )

        return True

    except Exception:

        logger.exception(
            "Upload verification failed for '%s.%s'",
            schema,
            table_name,
        )

        raise