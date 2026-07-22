"""
Database connection utilities for Finemed Pharma AI.
"""

from __future__ import annotations

from urllib.parse import quote_plus

import psycopg2
from psycopg2.extensions import connection
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

from finemed_ai.database.database_config import DATABASE_CONFIG
from finemed_ai.utils.logger import get_logger

logger = get_logger(__name__)


def get_engine() -> Engine:
    """
    Create and return a reusable SQLAlchemy Engine.
    """

    try:
        encoded_password = quote_plus(DATABASE_CONFIG.password)

        database_url = (
            f"postgresql+psycopg2://"
            f"{DATABASE_CONFIG.user}:"
            f"{encoded_password}@"
            f"{DATABASE_CONFIG.host}:"
            f"{DATABASE_CONFIG.port}/"
            f"{DATABASE_CONFIG.database}"
        )

        engine = create_engine(
            database_url,
            future=True,
            pool_pre_ping=True,
        )

        logger.info("SQLAlchemy Engine created successfully.")

        return engine

    except Exception:
        logger.exception("Failed to create SQLAlchemy Engine.")
        raise


def get_connection() -> connection:
    """
    Create and return a PostgreSQL connection.
    """

    logger.info(
        "Connecting to PostgreSQL database: %s",
        DATABASE_CONFIG.database,
    )

    try:
        conn = psycopg2.connect(
            host=DATABASE_CONFIG.host,
            port=DATABASE_CONFIG.port,
            dbname=DATABASE_CONFIG.database,
            user=DATABASE_CONFIG.user,
            password=DATABASE_CONFIG.password,
        )

        logger.info("Database connection established successfully.")

        return conn

    except Exception:
        logger.exception("Failed to connect to PostgreSQL.")
        raise


def close_connection(conn: connection) -> None:
    """
    Close an active PostgreSQL connection.
    """

    if conn is not None:
        conn.close()
        logger.info("Database connection closed.")