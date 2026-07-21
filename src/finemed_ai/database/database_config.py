"""
Database configuration for Finemed Pharma AI.

This module stores all PostgreSQL connection settings
used throughout the application.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class DatabaseConfig:
    """
    Immutable configuration object for PostgreSQL.
    """

    host: str
    port: int
    database: str
    user: str
    password: str
    schema: str


DATABASE_CONFIG = DatabaseConfig(
    host="localhost",
    port=5432,
    database="finemed_aiDB",
    user="postgres",
    password="Vicky3061@postsql",
    schema="warehouse")    