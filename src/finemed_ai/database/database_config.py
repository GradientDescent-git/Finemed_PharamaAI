"""
Database configuration for Finemed Pharma AI.
"""

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class DatabaseConfig:
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
    password=os.getenv("POSTGRES_PASSWORD"),
    schema="warehouse",
)