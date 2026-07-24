from dataclasses import dataclass
import os

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class DatabaseConfig:
    host: str
    port: int
    database: str
    user: str
    password: str
    schema: str


DATABASE_CONFIG = DatabaseConfig(
    host=os.getenv("POSTGRES_HOST", "localhost"),
    port=int(os.getenv("POSTGRES_PORT", 5432)),
    database=os.getenv("POSTGRES_DB", "finemed_aiDB"),
    user=os.getenv("POSTGRES_USER", "postgres"),
    password=os.getenv("POSTGRES_PASSWORD"),
    schema=os.getenv("POSTGRES_SCHEMA", "warehouse"),
)

if DATABASE_CONFIG.password is None:
    raise RuntimeError(
        "POSTGRES_PASSWORD is not configured. "
        "Create a .env file from .env.example."
    )