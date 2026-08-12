from dataclasses import dataclass
from pathlib import Path
import os

from dotenv import load_dotenv

# Project root:
# E:\Finemed_PharmaAI
BASE_DIR = Path(__file__).resolve().parents[3]
ENV_PATH = BASE_DIR / ".env"

# Load the project's .env and allow it to override stale environment variables.
load_dotenv(dotenv_path=ENV_PATH, override=True)


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
    port=int(os.getenv("POSTGRES_PORT", "5432")),
    database=os.getenv("POSTGRES_DB", "finemed_aiDB"),
    user=os.getenv("POSTGRES_USER", "postgres"),
    password=os.getenv("POSTGRES_PASSWORD"),
    schema=os.getenv("POSTGRES_SCHEMA", "warehouse"),
)

if DATABASE_CONFIG.password is None:
    raise RuntimeError(
        "POSTGRES_PASSWORD is not configured.\n"
        "Please create a .env file from .env.example "
        "and fill in your PostgreSQL password."
    )