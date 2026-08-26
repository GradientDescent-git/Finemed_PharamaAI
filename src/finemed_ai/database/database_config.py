from dataclasses import dataclass
from pathlib import Path
import os
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parents[3]
ENV_PATH = BASE_DIR / ".env"

load_dotenv(dotenv_path=ENV_PATH, override=True)


@dataclass(frozen=True)
class DatabaseConfig:
    host: str
    port: int
    database: str
    user: str
    password: str
    schema: str

    def validate(self) -> None:

        """Validate database configuration when connection is requested."""
        if not self.password:
            raise RuntimeError(
                "POSTGRES_PASSWORD is not configured. "
                "Please configure POSTGRES_PASSWORD in your environment or .env file."
            )


DATABASE_CONFIG = DatabaseConfig(
    host=os.getenv("POSTGRES_HOST", "localhost"),
    port=int(os.getenv("POSTGRES_PORT", "5432")),
    database=os.getenv("POSTGRES_DB", "finemed_aiDB"),
    user=os.getenv("POSTGRES_USER", "postgres"),
    password=os.getenv("POSTGRES_PASSWORD", ""),
    schema=os.getenv("POSTGRES_SCHEMA", "warehouse"),
)


def get_database_config(validate: bool = True) -> DatabaseConfig:
    """Get DatabaseConfig with lazy validation."""
    if validate:
        DATABASE_CONFIG.validate()
    return DATABASE_CONFIG