import os
import logging
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


class Settings:
    """
    Project-wide application settings with strict environment modes
    (development, testing, production) and fail-fast validation.
    """

    ENVIRONMENT = os.getenv("ENVIRONMENT", "development").lower()
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

    PROJECT_ROOT = Path(__file__).resolve().parents[3]
    DATA_DIR = PROJECT_ROOT / "data"
    SILVER_DIR = DATA_DIR / "04_silver"
    GOLD_DIR = DATA_DIR / "05_gold"

    DEMAND_DIR = SILVER_DIR / "demand_forecasting"
    FORECAST_DIR = GOLD_DIR / "demand_forecasting" / "production_forecasts"
    ROUTING_DIR = GOLD_DIR / "demand_forecasting" / "medicine_robustness"

    CHRONOS_MODEL_NAME = "amazon/chronos-2"
    CONTEXT_LENGTH = 730
    PREDICTION_LENGTH = 30
    QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    FORECAST_HORIZON_DAYS = 30
    TRAIN_END_DATE = "2024-12-31"
    VALIDATION_END_DATE = "2025-04-30"

    DEMAND_FILE = DEMAND_DIR / "daily_demand.parquet"
    PREPARED_HISTORY_FILE = DEMAND_DIR / "prepared_history.parquet"
    MEDICINE_SILVER_FILE = SILVER_DIR / "medicine" / "medicine_silver.parquet"

    LATEST_FORECAST_FILE = FORECAST_DIR / "latest.parquet"
    LATEST_ROUTING_FILE = ROUTING_DIR / "production_routing_table.parquet"
    FORECAST_EVALUATION_DIR = FORECAST_DIR / "evaluations"
    FORECAST_ALERT_DIR = FORECAST_DIR / "alerts"
    PERFORMANCE_HISTORY_FILE = FORECAST_DIR / "performance_history.parquet"
    DRIFT_REPORT_DIR = FORECAST_DIR / "drift_reports"

    FORECAST_FILE = LATEST_FORECAST_FILE

    ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "")
    CLIENT_API_KEY = os.getenv("CLIENT_API_KEY", "")

    MAX_UPLOAD_SIZE_BYTES = int(os.getenv("MAX_UPLOAD_SIZE_BYTES", 50 * 1024 * 1024))
    MAX_UNCOMPRESSED_SIZE_BYTES = int(os.getenv("MAX_UNCOMPRESSED_SIZE_BYTES", 200 * 1024 * 1024))

    @classmethod
    def validate_environment(cls) -> None:
        """
        Validate production requirements. Fail fast if required security
        credentials are unset when running in production mode.
        """
        if cls.ENVIRONMENT == "production":
            missing = []
            if not cls.ADMIN_TOKEN:
                missing.append("ADMIN_TOKEN")
            if not cls.CLIENT_API_KEY:
                missing.append("CLIENT_API_KEY")
            if missing:
                err_msg = (
                    f"CRITICAL PRODUCTION CONFIGURATION ERROR: Missing environment variables: {', '.join(missing)}. "
                    "Application cannot start in production mode without required credentials."
                )
                logger.critical(err_msg)
                raise RuntimeError(err_msg)
            logger.info("Production environment configuration validated successfully.")
        else:
            logger.info(f"Running in '{cls.ENVIRONMENT}' environment mode.")