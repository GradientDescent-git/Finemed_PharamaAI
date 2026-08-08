from pathlib import Path


class Settings:
    PROJECT_ROOT = Path(__file__).resolve().parents[3]

    DATA_DIR = PROJECT_ROOT / "data"

    SILVER_DIR = DATA_DIR / "04_silver"

    DEMAND_DIR = SILVER_DIR / "demand_forecasting"

    CHRONOS_MODEL_NAME = "amazon/chronos-2"

    CONTEXT_LENGTH = 730

    PREDICTION_LENGTH = 30

    QUANTILES = [
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
    ]

    # Forecast
    FORECAST_HORIZON_DAYS = 30

    TRAIN_END_DATE = "2024-12-31"

    VALIDATION_END_DATE = "2025-04-30"

    # Files
    DEMAND_FILE = DEMAND_DIR / "daily_demand.parquet"

    PREPARED_HISTORY_FILE = (
        DEMAND_DIR / "prepared_history.parquet"
    )

    FORECAST_FILE = (
        DEMAND_DIR / "forecast.parquet"
    )