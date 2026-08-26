from pathlib import Path


class Settings:
    """
    Project-wide application settings.

    Silver demand data and production forecast outputs are intentionally
    separated:

        Silver:
            data/04_silver/demand_forecasting/

        Gold production forecasts:
            data/05_gold/demand_forecasting/production_forecasts/
    """

    # ------------------------------------------------------------------
    # Project paths
    # ------------------------------------------------------------------

    PROJECT_ROOT = Path(__file__).resolve().parents[3]

    DATA_DIR = PROJECT_ROOT / "data"

    SILVER_DIR = DATA_DIR / "04_silver"

    GOLD_DIR = DATA_DIR / "05_gold"

    # ------------------------------------------------------------------
    # Demand forecasting paths
    # ------------------------------------------------------------------

    # Historical / prepared demand data
    DEMAND_DIR = (
        SILVER_DIR
        / "demand_forecasting"
    )

    # Versioned production forecast runs
    FORECAST_DIR = (
        GOLD_DIR
        / "demand_forecasting"
        / "production_forecasts"
    )

    # ------------------------------------------------------------------
    # Forecasting model configuration
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Forecast configuration
    # ------------------------------------------------------------------

    FORECAST_HORIZON_DAYS = 30

    TRAIN_END_DATE = "2024-12-31"

    VALIDATION_END_DATE = "2025-04-30"

    # ------------------------------------------------------------------
    # Silver-layer demand files
    # ------------------------------------------------------------------

    DEMAND_FILE = (
        DEMAND_DIR
        / "daily_demand.parquet"
    )

    PREPARED_HISTORY_FILE = (
        DEMAND_DIR
        / "prepared_history.parquet"
    )

    # ------------------------------------------------------------------
    # Production forecast files
    # ------------------------------------------------------------------

    # Current published forecast.
    # This is only replaced when the publication quality gate passes.
    LATEST_FORECAST_FILE = (
        FORECAST_DIR
        / "latest.parquet"
    )

    # Directory containing forecast evaluation outputs.
    FORECAST_EVALUATION_DIR = (
        FORECAST_DIR
        / "evaluations"
    )

    # Directory containing operational forecast alerts.
    FORECAST_ALERT_DIR = (
        FORECAST_DIR
        / "alerts"
    )

    # Backward-compatible alias.
    # Existing code referencing FORECAST_FILE will now point to the
    # published production forecast rather than the Silver layer.
    FORECAST_FILE = LATEST_FORECAST_FILE