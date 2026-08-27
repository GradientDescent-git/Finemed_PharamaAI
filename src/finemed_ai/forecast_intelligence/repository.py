from __future__ import annotations

import logging
from pathlib import Path
import pandas as pd

logger = logging.getLogger(__name__)


class ForecastRepository:
    """
    Read-only access layer for validated demand forecasting artifacts.

    If artifact files are missing, this repository returns empty DataFrames
    and flags `is_available() == False` to ensure honest failure handling without
    generating synthetic fallback data.
    """

    def __init__(
        self,
        forecast_path: str | Path = (
            "data/05_gold/demand_forecasting/production_forecasts/latest.parquet"
        ),
        routing_path: str | Path = (
            "data/05_gold/demand_forecasting/medicine_robustness/production_routing_table.parquet"
        ),
        medicine_path: str | Path = (
            "data/04_silver/medicine/medicine_silver.parquet"
        ),
    ) -> None:
        self.forecast_path = Path(forecast_path)
        self.routing_path = Path(routing_path)
        self.medicine_path = Path(medicine_path)

    def is_available(self) -> bool:
        """Return True if production forecast artifact exists and is non-empty."""
        if not self.forecast_path.exists() or not self.forecast_path.is_file():
            return False
        try:
            return self.forecast_path.stat().st_size > 0
        except Exception:
            return False

    def load_forecasts(self) -> pd.DataFrame:
        """Load currently promoted production forecast or empty DataFrame."""
        if not self.forecast_path.exists():
            logger.warning("Production forecast file not found at %s.", self.forecast_path)
            return pd.DataFrame()

        try:
            df = pd.read_parquet(self.forecast_path)
            return df.copy() if not df.empty else pd.DataFrame()
        except Exception as exc:
            logger.warning("Could not load production forecast from %s: %s", self.forecast_path, exc)
            return pd.DataFrame()

    def load_routing(self) -> pd.DataFrame:
        """Load production model routing decisions or empty DataFrame."""
        if not self.routing_path.exists():
            logger.warning("Production routing table not found at %s.", self.routing_path)
            return pd.DataFrame()

        try:
            df = pd.read_parquet(self.routing_path)
            return df.copy() if not df.empty else pd.DataFrame()
        except Exception as exc:
            logger.warning("Could not load production routing table from %s: %s", self.routing_path, exc)
            return pd.DataFrame()

    def load_medicines(self) -> pd.DataFrame:
        """Load medicine master data or empty DataFrame."""
        if not self.medicine_path.exists():
            logger.warning("Medicine master file not found at %s.", self.medicine_path)
            return pd.DataFrame()

        try:
            df = pd.read_parquet(self.medicine_path)
            return df.copy() if not df.empty else pd.DataFrame()
        except Exception as exc:
            logger.warning("Could not load medicine master from %s: %s", self.medicine_path, exc)
            return pd.DataFrame()

    def load_all(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load all intelligence artifacts."""
        forecasts = self.load_forecasts()
        routing = self.load_routing()
        medicines = self.load_medicines()

        logger.info(
            "Forecast intelligence artifacts loaded | forecasts=%d | routing=%d | medicines=%d",
            len(forecasts),
            len(routing),
            len(medicines),
        )

        return forecasts, routing, medicines

    def get_forecast_columns(self) -> list[str]:
        return self.load_forecasts().columns.tolist()

    def get_routing_columns(self) -> list[str]:
        return self.load_routing().columns.tolist()

    def get_medicine_columns(self) -> list[str]:
        return self.load_medicines().columns.tolist()