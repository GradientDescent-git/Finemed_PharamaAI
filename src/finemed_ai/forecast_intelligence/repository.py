from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd


logger = logging.getLogger(__name__)


class ForecastRepository:
    """
    Read-only access layer for validated demand forecasting artifacts.

    This class is the single place where the forecast intelligence
    layer reads production forecasting outputs.
    """

    def __init__(
        self,
        forecast_path: str | Path = (
            "data/05_gold/demand_forecasting/"
            "production_forecasts/latest.parquet"
        ),
        routing_path: str | Path = (
            "data/05_gold/demand_forecasting/"
            "medicine_robustness/production_routing_table.parquet"
        ),
        medicine_path: str | Path = (
            "data/04_silver/medicine/medicine_silver.parquet"
        ),
    ) -> None:
        self.forecast_path = Path(forecast_path)
        self.routing_path = Path(routing_path)
        self.medicine_path = Path(medicine_path)

    @staticmethod
    def _require_file(path: Path, label: str) -> None:
        if not path.exists():
            raise FileNotFoundError(
                f"{label} does not exist: {path}"
            )

    def load_forecasts(self) -> pd.DataFrame:
        """Load the currently promoted production forecast."""

        self._require_file(
            self.forecast_path,
            "Production forecast file",
        )

        logger.info(
            "Loading production forecasts from %s",
            self.forecast_path,
        )

        df = pd.read_parquet(self.forecast_path)

        if df.empty:
            raise ValueError(
                f"Production forecast is empty: {self.forecast_path}"
            )

        return df.copy()

    def load_routing(self) -> pd.DataFrame:
        """Load frozen production model routing decisions."""

        self._require_file(
            self.routing_path,
            "Production routing table",
        )

        logger.info(
            "Loading production routing from %s",
            self.routing_path,
        )

        df = pd.read_parquet(self.routing_path)

        if df.empty:
            raise ValueError(
                f"Production routing table is empty: "
                f"{self.routing_path}"
            )

        return df.copy()

    def load_medicines(self) -> pd.DataFrame:
        """
        Load medicine master data.

        This is used only for entity resolution and employee-friendly
        medicine names. Forecast values always come from the
        production forecast artifact.
        """

        self._require_file(
            self.medicine_path,
            "Medicine master file",
        )

        logger.info(
            "Loading medicine master from %s",
            self.medicine_path,
        )

        df = pd.read_parquet(self.medicine_path)

        if df.empty:
            raise ValueError(
                f"Medicine master is empty: {self.medicine_path}"
            )

        return df.copy()

    def load_all(
        self,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load all intelligence artifacts.

        Returns:
            forecasts, routing, medicines
        """

        forecasts = self.load_forecasts()
        routing = self.load_routing()
        medicines = self.load_medicines()

        logger.info(
            "Forecast intelligence artifacts loaded | "
            "forecasts=%d | routing=%d | medicines=%d",
            len(forecasts),
            len(routing),
            len(medicines),
        )

        return forecasts, routing, medicines

    def get_forecast_columns(self) -> list[str]:
        """Return production forecast schema."""

        return self.load_forecasts().columns.tolist()

    def get_routing_columns(self) -> list[str]:
        """Return production routing schema."""

        return self.load_routing().columns.tolist()

    def get_medicine_columns(self) -> list[str]:
        """Return medicine master schema."""

        return self.load_medicines().columns.tolist()