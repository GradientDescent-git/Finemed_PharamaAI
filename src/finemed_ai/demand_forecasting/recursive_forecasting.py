from __future__ import annotations

import logging
from typing import List

import pandas as pd
from tqdm import tqdm

from finemed_ai.demand_forecasting.forecasting_engine import (
    ForecastingEngine,
)

logger = logging.getLogger(__name__)


class RecursiveForecasting:
    def __init__(self,forecasting_engine: ForecastingEngine) -> None:
        self.engine = forecasting_engine

    def recursive_forecast_single_medicine(self,history_df: pd.DataFrame,forecast_end_date: pd.Timestamp) -> pd.DataFrame:
        if history_df.empty:
            raise ValueError("History dataframe is empty.")

        medicine_id = history_df["item_id"].iloc[0]
        logger.info("Starting Recursive Forecast | Medicine=%s",medicine_id)

        current_history = history_df.copy()
        recursive_forecasts = []

        while True:
            # Forecast Next Window
            forecast = self.engine.forecast_single_medicine(current_history)
            recursive_forecasts.append(forecast)

            # Append Forecast to History
            current_history = (self.engine.append_forecast_to_history(current_history,forecast))

            # Stop Condition
            latest_date = forecast["Forecast_Date"].max()

            if latest_date >= forecast_end_date:
                break

        # Merge All Windows
        recursive_forecast_df = pd.concat(recursive_forecasts,ignore_index=True)

        # Trim Extra Forecast Days
        recursive_forecast_df = (recursive_forecast_df[recursive_forecast_df["Forecast_Date"] <= forecast_end_date].sort_values("Forecast_Date").reset_index(drop=True))

        logger.info("Recursive Forecast Completed | Medicine=%s | Rows=%d",medicine_id,len(recursive_forecast_df))

        return recursive_forecast_df

    from tqdm import tqdm

    def recursive_forecast_validation(self,prepared_train: pd.DataFrame,forecast_end_date: pd.Timestamp) -> pd.DataFrame:
        if prepared_train.empty:
            raise ValueError("Prepared history dataframe is empty.")

        logger.info("=" * 80)
        logger.info("Starting Recursive Validation Forecast")
        logger.info("=" * 80)

        medicine_ids = sorted(
            prepared_train["item_id"].unique())
        logger.info("Medicines : %d",len(medicine_ids))

        forecasts = []

        for medicine_id in tqdm(medicine_ids,desc="Validation Forecast"):

            medicine_history = (prepared_train[prepared_train["item_id"] == medicine_id].copy())
            medicine_forecast = (self.recursive_forecast_single_medicine(history_df=medicine_history,forecast_end_date=forecast_end_date))
            forecasts.append(medicine_forecast)

        recursive_validation_df = pd.concat(forecasts,ignore_index=True)
        recursive_validation_df = (recursive_validation_df.sort_values(["Medicine_ID","Forecast_Date"]).reset_index(drop=True))
        logger.info("=" * 80)
        logger.info("Recursive Validation Forecast Completed")
        logger.info("=" * 80)
        logger.info("Rows      : %d",len(recursive_validation_df))
        logger.info("Medicines : %d",recursive_validation_df["Medicine_ID"].nunique())

        return recursive_validation_df

    def recursive_forecast_production(self,prepared_history: pd.DataFrame,forecast_end_date: pd.Timestamp) -> pd.DataFrame:
        if prepared_history.empty:
            raise ValueError("Prepared history dataframe is empty.")

        logger.info("=" * 80)
        logger.info("Starting Production Forecast")
        logger.info("=" * 80)

        medicine_ids = sorted(prepared_history["item_id"].unique())
        forecasts = []

        for medicine_id in tqdm(medicine_ids,desc="Production Forecast"):
            medicine_history = (prepared_history[prepared_history["item_id"] == medicine_id].copy())
            medicine_forecast = (self.recursive_forecast_single_medicine(history_df=medicine_history,forecast_end_date=forecast_end_date,))
            forecasts.append(medicine_forecast)

        production_forecast = pd.concat(forecasts,ignore_index=True)
        production_forecast = (production_forecast.sort_values(["Medicine_ID","Forecast_Date"]).reset_index(drop=True))

        logger.info("=" * 80)
        logger.info("Production Forecast Completed")
        logger.info("=" * 80)
        logger.info("Rows : %d",len(production_forecast))
        logger.info("Medicines : %d",production_forecast["Medicine_ID"].nunique())
        return production_forecast
