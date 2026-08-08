from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
import torch

from chronos import Chronos2Pipeline

from finemed_ai.demand_forecasting.chronos_pipeline import (
    ChronosPipelineLoader,
)

logger = logging.getLogger(__name__)

class ForecastingEngine:
    def __init__(
            self,
            pipeline_loader: ChronosPipelineLoader,
            prediction_length: int = 30,
            context_length: int = 730,
            quantiles: Optional[list[float]] = None) -> None:
        self.pipeline = pipeline_loader.get_pipeline()

        self.prediction_length = prediction_length

        self.context_length = context_length

        self.quantiles = (
            quantiles
            if quantiles is not None
            else [
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
        )

    def _prepare_history(self,history_df: pd.DataFrame) -> torch.Tensor:
        if history_df.empty:
            raise ValueError("History dataframe is empty.")

        if "target" not in history_df.columns:
            raise KeyError("'target' column not found in history dataframe.")

        # Sort by timestamp
        history_df = (history_df.sort_values("timestamp").reset_index(drop=True))

        # Extract only demand values
        demand = history_df["target"].astype(float).to_numpy()

        # Keep only latest context window
        if len(demand) > self.context_length:
            demand = demand[-self.context_length:]

        # Convert to tensor
        context = torch.tensor(demand,dtype=torch.float32,)

        return context

    def forecast_single_medicine(self,history_df: pd.DataFrame) -> pd.DataFrame:
        if history_df.empty:
            raise ValueError("History dataframe is empty.")

        medicine_id = history_df["item_id"].iloc[0]
        logger.info("Generating Forecast for Medicine : %s",medicine_id)

        # Prepare Context
        context = self._prepare_history(history_df)

        # Forecast
        forecast = self.pipeline.predict(context=context,prediction_length=self.prediction_length)

        # Convert Prediction
        median_prediction = (forecast.quantile(0.5).cpu().numpy().flatten())

        # Forecast Dates
        last_date = history_df["timestamp"].max()
        forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1),periods=self.prediction_length,freq="D")

        # Final Output
        forecast_df = pd.DataFrame({
            "Medicine_ID": medicine_id,
            "Forecast_Date": forecast_dates,
            "Predicted_Demand": median_prediction})

        return forecast_df

    def append_forecast_to_history(
        self,
        history_df: pd.DataFrame,
        forecast_df: pd.DataFrame) -> pd.DataFrame:

        if history_df.empty:
            raise ValueError("History dataframe is empty.")

        if forecast_df.empty:
            raise ValueError("Forecast dataframe is empty.")

        medicine_id = history_df["item_id"].iloc[0]

        # Convert forecast into history format
        forecast_history = pd.DataFrame({
            "item_id": medicine_id,
            "timestamp": pd.to_datetime(forecast_df["Forecast_Date"]),
            "target": forecast_df["Predicted_Demand"].astype(float)})

        # Append
        updated_history = pd.concat([
            history_df,
            forecast_history],
            ignore_index=True)

        # Sort
        updated_history = (updated_history.sort_values("timestamp").reset_index(drop=True))

        logger.info("History Updated | Medicine=%s | Rows=%d",medicine_id,len(updated_history))

        return updated_history

    from tqdm import tqdm

    def forecast_batch(
        self,
        prepared_history: pd.DataFrame) -> pd.DataFrame:

        if prepared_history.empty:
            raise ValueError("Prepared history dataframe is empty.")

        logger.info("=" * 80)
        logger.info("Starting Batch Forecast")
        logger.info("=" * 80)

        medicine_ids = sorted(prepared_history["item_id"].unique())

        logger.info("Medicines to Forecast : %d",len(medicine_ids))

        all_forecasts = []

        for medicine_id in tqdm(medicine_ids,desc="Forecasting Medicines"):

            history = (prepared_history[prepared_history["item_id"] == medicine_id].copy())

            forecast = self.forecast_single_medicine(history)

            all_forecasts.append(forecast)

        forecast_df = pd.concat(all_forecasts,ignore_index=True)
        forecast_df = (forecast_df.sort_values(["Medicine_ID","Forecast_Date"]).reset_index(drop=True))

        logger.info("=" * 80)
        logger.info("Batch Forecast Completed")
        logger.info("=" * 80)

        logger.info("Rows      : %d",len(forecast_df))

        logger.info("Medicines : %d",forecast_df["Medicine_ID"].nunique())

        return forecast_df
