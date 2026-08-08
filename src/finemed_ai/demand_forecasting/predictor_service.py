import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

from chronos import Chronos2Pipeline

from finemed_ai.config.settings import Settings
from finemed_ai.utils.logger import get_logger

logger = get_logger(__name__)


class PredictorService:
    def __init__(self,settings: Settings,) -> None:
        self.settings = settings
        self.pipeline: Optional[Chronos2Pipeline] = None
        logger.info("PredictorService Initialized")

    def load_pipeline(self) -> None:
        if self.pipeline is not None:
            logger.info("Chronos pipeline already loaded.")
            return
        logger.info("Loading Chronos-2 Pipeline...")
        device = ("cuda" if torch.cuda.is_available()else "cpu")
        dtype = (torch.bfloat16 if torch.cuda.is_available() else torch.float32)

        self.pipeline = (Chronos2Pipeline.from_pretrained(self.settings.CHRONOS_MODEL_NAME,device_map=device,torch_dtype=dtype,))
        logger.info("Chronos-2 Pipeline Loaded Successfully.")
        logger.info("Device: %s | Dtype: %s",device,dtype)

    def load_history(self,history_path: Path | None = None) -> pd.DataFrame:
        if history_path is None:
            history_path = self.settings.PREPARED_HISTORY_FILE
        logger.info("Loading history dataset from %s",history_path)

        if not history_path.exists():
            raise FileNotFoundError(f"History file not found: {history_path}")

        history_df = pd.read_parquet(history_path)
        logger.info("History loaded successfully.")
        logger.info("Rows: %d | Medicines: %d",len(history_df),history_df["item_id"].nunique())

        return history_df

    def prepare_history(self,history_df: pd.DataFrame,medicine_id: str) -> pd.DataFrame:
        logger.info("Preparing history for medicine: %s",medicine_id)

        medicine_history = (history_df[history_df["item_id"] == medicine_id].copy().sort_values("timestamp").reset_index(drop=True))

        if medicine_history.empty:
            raise ValueError(f"No history found for medicine '{medicine_id}'")

        logger.info("History prepared successfully.")
        logger.info("History Length: %d",len(medicine_history),)
        return medicine_history

    def forecast_single(self,history: pd.DataFrame) -> pd.DataFrame:
        if self.pipeline is None:
            raise RuntimeError(
                "Chronos pipeline has not been loaded. "
                "Call load_pipeline() first.")

        logger.info("Forecasting medicine: %s",history["item_id"].iloc[0],)

        history_values = torch.tensor(history["target"].values,dtype=torch.float32,)

        forecast = self.pipeline.predict(context=history_values,prediction_length=self.settings.PREDICTION_LENGTH)        quantile_predictions = (
        forecast.quantiles[0].cpu().numpy())

        median_index = (self.settings.QUANTILES.index(0.5))
        predicted_demand = quantile_predictions[ :, median_index,]

        forecast_dates = pd.date_range(start=history["timestamp"].max()+ pd.Timedelta(days=1),periods=self.settings.PREDICTION_LENGTH,freq="D")
        forecast_df = pd.DataFrame({
            "Medicine_ID": history["item_id"].iloc[0],"Forecast_Date": forecast_dates,"Predicted_Demand": predicted_demand,})

        logger.info("Forecast generated successfully.")

        return forecast_df

    def run_pipeline(self,history_path: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
        logger.info("=" * 80)
        logger.info("Starting Demand Forecasting Pipeline")
        logger.info("=" * 80)

        # Load History
        history_df = self.load_history(history_path)

        # Forecast
        forecast_df = self.forecast_multiple(history_df)

        # Summary
        summary_df = self.forecast_summary(forecast_df)

        # Save Outputs

        self.save_forecast(forecast_df,summary_df)

        logger.info("=" * 80)
        logger.info("Demand Forecasting Completed")
        logger.info("=" * 80)

        return (forecast_df,summary_df)

    

