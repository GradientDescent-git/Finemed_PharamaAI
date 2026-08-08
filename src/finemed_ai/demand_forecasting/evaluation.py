from __future__ import annotations

import logging
import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


class EvaluationEngine:
    def __init__(self,output_dir: Path) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True,exist_ok=True)

    def prepare_evaluation_dataframe(self,actual_df: pd.DataFrame,forecast_df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Preparing Evaluation Dataset")
        actual = actual_df.rename(columns={
            "item_id": "Medicine_ID",
            "timestamp": "Forecast_Date",
            "target": "Actual_Demand"})

        merged = actual.merge(forecast_df,on=["Medicine_ID","Forecast_Date"],how="inner",)
        logger.info("Rows : %d",len(merged))

        return merged

    def calculate_metrics(self,evaluation_df: pd.DataFrame) -> dict:
        logger.info("Calculating Evaluation Metrics")
        actual = evaluation_df["Actual_Demand"].astype(float)
        predicted = evaluation_df["Predicted_Demand"].astype(float)

        #Core Metrics 
        mae = mean_absolute_error(actual,predicted,)
        rmse = np.sqrt(mean_squared_error(actual,predicted))

        mask = actual != 0

        #MAPE
        if mask.sum() > 0:
            mape = (np.abs((actual[mask]- predicted[mask])/ actual[mask]).mean()) * 100

        else:
            mape = np.nan

        #WAPE
        denominator = np.abs(actual).sum()
        if denominator == 0:
            wape = np.nan
        else:
            wape = (np.abs(actual - predicted).sum() / denominator) * 100

        #Bias 
        bias = (predicted- actual).mean()

        #R2
        r2 = r2_score(actual,predicted)

        metrics = {

        "MAE": round(mae, 4),

        "RMSE": round(rmse, 4),

        "MAPE": round(mape, 4),

        "WAPE": round(wape, 4),

        "Bias": round(bias, 4),

        "R2": round(r2, 4)}

        logger.info("Metrics Calculated Successfully")

        return metrics

    def overall_metrics(self,evaluation_df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Calculating Overall Metrics")
        metrics = self.calculate_metrics(evaluation_df)
        overall = pd.DataFrame([metrics])
        logger.info("Overall Metrics Completed")
        return overall

    def save_reports(self,medicine_metrics: pd.DataFrame,overall_metrics: pd.DataFrame) -> None:
        medicine_path = (self.output_dir/ "medicine_metrics.parquet")
        overall_path = (self.output_dir/ "overall_metrics.parquet")

        medicine_metrics.to_parquet(medicine_path,index=False,)

        overall_metrics.to_parquet(overall_path,index=False,)
        logger.info("Evaluation Reports Saved")

    def run_evaluation(self,actual_df: pd.DataFrame,forecast_df: pd.DataFrame) -> dict:
        evaluation_df = (self.prepare_evaluation_dataframe(actual_df,forecast_df,))
        medicine_metrics = (self.medicine_wise_metrics(evaluation_df))
        overall_metrics = (self.overall_metrics(evaluation_df))
        self.save_reports(medicine_metrics, overall_metrics,)
        return {
            "evaluation": evaluation_df,
            "medicine_metrics": medicine_metrics,
            "overall_metrics": overall_metrics}