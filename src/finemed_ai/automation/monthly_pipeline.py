from __future__ import annotations

from finemed_ai.config.settings import Settings
from finemed_ai.utils.logger import get_logger

from finemed_ai.pipeline.run_pipeline import run_pipeline

from finemed_ai.demand_forecasting.data_preparation import (
    DemandDataPreparation
)

from finemed_ai.demand_forecasting.predictor_service import (
    PredictorService,
)

from finemed_ai.demand_forecasting.evaluation import (
    EvaluationEngine
)

logger = get_logger(__name__)

class MonthlyPipeline:

    def __init__(self,settings: Settings):

        self.settings = settings

        self.predictor = PredictorService(settings)

        self.preparation = DemandDataPreparation(settings)

        self.evaluation = EvaluationEngine(settings)

    def run_etl_pipeline(self) -> None:
        logger.info("Running ETL Pipeline...")

        run_pipeline()
        logger.info("ETL Pipeline Completed.")

    def prepare_demand_data(self) -> None:
        logger.info("Preparing Demand Dataset...")
        self.preparation.run()

        logger.info("Demand Dataset Ready.")

    def generate_forecast(self):
        logger.info("Generating Forecast...")

        forecast_df, summary_df = (self.predictor.run_pipeline())

        logger.info("Forecast Generated.")

        return forecast_df, summary_df

    def evaluate_forecast(self):
        logger.info("Evaluating Forecast...")

        self.evaluation.run()

        logger.info("Evaluation Completed.")

    def refresh_llm(self):
        """Refresh Vector Database with latest forecast."""

    def run(self):
        self.run_etl_pipeline()
        self.prepare_demand_data()
        self.generate_forecast()
        self.evaluate_forecast()
        self.refresh_llm()