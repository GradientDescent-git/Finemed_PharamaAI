from __future__ import annotations
 
from pathlib import Path
 
from finemed_ai.config.settings import Settings
from finemed_ai.utils.logger import get_logger
 
from finemed_ai.pipeline.run_pipeline import run_pipeline
 
from finemed_ai.demand_forecasting.config import ForecastConfig
from finemed_ai.demand_forecasting.pipeline import run_monthly_forecast
 
logger = get_logger(__name__)
 
 
class MonthlyPipeline:
    def __init__(self, settings: Settings):
        self.settings = settings
 
        # Translate Settings (UPPER_CASE, project-wide config) into the
        # ForecastConfig the forecasting module actually expects
        # (lower_case, forecasting-specific). Keeping these as two separate
        # classes is intentional -- Settings covers the whole project,
        # ForecastConfig is the frozen, backtested forecasting config -- but
        # they need to agree at this one call site.
        self.forecast_config = ForecastConfig(
            model_id=settings.CHRONOS_MODEL_NAME,
            context_length=settings.CONTEXT_LENGTH,
            prediction_length=settings.PREDICTION_LENGTH,
            quantile_levels=tuple(settings.QUANTILES),
        )
 
    def run_etl_pipeline(self) -> None:
        logger.info("Running ETL Pipeline...")
        run_pipeline()
        logger.info("ETL Pipeline Completed.")
 
    def prepare_demand_data(self) -> None:
        """
        Silver-layer demand prep (daily calendar fill, etc.) currently
        happens inside demand_forecasting.pipeline.run_monthly_forecast()
        itself (_load_and_prepare_daily_demand). This method is a hook for
        splitting that out into its own step later -- it is a no-op today,
        logged explicitly so it's never mistaken for having done work.
        """
        logger.info(
            "prepare_demand_data: no separate step yet -- daily-calendar "
            "prep runs inside run_monthly_forecast(). Nothing to do here."
        )
 
    def generate_forecast(self):
        logger.info("Generating Forecast...")
 
        manifest = run_monthly_forecast(
            silver_demand_path=Path(self.settings.DEMAND_FILE),
            output_dir=Path(self.settings.DEMAND_DIR),
            config=self.forecast_config,
        )
 
        logger.info(
            "Forecast Generated: %d/%d medicines succeeded (run_id=%s).",
            manifest.medicines_succeeded, manifest.medicines_requested, manifest.run_id,
        )
        return manifest
 
    def evaluate_forecast(self) -> None:
        """
        Not implemented yet: comparing this run's forecast against actuals
        once they land next month (the WAPE/MAE backtest you already did in
        the notebook, but automated). Raises loudly instead of pretending.
        """
        raise NotImplementedError(
            "evaluate_forecast() is not implemented yet. The notebook's "
            "backtest methodology (WAPE/MAE/SMAPE vs actuals) needs to be "
            "ported into an automated module before this can run monthly. "
            "Skip this step for now by not calling it from run()."
        )
 
    def refresh_llm(self) -> None:
        """
        No-op by design: the LLM layer (finemed_ai.llm) reads forecasts via
        ForecastStore, which lazily reloads whenever the underlying parquet
        file changes (ForecastStore.is_stale() / reload()). There is
        nothing to push -- the next /chat request just sees the new data.
        """
        logger.info("refresh_llm: no-op -- ForecastStore reloads lazily on next read.")
 
    def run(self):
        self.run_etl_pipeline()
        self.prepare_demand_data()
        manifest = self.generate_forecast()
        self.refresh_llm()
        return manifest
 