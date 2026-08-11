from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from finemed_ai.config.settings import Settings
from finemed_ai.utils.logger import get_logger

from finemed_ai.pipeline.run_pipeline import run_pipeline

from finemed_ai.demand_forecasting.config import ForecastConfig
from finemed_ai.demand_forecasting.pipeline import run_monthly_forecast
from finemed_ai.demand_forecasting.data_preparation import prepare_demand_data as _prepare_demand_data
from finemed_ai.demand_forecasting.evaluation import ForecastEvaluator, OverallEvaluation
from finemed_ai.automation.alert_engine import AlertEngine, AlertStore

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
        Bridges the warehouse (Postgres fact_sales_line/fact_sales_invoice)
        into the flat daily_demand.parquet the forecasting module reads.
        Real logic lives in demand_forecasting.data_preparation -- this is
        no longer a no-op.
        """
        logger.info("Preparing demand data from warehouse...")
        output_path = _prepare_demand_data()
        logger.info("Demand data written to %s", output_path)

    def evaluate_forecast(self) -> Optional[OverallEvaluation]:
        """
        Evaluates the CURRENT latest.parquet -- i.e. LAST month's forecast --
        against the actuals that prepare_demand_data() just refreshed.
        MUST run before generate_forecast() overwrites latest.parquet (see
        run()'s call order below) -- there's nothing left to evaluate
        against fresh actuals once the new forecast replaces the old one.
        """
        latest_forecast_path = Path(self.settings.DEMAND_DIR) / "latest.parquet"
        if not latest_forecast_path.exists():
            logger.info("No prior forecast exists yet (first run) -- nothing to evaluate.")
            return None

        actuals_df = pd.read_parquet(self.settings.DEMAND_FILE)
        forecast_df = pd.read_parquet(latest_forecast_path)

        evaluator = ForecastEvaluator(Path(self.settings.DEMAND_DIR) / "evaluations")
        result = evaluator.evaluate(actuals_df, forecast_df)

        if result.total_medicines_evaluated == 0:
            logger.info(
                "Evaluation found no overlapping actuals yet -- too early to "
                "evaluate this forecast (expected right after the first run)."
            )
        else:
            logger.info(
                "Forecast evaluation: %d medicines, WAPE=%.2f%%, SMAPE=%.2f%%, "
                "coverage=%.1f%% (P10-P90 calibration).",
                result.total_medicines_evaluated, result.overall_wape_pct,
                result.overall_smape_pct, result.overall_coverage_pct,
            )
        return result

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

    def run_alerts(self, manifest) -> AlertStore:
        """
        Scans the freshly generated forecast for operational risks.
        NOTE: stockout-risk alerts will not fire yet -- no real stock-on-hand
        data currently flows into the warehouse (the existing "inventory"
        transform is derived from purchase history, not live stock levels).
        Demand-spike and high-uncertainty alerts DO work today since they
        only need the forecast and historical demand, both of which exist.
        """
        forecast_df = pd.read_parquet(manifest.output_path)
        actuals_df = pd.read_parquet(self.settings.DEMAND_FILE)

        engine = AlertEngine(Path(self.settings.DEMAND_DIR) / "alerts")
        store = engine.scan_forecasts(forecast_df, historical_demand_df=actuals_df)

        logger.info(
            "Alert scan complete: %d alerts (%d critical, %d warning). "
            "Stockout-risk alerts require inventory data not yet available.",
            store.total_alerts, store.critical_count, store.warning_count,
        )
        return store

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
        self.evaluate_forecast()  # evaluate PRIOR forecast against fresh actuals, before overwrite
        manifest = self.generate_forecast()
        self.run_alerts(manifest)
        self.refresh_llm()
        return manifest
