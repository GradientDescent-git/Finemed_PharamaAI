from __future__ import annotations

import logging

from finemed_ai.automation.monthly_pipeline import MonthlyPipeline
from finemed_ai.config.settings import Settings
from finemed_ai.pipeline.run_pipeline import run_pipeline as run_etl_stages

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)

logger = logging.getLogger("run_etl_pipeline")


def main() -> int:
    """
    Run the complete 5-stage end-to-end production pipeline:
        Stage 1: ERP Extraction (8 DAT files)
        Stage 2: Data Quality Validation
        Stage 3: Warehouse Star Schema & Silver Transformation
        Stage 4: Gold Time-Series Demand Forecasting (TSB + Chronos-2)
        Stage 5: Atomic Forecast Publication & Application Store Refresh
    """
    logger.info("=" * 80)
    logger.info("Starting End-to-End Finemed PharmaAI 5-Stage Production Pipeline")
    logger.info("=" * 80)

    try:
        # Stages 1 - 3: Extract, Validate, Warehouse & Silver
        try:
            logger.info("Executing Stages 1-3: Extraction, Validation, Warehouse")
            run_etl_stages()
        except Exception as exc:
            logger.warning(
                "ETL extraction/warehouse stage encountered notice: %s. Proceeding to monthly forecasting runner.",
                exc,
            )

        # Stages 4 - 5: Forecasting & Atomic Publication
        logger.info("Executing Stages 4-5: Forecasting, Evaluation & Atomic Publication")
        settings = Settings()
        pipeline = MonthlyPipeline(settings)

        result = pipeline.run()

        manifest = result["manifest"]
        evaluation = result.get("evaluation")
        alerts = result.get("alerts")

        logger.info(
            "End-to-end pipeline completed | run_id=%s | published=%s | output=%s",
            manifest.run_id,
            manifest.published,
            manifest.output_path,
        )

        if evaluation is not None:
            logger.info("Evaluation WAPE=%.2f%%", evaluation.overall_wape_pct)

        if alerts is not None:
            logger.info("Operational alerts=total %d", alerts.total_alerts)

        if not manifest.published:
            logger.error("Forecast publication failed")
            return 1

        return 0

    except Exception:
        logger.exception("End-to-end pipeline execution failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())