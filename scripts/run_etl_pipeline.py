from __future__ import annotations

import logging

from finemed_ai.automation.monthly_pipeline import MonthlyPipeline
from finemed_ai.config.settings import Settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)

logger = logging.getLogger("run_etl_pipeline")


def main() -> int:
    """
    Run the end-to-end 5-stage production pipeline:
    Extraction (8 DAT files) -> Data Quality Validation -> Warehouse & Silver -> Gold Demand Forecasting -> Atomic Publication.
    """
    logger.info("Starting end-to-end Finemed PharmaAI production pipeline")

    try:
        settings = Settings()
        pipeline = MonthlyPipeline(settings)

        result = pipeline.run()

        manifest = result["manifest"]
        evaluation = result.get("evaluation")
        alerts = result.get("alerts")

        logger.info(
            "Production pipeline completed | run_id=%s | published=%s | output=%s",
            manifest.run_id,
            manifest.published,
            manifest.output_path,
        )

        if evaluation is not None:
            logger.info("Evaluation metrics | WAPE=%.2f%%", evaluation.overall_wape_pct)

        if alerts is not None:
            logger.info("Operational alerts count | total=%d", alerts.total_alerts)

        if not manifest.published:
            logger.error("Forecast publication failed")
            return 1

        return 0

    except Exception:
        logger.exception("Pipeline execution failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())