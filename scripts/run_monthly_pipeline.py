from __future__ import annotations

import logging

from finemed_ai.automation.monthly_pipeline import MonthlyPipeline
from finemed_ai.config.settings import Settings


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)

logger = logging.getLogger("run_monthly_pipeline")


def main() -> int:
    """
    Run the complete monthly production pipeline.

    Execution order:

        1. ETL pipeline
        2. Demand data preparation
        3. Previous forecast evaluation
        4. Production forecast generation
        5. Operational alert generation
        6. LLM refresh
    """

    try:
        settings = Settings()
        pipeline = MonthlyPipeline(settings)

        result = pipeline.run()

        manifest = result["manifest"]
        evaluation = result.get("evaluation")
        alerts = result.get("alerts")

    except Exception:
        logger.exception("Monthly production pipeline failed")
        return 1

    logger.info(
        "Monthly pipeline completed | run_id=%s | published=%s | output=%s",
        manifest.run_id,
        manifest.published,
        manifest.output_path,
    )

    if evaluation is not None:
        logger.info(
            "Previous forecast evaluation | WAPE=%.2f%%",
            evaluation.overall_wape_pct,
        )

    if alerts is not None:
        logger.info(
            "Operational alerts | total=%d | critical=%d | warning=%d",
            alerts.total_alerts,
            alerts.critical_count,
            alerts.warning_count,
        )

    if not manifest.published:
        logger.error(
            "Forecast publication failed | reason=%s",
            getattr(
                manifest,
                "publish_note",
                "Unknown publication failure",
            ),
        )
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())