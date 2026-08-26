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
    """Run the complete monthly production pipeline."""

    try:
        settings = Settings()
        pipeline = MonthlyPipeline(settings)

        result = pipeline.run()

        manifest = result["manifest"]
        evaluation = result.get("evaluation")
        alerts = result.get("alerts")

        if not manifest.published:
            logger.error(
                "Forecast run %s was not published: %s",
                manifest.run_id,
                getattr(
                    manifest,
                    "publish_note",
                    "Publication gate failed.",
                ),
            )
            return 1

        logger.info(
            "Monthly pipeline completed successfully | "
            "run_id=%s | output=%s",
            manifest.run_id,
            manifest.output_path,
        )

        if evaluation is not None:
            logger.info(
                "Previous forecast evaluation | WAPE=%.2f%%",
                evaluation.overall_wape_pct,
            )

        if alerts is not None:
            logger.info(
                "Alerts | total=%d | critical=%d | warning=%d",
                alerts.total_alerts,
                alerts.critical_count,
                alerts.warning_count,
            )

        return 0

    except Exception:
        logger.exception("Monthly production pipeline failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())