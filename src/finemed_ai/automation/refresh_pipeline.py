from __future__ import annotations

import sys

from finemed_ai.config.settings import Settings
from finemed_ai.utils.logger import get_logger
from finemed_ai.automation.monthly_pipeline import MonthlyPipeline


logger = get_logger(__name__)


class RefreshPipeline:
    """
    Manual/on-demand entry point for the complete monthly pipeline.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.pipeline = MonthlyPipeline(settings)

    def refresh(self) -> dict:
        logger.info("=" * 80)
        logger.info("Starting Refresh Pipeline")
        logger.info("=" * 80)

        try:
            result = self.pipeline.run()

            manifest = result["manifest"]

            if not manifest.published:
                logger.error(
                    "Refresh pipeline completed, but forecast publication "
                    "failed | run_id=%s | reason=%s",
                    manifest.run_id,
                    manifest.publish_note,
                )
                raise RuntimeError(
                    f"Forecast publication failed: "
                    f"{manifest.publish_note}"
                )

            logger.info("=" * 80)
            logger.info(
                "Refresh Completed Successfully | run_id=%s",
                manifest.run_id,
            )
            logger.info("=" * 80)

            return result

        except Exception:
            logger.exception("Refresh Pipeline Failed")
            raise


def main() -> None:
    settings = Settings()

    pipeline = RefreshPipeline(settings)
    pipeline.refresh()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        sys.exit(1)