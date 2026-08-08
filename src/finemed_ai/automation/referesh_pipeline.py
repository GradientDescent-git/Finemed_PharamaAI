from __future__ import annotations

from finemed_ai.config.settings import Settings
from finemed_ai.utils.logger import get_logger

from finemed_ai.automation.monthly_pipeline import (
    MonthlyPipeline,
)

logger = get_logger(__name__)

class RefreshPipeline:
    def __init__(self,settings: Settings) -> None:
        self.settings = settings
        self.pipeline = MonthlyPipeline(settings)

    def refresh(self) -> None:
        logger.info("=" * 80)

        logger.info(
            "Starting Refresh Pipeline"
        )

        logger.info("=" * 80)

        self.pipeline.run()

        logger.info("=" * 80)

        logger.info(
            "Refresh Completed Successfully"
        )

        logger.info("=" * 80)

if __name__ == "__main__":

    settings = Settings()

    pipeline = RefreshPipeline(
        settings
    )

    pipeline.refresh()