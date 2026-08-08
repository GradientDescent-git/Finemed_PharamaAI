from __future__ import annotations

import time
import schedule

from finemed_ai.config.settings import Settings
from finemed_ai.utils.logger import get_logger

from finemed_ai.automation.monthly_pipeline import (
    MonthlyPipeline,
)

logger = get_logger(__name__)

class PipelineScheduler:
    def __init__(self,settings: Settings):
        self.settings = settings
        self.pipeline = MonthlyPipeline(settings)
        logger.info("Scheduler Initialized.")

    def run_monthly_job(self) -> None:
        logger.info("=" * 80)
        logger.info("Monthly Job Started")
        logger.info("=" * 80)
        self.pipeline.run()
        logger.info("=" * 80)
        logger.info("Monthly Job Finished")
        logger.info( "=" * 80)

    def register_jobs(self,) -> None:
        """
        Register scheduled jobs.
        """
        schedule.every().month.at("00:30").do(self.run_monthly_job)
        logger.info("Monthly schedule registered.")

    def start(self) -> None:
        self.register_jobs()
        logger.info("Scheduler Running...")

        while True:
            schedule.run_pending()
            time.sleep(30)

if __name__ == "__main__":

    settings = Settings()

    scheduler = PipelineScheduler(settings)

    scheduler.start()