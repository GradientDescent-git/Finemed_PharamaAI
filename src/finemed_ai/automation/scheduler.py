from __future__ import annotations

import sys
import time
from threading import Lock

import schedule

from finemed_ai.config.settings import Settings
from finemed_ai.utils.logger import get_logger
from finemed_ai.automation.monthly_pipeline import MonthlyPipeline


logger = get_logger(__name__)


class PipelineScheduler:
    """
    Scheduler for the monthly production forecasting pipeline.

    Prevents overlapping executions within the running scheduler process.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.pipeline = MonthlyPipeline(settings)

        self._run_lock = Lock()
        self._jobs_registered = False

        logger.info("Scheduler initialized.")

    def run_monthly_job(self) -> None:
        """
        Execute the monthly production pipeline safely.

        If a previous run is still active, the new run is skipped.
        """

        if not self._run_lock.acquire(blocking=False):
            logger.warning(
                "Monthly job skipped because another pipeline run "
                "is already in progress."
            )
            return

        try:
            logger.info("=" * 80)
            logger.info("Monthly Job Started")
            logger.info("=" * 80)

            result = self.pipeline.run()

            manifest = result["manifest"]

            if manifest.published:
                logger.info(
                    "Monthly Job Finished Successfully | run_id=%s",
                    manifest.run_id,
                )
            else:
                logger.error(
                    "Monthly Job Completed But Publication Failed | "
                    "run_id=%s | reason=%s",
                    manifest.run_id,
                    manifest.publish_note,
                )

        except Exception:
            logger.exception(
                "Monthly Job Failed With An Unexpected Error"
            )

        finally:
            self._run_lock.release()

            logger.info("=" * 80)

    def register_jobs(self) -> None:
        """
        Register scheduled jobs exactly once.
        """

        if self._jobs_registered:
            logger.warning(
                "Scheduler jobs already registered. "
                "Skipping duplicate registration."
            )
            return

        schedule.every().month.at("00:30").do(
            self.run_monthly_job
        )

        self._jobs_registered = True

        logger.info(
            "Monthly schedule registered for 00:30."
        )

    def start(self) -> None:
        """
        Start the scheduler loop.
        """

        self.register_jobs()

        logger.info("Scheduler running.")

        try:
            while True:
                schedule.run_pending()
                time.sleep(30)

        except KeyboardInterrupt:
            logger.info("Scheduler stopped by user.")


def main() -> None:
    settings = Settings()

    scheduler = PipelineScheduler(settings)
    scheduler.start()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception(
            "Scheduler terminated due to an unexpected error."
        )
        sys.exit(1)