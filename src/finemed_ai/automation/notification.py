from __future__ import annotations

from finemed_ai.utils.logger import get_logger

logger = get_logger(__name__)


class NotificationService:
    def __init__(self):
        logger.info("Notification Service Initialized.")

    def notify_success(self,message: str) -> None:
        logger.info("[SUCCESS] %s",message)

    def notify_failure(self,message: str) -> None:
        logger.error("[FAILED] %s",message)

    def notify_warning(self,message: str) -> None:
        logger.warning("[WARNING] %s",message)

    def notify_info(self,message: str) -> None:
        logger.info("[INFO] %s",message)

        