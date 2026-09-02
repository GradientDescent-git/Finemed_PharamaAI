import json
import os
import smtplib
import urllib.request
import urllib.error
from email.mime.text import MIMEText
from typing import Any, Dict, Optional

from finemed_ai.utils.logger import get_logger

logger = get_logger(__name__)


class NotificationService:
    """
    Multi-channel Notification Service for FineMed Pharma AI.

    Transports supported:
        - Structured Logger (always active)
        - HTTP Webhook (Slack / Teams / Custom Webhook endpoint)
        - SMTP Email (TLS / SSL support)
    """

    def __init__(
        self,
        webhook_url: Optional[str] = None,
        smtp_host: Optional[str] = None,
        smtp_port: Optional[int] = None,
        smtp_user: Optional[str] = None,
        smtp_password: Optional[str] = None,
        email_from: Optional[str] = None,
        email_to: Optional[str] = None,
    ) -> None:
        self.webhook_url = webhook_url or os.environ.get("NOTIFICATION_WEBHOOK_URL", "").strip()
        self.smtp_host = smtp_host or os.environ.get("NOTIFICATION_SMTP_HOST", "").strip()
        self.smtp_port = smtp_port or int(os.environ.get("NOTIFICATION_SMTP_PORT", "587"))
        self.smtp_user = smtp_user or os.environ.get("NOTIFICATION_SMTP_USER", "").strip()
        self.smtp_password = smtp_password or os.environ.get("NOTIFICATION_SMTP_PASSWORD", "").strip()
        self.email_from = email_from or os.environ.get("NOTIFICATION_EMAIL_FROM", "alerts@finemed.ai").strip()
        self.email_to = email_to or os.environ.get("NOTIFICATION_EMAIL_TO", "").strip()

        logger.info(
            "NotificationService initialized | webhook=%s | smtp=%s | email_to=%s",
            bool(self.webhook_url),
            bool(self.smtp_host),
            bool(self.email_to),
        )

    def send_alert(
        self,
        subject: str,
        message: str,
        level: str = "ERROR",
        details: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, bool]:
        """
        Dispatch an alert across all active notification transports.

        Returns a dictionary of transport delivery statuses.
        """
        results = {"logger": True, "webhook": False, "email": False}

        # 1. Log locally
        if level.upper() in ("CRITICAL", "ERROR", "FAILED"):
            logger.error("[%s] %s: %s | details=%s", level.upper(), subject, message, details or {})
        elif level.upper() in ("WARNING", "WARN"):
            logger.warning("[%s] %s: %s", level.upper(), subject, message)
        else:
            logger.info("[%s] %s: %s", level.upper(), subject, message)

        # 2. HTTP Webhook dispatch
        if self.webhook_url:
            results["webhook"] = self._send_webhook(subject, message, level, details)

        # 3. SMTP Email dispatch
        if self.smtp_host and self.email_to:
            results["email"] = self._send_email(subject, message, level, details)

        return results

    def _send_webhook(
        self,
        subject: str,
        message: str,
        level: str,
        details: Optional[Dict[str, Any]],
    ) -> bool:
        """
        Send alert via HTTP POST webhook.
        """
        payload = {
            "text": f"*{level.upper()} Alert*: {subject}\n{message}",
            "subject": subject,
            "level": level,
            "message": message,
            "details": details or {},
            "service": "FineMed Pharma AI",
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.webhook_url,
            data=data,
            headers={"Content-Type": "application/json", "User-Agent": "FineMedAI-AlertService/1.0"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                if resp.status in (200, 201, 202, 204):
                    logger.info("Webhook alert dispatched successfully to %s", self.webhook_url)
                    return True
                logger.warning("Webhook alert failed with HTTP status %d", resp.status)
                return False
        except Exception as exc:
            logger.exception("Failed to dispatch webhook alert to %s: %s", self.webhook_url, exc)
            return False

    def _send_email(
        self,
        subject: str,
        message: str,
        level: str,
        details: Optional[Dict[str, Any]],
    ) -> bool:
        """
        Send alert email via SMTP.
        """
        full_subject = f"[{level.upper()}] FineMed AI Alert: {subject}"
        body_text = f"FineMed Pharma AI Operational Alert\n\nLevel: {level}\nSubject: {subject}\n\nMessage:\n{message}\n"
        if details:
            body_text += f"\nDetails:\n{json.dumps(details, indent=2)}\n"

        msg = MIMEText(body_text)
        msg["Subject"] = full_subject
        msg["From"] = self.email_from
        msg["To"] = self.email_to

        try:
            with smtplib.SMTP(self.smtp_host, self.smtp_port, timeout=10) as server:
                server.ehlo()
                if server.has_extn("STARTTLS"):
                    server.starttls()
                if self.smtp_user and self.smtp_password:
                    server.login(self.smtp_user, self.smtp_password)
                server.sendmail(self.email_from, [self.email_to], msg.as_string())
            logger.info("Email alert dispatched successfully to %s", self.email_to)
            return True
        except Exception as exc:
            logger.exception("Failed to send alert email via SMTP to %s: %s", self.email_to, exc)
            return False

    def notify_success(self, message: str) -> None:
        self.send_alert(subject="Pipeline Execution Succeeded", message=message, level="INFO")

    def notify_failure(self, message: str) -> None:
        self.send_alert(subject="Pipeline Execution Failed", message=message, level="ERROR")

    def notify_warning(self, message: str) -> None:
        self.send_alert(subject="Pipeline Warning", message=message, level="WARNING")

    def notify_info(self, message: str) -> None:
        self.send_alert(subject="Pipeline Information", message=message, level="INFO")


        