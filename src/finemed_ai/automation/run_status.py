"""
finemed_ai.automation.run_status
===================================
Tracks the status of the full monthly chain (ETL -> demand prep -> forecast)
so the upload endpoint can return immediately (the chain takes minutes) and
the frontend can poll progress instead of holding one HTTP request open the
whole time.

Deliberately file-based, not in-memory: an in-memory dict would be lost on
server restart mid-run, and would break the moment you run more than one
API worker process. A small JSON file is simple, durable, and good enough
at this scale -- no need for Redis/a real job queue for a single client's
monthly job.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

_LOCK = threading.Lock()


class RunStatus:
    STAGES = ["queued", "etl", "demand_prep", "forecasting", "done", "failed"]

    def __init__(self, status_file: Path):
        self.status_file = status_file
        self.status_file.parent.mkdir(parents=True, exist_ok=True)

    def _write(self, data: dict) -> None:
        with _LOCK:
            self.status_file.write_text(json.dumps(data, indent=2, default=str))

    def start(self, run_id: str, month: str) -> None:
        self._write({
            "run_id": run_id,
            "month": month,
            "stage": "queued",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "error": None,
        })

    def update(self, stage: str) -> None:
        data = self.read()
        if data is None:
            return
        data["stage"] = stage
        data["updated_at"] = datetime.now(timezone.utc).isoformat()
        self._write(data)

    def fail(self, error: str) -> None:
        data = self.read() or {}
        data["stage"] = "failed"
        data["error"] = error
        data["updated_at"] = datetime.now(timezone.utc).isoformat()
        self._write(data)

    def read(self) -> Optional[dict]:
        if not self.status_file.exists():
            return None
        try:
            return json.loads(self.status_file.read_text())
        except (json.JSONDecodeError, OSError):
            return None
