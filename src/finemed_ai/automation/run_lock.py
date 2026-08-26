from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path


class PipelineAlreadyRunningError(RuntimeError):
    """Raised when another pipeline execution already holds the lock."""


class PipelineRunLock:
    """
    Cross-process filesystem lock for production pipeline execution.

    The lock is acquired atomically using exclusive file creation.

    A stale lock may be removed when its age exceeds stale_after_seconds.
    """

    def __init__(self,lock_path: Path,stale_after_seconds: int = 60 * 60 * 6,) -> None:
        self.lock_path = Path(lock_path)
        self.stale_after_seconds = stale_after_seconds
        self._acquired = False

    def _is_stale(self) -> bool:
        if not self.lock_path.exists():
            return False

        age_seconds = (
            datetime.now(timezone.utc).timestamp()
            - self.lock_path.stat().st_mtime
        )

        return age_seconds > self.stale_after_seconds

    def _remove_stale_lock(self) -> None:
        if self._is_stale():
            try:
                self.lock_path.unlink()
            except FileNotFoundError:
                pass

    def acquire(self) -> None:
        """
        Acquire the lock atomically.

        Raises:
            PipelineAlreadyRunningError:
                If another active process already holds the lock.
        """

        self.lock_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._remove_stale_lock()

        payload = {
            "pid": os.getpid(),
            "started_at": datetime.now(
                timezone.utc
            ).isoformat(),
        }

        try:
            with self.lock_path.open(
                "x",
                encoding="utf-8",
            ) as file:
                json.dump(
                    payload,
                    file,
                    indent=2,
                )

        except FileExistsError as exc:
            raise PipelineAlreadyRunningError(
                "Another production pipeline run is already active. "
                f"Lock file: {self.lock_path}"
            ) from exc

        self._acquired = True

    def release(self) -> None:
        """Release the lock if this instance acquired it."""

        if not self._acquired:
            return

        try:
            self.lock_path.unlink()
        except FileNotFoundError:
            pass
        finally:
            self._acquired = False

    def __enter__(self) -> "PipelineRunLock":
        self.acquire()
        return self

    def __exit__(self, exc_type,exc_value,traceback) -> None:
        self.release()