from __future__ import annotations

import logging
from pathlib import Path
import pandas as pd
from finemed_ai.config.settings import Settings

logger = logging.getLogger(__name__)


class PerformanceHistoryTracker:
    """
    Persists historical ML model evaluation metrics over time.

    Enables tracking of model degradation, routing history, and accuracy trends.
    """

    def __init__(
        self,
        history_file: str | Path = Settings.PERFORMANCE_HISTORY_FILE,
    ) -> None:
        self.history_file = Path(history_file)

    def record_run(
        self,
        run_id: str,
        wape: float,
        bias: float,
        mae: float,
        selected_models_summary: dict[str, int],
        status: str = "PASSED",
    ) -> None:
        """
        Append a new forecast run performance entry to history.parquet.
        """
        self.history_file.parent.mkdir(parents=True, exist_ok=True)

        new_entry = {
            "Run_ID": run_id,
            "Recorded_At": pd.Timestamp.now(),
            "WAPE": wape,
            "Bias": bias,
            "MAE": mae,
            "TSB_Count": selected_models_summary.get("tsb", 0),
            "Chronos_Count": selected_models_summary.get("chronos-2", 0),
            "Status": status,
        }

        new_df = pd.DataFrame([new_entry])

        if self.history_file.exists() and self.history_file.stat().st_size > 0:
            try:
                existing_df = pd.read_parquet(self.history_file)
                combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            except Exception as exc:
                logger.warning("Could not read existing performance history: %s. Overwriting.", exc)
                combined_df = new_df
        else:
            combined_df = new_df

        combined_df.to_parquet(self.history_file, index=False)
        logger.info("Recorded performance history for run %s to %s", run_id, self.history_file)

    def get_history(self) -> pd.DataFrame:
        """Read full performance history table."""
        if not self.history_file.exists() or self.history_file.stat().st_size == 0:
            return pd.DataFrame()
        try:
            return pd.read_parquet(self.history_file)
        except Exception:
            return pd.DataFrame()
