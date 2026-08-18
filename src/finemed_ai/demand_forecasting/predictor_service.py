from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import List, Optional

import pandas as pd

from finemed_ai.demand_forecasting.config import DEFAULT_CONFIG, ForecastConfig
from finemed_ai.demand_forecasting.schemas import (
    ForecastDayResult,
    MedicineForecastResult,
    QuantileForecast,
)

logger = logging.getLogger(__name__)


class InsufficientHistoryError(ValueError):
    """Raised when a medicine has no usable history to forecast from."""


class PredictorService:
    """
    Singleton-style Chronos-2 predictor.

    The service:
        1. Validates the incoming dataframe schema.
        2. Filters one medicine.
        3. Normalizes timestamps and target values.
        4. Aggregates duplicate dates.
        5. Calendarizes the series to a complete daily frequency.
        6. Uses the validated production context length.
        7. Runs Chronos-2 P50 forecasting.
    """

    _instance: Optional["PredictorService"] = None

    def __init__(
        self,
        config: ForecastConfig = DEFAULT_CONFIG,
        device: Optional[str] = None,
    ):
        import torch

        self.config = config
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        logger.info(
            "Loading Chronos-2 (%s) on device=%s",
            config.model_id,
            self.device,
        )

        t0 = datetime.now()

        from chronos import Chronos2Pipeline

        self.pipeline = Chronos2Pipeline.from_pretrained(
            config.model_id,
            device_map=self.device,
            torch_dtype=(
                torch.bfloat16
                if self.device == "cuda"
                else torch.float32
            ),
        )

        logger.info(
            "Chronos-2 loaded in %.1fs (device=%s)",
            (datetime.now() - t0).total_seconds(),
            self.device,
        )

    @classmethod
    def get_instance(
        cls,
        config: ForecastConfig = DEFAULT_CONFIG,
    ) -> "PredictorService":
        """
        Process-wide singleton accessor.
        """
        if cls._instance is None:
            cls._instance = cls(config=config)

        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """For tests only."""
        cls._instance = None

    # ------------------------------------------------------------------
    # Input preparation
    # ------------------------------------------------------------------

    def _validate_input_schema(
        self,
        history_df: pd.DataFrame,
    ) -> None:
        """
        Validate the dataframe contract before any filtering occurs.
        """

        if not isinstance(history_df, pd.DataFrame):
            raise TypeError(
                "history_df must be a pandas DataFrame."
            )

        required = {
            self.config.id_column,
            self.config.timestamp_column,
            self.config.target_column,
        }

        missing = required - set(history_df.columns)

        if missing:
            raise ValueError(
                "History dataframe is missing required columns: "
                f"{sorted(missing)}. "
                f"Expected columns: {sorted(required)}"
            )

        if history_df.empty:
            raise InsufficientHistoryError(
                "History dataframe is empty."
            )

    def _prepare_medicine_history(
        self,
        item_id: str,
        history_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Prepare one medicine as a complete daily time series.

        Input may be sparse. Missing calendar days are interpreted as
        zero demand.

        Duplicate observations on the same date are summed.
        """

        cfg = self.config
        item_id = str(item_id)

        self._validate_input_schema(history_df)

        history = history_df[
            history_df[cfg.id_column].astype(str) == item_id
        ].copy()

        if history.empty:
            raise InsufficientHistoryError(
                f"No history for item_id={item_id}"
            )

        history[cfg.id_column] = history[cfg.id_column].astype(str)

        history[cfg.timestamp_column] = pd.to_datetime(
            history[cfg.timestamp_column],
            errors="coerce",
        )

        history[cfg.target_column] = pd.to_numeric(
            history[cfg.target_column],
            errors="coerce",
        )

        invalid_dates = history[cfg.timestamp_column].isna()

        if invalid_dates.any():
            raise ValueError(
                f"item_id={item_id} contains "
                f"{int(invalid_dates.sum())} invalid timestamps."
            )

        invalid_targets = history[cfg.target_column].isna()

        if invalid_targets.any():
            raise ValueError(
                f"item_id={item_id} contains "
                f"{int(invalid_targets.sum())} non-numeric demand values."
            )

        if (history[cfg.target_column] < 0).any():
            raise ValueError(
                f"item_id={item_id} contains negative demand values."
            )

        history = history.sort_values(
            cfg.timestamp_column
        )

        # Normalize to daily timestamps.
        history[cfg.timestamp_column] = (
            history[cfg.timestamp_column]
            .dt.normalize()
        )

        # Multiple ERP rows on the same day are valid.
        # Aggregate them into one daily demand observation.
        history = (
            history.groupby(
                cfg.timestamp_column,
                as_index=False,
            )[cfg.target_column]
            .sum()
        )

        if history.empty:
            raise InsufficientHistoryError(
                f"item_id={item_id} has no usable observations."
            )

        start_date = history[cfg.timestamp_column].min()
        end_date = history[cfg.timestamp_column].max()

        complete_dates = pd.date_range(
            start=start_date,
            end=end_date,
            freq="D",
        )

        history = (
            history.set_index(cfg.timestamp_column)
            .reindex(complete_dates)
            .rename_axis(cfg.timestamp_column)
            .reset_index()
        )

        history[cfg.target_column] = (
            pd.to_numeric(
                history[cfg.target_column],
                errors="coerce",
            )
            .fillna(0.0)
        )

        history[cfg.id_column] = item_id

        history = history[
            [
                cfg.id_column,
                cfg.timestamp_column,
                cfg.target_column,
            ]
        ]

        return history

    # ------------------------------------------------------------------
    # Core forecasting
    # ------------------------------------------------------------------

    def forecast_medicine(
        self,
        item_id: str,
        history_df: pd.DataFrame,
    ) -> MedicineForecastResult:
        """
        Forecast the next prediction_length days for one medicine.

        The incoming data does not have to already be calendarized;
        this service guarantees a complete daily series before passing
        it to Chronos-2.
        """

        cfg = self.config
        item_id = str(item_id)

        history = self._prepare_medicine_history(
            item_id=item_id,
            history_df=history_df,
        )

        # Use only the validated production context.
        history = history.tail(cfg.context_length).copy()

        actual_context = len(history)

        logger.info(
            "CHRONOS INPUT | item=%s | configured_context=%d | "
            "actual_rows=%d | start=%s | end=%s",
            item_id,
            cfg.context_length,
            actual_context,
            history[cfg.timestamp_column].min(),
            history[cfg.timestamp_column].max(),
        )

        # Chronos needs enough observations to infer frequency.
        min_observations = 3

        if actual_context < min_observations:
            raise InsufficientHistoryError(
                f"item_id={item_id} has only {actual_context} "
                f"daily observation(s); minimum {min_observations} "
                "required for Chronos-2."
            )

        if actual_context < cfg.prediction_length:
            logger.warning(
                "item_id=%s has only %d days of context "
                "(prediction_length=%d). Forecast quality at this "
                "short context is not validated.",
                item_id,
                actual_context,
                cfg.prediction_length,
            )

        raw = self.pipeline.predict_df(
            history,
            prediction_length=cfg.prediction_length,
            quantile_levels=list(cfg.quantile_levels),
            id_column=cfg.id_column,
            timestamp_column=cfg.timestamp_column,
            target=cfg.target_column,
        )

        return self._to_result(
            item_id=item_id,
            context_used=actual_context,
            raw_forecast=raw,
        )

    def forecast_batch(
        self,
        history_df: pd.DataFrame,
        item_ids: Optional[List[str]] = None,
    ) -> tuple[List[MedicineForecastResult], List[str]]:
        """
        Forecast many medicines.

        Failures are isolated per medicine.
        """

        self._validate_input_schema(history_df)

        cfg = self.config

        ids = (
            item_ids
            if item_ids is not None
            else sorted(
                history_df[cfg.id_column]
                .astype(str)
                .unique()
            )
        )

        results: List[MedicineForecastResult] = []
        failed: List[str] = []

        for item_id in ids:
            item_id = str(item_id)

            try:
                results.append(
                    self.forecast_medicine(
                        item_id,
                        history_df,
                    )
                )

            except InsufficientHistoryError as exc:
                logger.warning(
                    "Skipping item_id=%s: %s",
                    item_id,
                    exc,
                )
                failed.append(item_id)

            except Exception:
                logger.exception(
                    "Forecast failed for item_id=%s",
                    item_id,
                )
                failed.append(item_id)

        return results, failed

    # ------------------------------------------------------------------
    # Internal result conversion
    # ------------------------------------------------------------------

    def _to_result(
        self,
        item_id: str,
        context_used: int,
        raw_forecast: pd.DataFrame,
    ) -> MedicineForecastResult:

        cfg = self.config

        days: List[ForecastDayResult] = []

        expected_quantile_columns = [
            "0.1",
            "0.2",
            "0.3",
            "0.4",
            "0.5",
            "0.6",
            "0.7",
            "0.8",
            "0.9",
        ]

        missing_quantiles = [
            column
            for column in expected_quantile_columns
            if column not in raw_forecast.columns
        ]

        if missing_quantiles:
            raise ValueError(
                "Chronos output is missing expected quantile columns: "
                f"{missing_quantiles}"
            )

        for _, row in raw_forecast.iterrows():

            raw_values = [
                max(float(row[column]), 0.0)
                for column in expected_quantile_columns
            ]

            # Enforce monotonic quantiles defensively.
            monotonic_values = []
            running_max = 0.0
            corrected = False

            for value in raw_values:

                if value < running_max:
                    corrected = True
                    value = running_max

                running_max = value
                monotonic_values.append(value)

            if corrected:
                logger.warning(
                    "Non-monotonic quantiles detected for item_id=%s "
                    "on %s. Corrected using cumulative maximum. "
                    "Raw=%s",
                    item_id,
                    row[cfg.timestamp_column],
                    raw_values,
                )

            quantiles = QuantileForecast(
                p10=monotonic_values[0],
                p20=monotonic_values[1],
                p30=monotonic_values[2],
                p40=monotonic_values[3],
                p50=monotonic_values[4],
                p60=monotonic_values[5],
                p70=monotonic_values[6],
                p80=monotonic_values[7],
                p90=monotonic_values[8],
            )

            point_quantile_name = (
                f"p{int(cfg.point_quantile * 100)}"
            )

            point = getattr(
                quantiles,
                point_quantile_name,
            )

            days.append(
                ForecastDayResult(
                    forecast_date=pd.Timestamp(
                        row[cfg.timestamp_column]
                    ).date(),
                    predicted_demand=round(
                        point,
                        2,
                    ),
                    quantiles=quantiles,
                )
            )

        return MedicineForecastResult(
            medicine_id=item_id,
            generated_at=datetime.now(timezone.utc),
            context_length_used=context_used,
            prediction_length=cfg.prediction_length,
            model_id=cfg.model_id,
            days=days,
        )