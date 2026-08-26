from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone
from typing import Optional, Sequence
from threading import RLock

import numpy as np
import pandas as pd

from finemed_ai.demand_forecasting.config import (
    DEFAULT_CONFIG,
    ForecastConfig,
)
from finemed_ai.demand_forecasting.schemas import (
    ForecastDayResult,
    MedicineForecastResult,
    QuantileForecast,
)


logger = logging.getLogger(__name__)


# =============================================================================
# ERRORS
# =============================================================================


class InsufficientHistoryError(ValueError):
    """Raised when a medicine does not have sufficient usable history."""


class ForecastValidationError(ValueError):
    """Raised when input or model forecast output violates the data contract."""


# =============================================================================
# PREDICTOR SERVICE
# =============================================================================


class PredictorService:
    """
    Production Chronos-2 forecasting service.

    Responsibilities
    ----------------
    1. Validate the incoming dataframe contract.
    2. Isolate history for one medicine.
    3. Normalize and validate timestamps.
    4. Validate demand values.
    5. Aggregate duplicate daily observations.
    6. Calendarize to a continuous daily time series.
    7. Apply the configured maximum context length.
    8. Execute Chronos-2 inference.
    9. Validate the returned forecast contract.
    10. Enforce non-negative, monotonic quantiles.
    11. Convert the forecast into typed project schemas.
    12. Support isolated batch failures.

    Production assumptions
    ----------------------
    - Chronos-2 model: amazon/chronos-2
    - Daily demand frequency
    - Missing calendar days represent zero observed demand
    - Context length is a maximum window, not a minimum requirement
    - Point forecast is the configured quantile, normally P50
    - Negative forecast values are clamped to zero
    - Quantile crossing is repaired using cumulative maximum
    """

    _instance: Optional["PredictorService"] = None
    _instance_lock = threading.RLock()

    _MIN_OBSERVATIONS = 3

    # -------------------------------------------------------------------------
    # CONSTRUCTION
    # -------------------------------------------------------------------------

    def __init__(
        self,
        config: ForecastConfig = DEFAULT_CONFIG,
        device: Optional[str] = None,
    ) -> None:
        """
        Initialize the Chronos-2 predictor.

        The model is loaded once for each PredictorService instance.
        """

        if config is None:
            raise ValueError("config must not be None.")

        self.config = config

        self._validate_config()

        self.device = self._resolve_device(device)

        self._prediction_lock = RLock()

        logger.info(
            (
                "Loading Chronos-2 | model=%s | device=%s | "
                "context=%d | horizon=%d | point=P%d"
            ),
            self.config.model_id,
            self.device,
            self.config.context_length,
            self.config.prediction_length,
            int(self.config.point_quantile * 100),
        )

        started_at = datetime.now(timezone.utc)

        try:
            import torch
            from chronos import Chronos2Pipeline

        except ImportError as exc:
            raise ImportError(
                "Chronos-2 dependencies are unavailable. "
                "Ensure both PyTorch and chronos are installed "
                "in the active environment."
            ) from exc

        torch_dtype = self._resolve_torch_dtype(
            torch=torch,
            device=self.device,
        )

        try:
            self.pipeline = Chronos2Pipeline.from_pretrained(
                self.config.model_id,
                device_map=self.device,
                torch_dtype=torch_dtype,
            )

        except Exception as exc:
            logger.exception(
                "Failed to load Chronos-2 | model=%s | device=%s",
                self.config.model_id,
                self.device,
            )

            raise RuntimeError(
                "Failed to initialize Chronos-2 pipeline "
                f"for model={self.config.model_id!r} "
                f"on device={self.device!r}."
            ) from exc

        elapsed_seconds = (
            datetime.now(timezone.utc) - started_at
        ).total_seconds()

        logger.info(
            (
                "Chronos-2 loaded successfully | model=%s | "
                "device=%s | elapsed=%.2fs"
            ),
            self.config.model_id,
            self.device,
            elapsed_seconds,
        )

    def _get_prediction_lock(self) -> RLock:
        lock = getattr(self, "_prediction_lock", None)

        if lock is None:
            lock = RLock()
            self._prediction_lock = lock

        return lock

    # -------------------------------------------------------------------------
    # CONFIGURATION VALIDATION
    # -------------------------------------------------------------------------

    def _validate_config(self) -> None:
        """
        Validate ForecastConfig invariants required by this service.
        """

        cfg = self.config

        if not str(cfg.model_id).strip():
            raise ValueError("config.model_id must not be empty.")

        if cfg.context_length < self._MIN_OBSERVATIONS:
            raise ValueError(
                "config.context_length must be at least "
                f"{self._MIN_OBSERVATIONS}."
            )

        if cfg.prediction_length <= 0:
            raise ValueError(
                "config.prediction_length must be greater than zero."
            )

        if not cfg.quantile_levels:
            raise ValueError(
                "config.quantile_levels must not be empty."
            )

        quantiles = tuple(
            float(level)
            for level in cfg.quantile_levels
        )

        if len(set(quantiles)) != len(quantiles):
            raise ValueError(
                "config.quantile_levels must not contain duplicates."
            )

        if tuple(sorted(quantiles)) != quantiles:
            raise ValueError(
                "config.quantile_levels must be strictly sorted "
                "in ascending order."
            )

        if any(
            level <= 0.0 or level >= 1.0
            for level in quantiles
        ):
            raise ValueError(
                "All config.quantile_levels must be strictly "
                "between 0 and 1."
            )

        point_quantile = float(cfg.point_quantile)

        if point_quantile not in quantiles:
            raise ValueError(
                "config.point_quantile must exist in "
                "config.quantile_levels."
            )

        required_schema_fields = {
            "id_column": cfg.id_column,
            "timestamp_column": cfg.timestamp_column,
            "target_column": cfg.target_column,
        }

        for field_name, value in required_schema_fields.items():
            if not str(value).strip():
                raise ValueError(
                    f"config.{field_name} must not be empty."
                )

        if len(
            {
                cfg.id_column,
                cfg.timestamp_column,
                cfg.target_column,
            }
        ) != 3:
            raise ValueError(
                "id_column, timestamp_column and target_column "
                "must be distinct."
            )

        # The current QuantileForecast schema exposes P10 through P90.
        required_schema_quantiles = (
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
        )

        if tuple(quantiles) != required_schema_quantiles:
            raise ValueError(
                "The current QuantileForecast schema requires exactly "
                "quantiles (0.1, 0.2, ..., 0.9). "
                f"Received: {quantiles}"
            )

    # -------------------------------------------------------------------------
    # DEVICE RESOLUTION
    # -------------------------------------------------------------------------

    @staticmethod
    def _resolve_device(
        device: Optional[str],
    ) -> str:
        """
        Resolve the execution device.

        Automatic selection:

            CUDA available -> cuda
            otherwise      -> cpu
        """

        if device is not None:

            normalized = str(device).strip().lower()

            if not normalized:
                raise ValueError(
                    "device must not be empty."
                )

            valid = (
                normalized == "cpu"
                or normalized == "mps"
                or normalized == "cuda"
                or normalized.startswith("cuda:")
            )

            if not valid:
                raise ValueError(
                    f"Unsupported device={device!r}. "
                    "Expected 'cpu', 'mps', 'cuda', or 'cuda:N'."
                )

            return normalized

        try:
            import torch

        except ImportError as exc:
            raise ImportError(
                "PyTorch is required to resolve the execution device."
            ) from exc

        if torch.cuda.is_available():
            return "cuda"

        if (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        ):
            return "mps"

        return "cpu"

    @staticmethod
    def _resolve_torch_dtype(
        torch,
        device: str,
    ):
        """
        Select a safe inference dtype for the selected device.
        """

        if device.startswith("cuda"):
            return torch.bfloat16

        return torch.float32

    # -------------------------------------------------------------------------
    # SINGLETON
    # -------------------------------------------------------------------------

    @classmethod
    def get_instance(
        cls,
        config: ForecastConfig = DEFAULT_CONFIG,
        device: Optional[str] = None,
    ) -> "PredictorService":
        """
        Return the process-wide PredictorService instance.

        The singleton is configuration-safe and thread-safe.
        """

        if config is None:
            raise ValueError("config must not be None.")

        with cls._instance_lock:

            if cls._instance is None:

                cls._instance = cls(
                    config=config,
                    device=device,
                )

                return cls._instance

            existing = cls._instance

            if existing.config != config:
                raise RuntimeError(
                    "PredictorService singleton already exists with a "
                    "different ForecastConfig. Reset the service before "
                    "creating it with another configuration."
                )

            requested_device = (
                cls._resolve_device(device)
                if device is not None
                else existing.device
            )

            if requested_device != existing.device:
                raise RuntimeError(
                    "PredictorService singleton already exists on "
                    f"device={existing.device!r}, but "
                    f"device={requested_device!r} was requested. "
                    "Reset the service before changing devices."
                )

            return existing

    @classmethod
    def reset_instance(cls) -> None:
        """
        Reset the process-wide singleton.

        Intended for tests and controlled application shutdown.
        """

        with cls._instance_lock:

            instance = cls._instance
            cls._instance = None

            if instance is None:
                return

            instance.pipeline = None

            try:
                import gc
                import torch

                gc.collect()

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception:
                logger.debug(
                    "Unable to fully clean up PredictorService resources.",
                    exc_info=True,
                )

    # =========================================================================
    # INPUT VALIDATION
    # =========================================================================

    def _validate_input_schema(
        self,
        history_df: pd.DataFrame,
    ) -> None:
        """
        Validate the incoming dataframe contract.
        """

        if not isinstance(history_df, pd.DataFrame):
            raise TypeError(
                "history_df must be a pandas DataFrame."
            )

        required_columns = {
            self.config.id_column,
            self.config.timestamp_column,
            self.config.target_column,
        }

        missing_columns = (
            required_columns
            - set(history_df.columns)
        )

        if missing_columns:
            raise ForecastValidationError(
                "History dataframe is missing required columns: "
                f"{sorted(missing_columns)}. "
                f"Expected: {sorted(required_columns)}"
            )

        if history_df.empty:
            raise InsufficientHistoryError(
                "History dataframe is empty."
            )

    # =========================================================================
    # TIMESTAMP NORMALIZATION
    # =========================================================================

    @staticmethod
    def _normalize_timestamps(
        values: pd.Series,
    ) -> pd.Series:
        """
        Convert timestamps to timezone-naive normalized daily timestamps.

        UTC is used when timezone-aware timestamps are supplied so that
        mixed timezone representations do not silently produce inconsistent
        calendar days.
        """

        timestamps = pd.to_datetime(
            values,
            errors="coerce",
            utc=True,
        )

        if timestamps.isna().any():
            return timestamps

        return (
            timestamps
            .dt.tz_convert(None)
            .dt.normalize()
        )

    # =========================================================================
    # HISTORY PREPARATION
    # =========================================================================

    def _prepare_medicine_history(
        self,
        item_id: str,
        history_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Convert raw medicine observations into a continuous daily series.

        Pipeline
        --------

        raw history
            -> filter medicine
            -> normalize ID
            -> validate timestamps
            -> validate demand
            -> aggregate duplicate dates
            -> create complete daily calendar
            -> fill missing days with zero demand
            -> validate final series
        """

        cfg = self.config

        normalized_item_id = str(item_id).strip()

        if not normalized_item_id:
            raise ValueError(
                "item_id must not be empty."
            )

        self._validate_input_schema(history_df)

        normalized_ids = (
            history_df[cfg.id_column]
            .astype("string")
            .str.strip()
        )

        history = history_df.loc[
            normalized_ids == normalized_item_id,
            [
                cfg.id_column,
                cfg.timestamp_column,
                cfg.target_column,
            ],
        ].copy()

        if history.empty:
            raise InsufficientHistoryError(
                f"No history found for item_id={normalized_item_id!r}."
            )

        # ---------------------------------------------------------------------
        # Normalize medicine ID
        # ---------------------------------------------------------------------

        history[cfg.id_column] = normalized_item_id

        # ---------------------------------------------------------------------
        # Normalize timestamps
        # ---------------------------------------------------------------------

        history[cfg.timestamp_column] = (
            self._normalize_timestamps(
                history[cfg.timestamp_column]
            )
        )

        invalid_dates = (
            history[cfg.timestamp_column]
            .isna()
        )

        if invalid_dates.any():

            invalid_count = int(
                invalid_dates.sum()
            )

            raise ForecastValidationError(
                f"item_id={normalized_item_id!r} contains "
                f"{invalid_count} invalid timestamp(s)."
            )

        # ---------------------------------------------------------------------
        # Normalize demand
        # ---------------------------------------------------------------------

        history[cfg.target_column] = pd.to_numeric(
            history[cfg.target_column],
            errors="coerce",
        )

        invalid_targets = (
            history[cfg.target_column]
            .isna()
        )

        if invalid_targets.any():

            invalid_count = int(
                invalid_targets.sum()
            )

            raise ForecastValidationError(
                f"item_id={normalized_item_id!r} contains "
                f"{invalid_count} non-numeric demand value(s)."
            )

        demand = (
            history[cfg.target_column]
            .to_numpy(dtype=float)
        )

        if not np.isfinite(demand).all():
            raise ForecastValidationError(
                f"item_id={normalized_item_id!r} contains "
                "non-finite demand values."
            )

        if (demand < 0.0).any():
            raise ForecastValidationError(
                f"item_id={normalized_item_id!r} contains "
                "negative demand values."
            )

        # ---------------------------------------------------------------------
        # Aggregate duplicate observations
        # ---------------------------------------------------------------------

        daily_observed = (
            history
            .groupby(
                cfg.timestamp_column,
                as_index=False,
                sort=True,
            )[cfg.target_column]
            .sum()
        )

        if daily_observed.empty:
            raise InsufficientHistoryError(
                f"item_id={normalized_item_id!r} has no usable observations."
            )

        aggregated_values = (
            daily_observed[cfg.target_column]
            .to_numpy(dtype=float)
        )

        if not np.isfinite(aggregated_values).all():
            raise ForecastValidationError(
                f"item_id={normalized_item_id!r} contains "
                "non-finite values after daily aggregation."
            )

        if (aggregated_values < 0.0).any():
            raise ForecastValidationError(
                f"item_id={normalized_item_id!r} contains "
                "negative values after daily aggregation."
            )

        daily_observed = daily_observed.sort_values(
            cfg.timestamp_column,
            kind="stable",
        )

        start_date = daily_observed[
            cfg.timestamp_column
        ].iloc[0]

        end_date = daily_observed[
            cfg.timestamp_column
        ].iloc[-1]

        # ---------------------------------------------------------------------
        # Calendarization
        # ---------------------------------------------------------------------

        complete_dates = pd.date_range(
            start=start_date,
            end=end_date,
            freq="D",
        )

        daily = (
            daily_observed
            .set_index(cfg.timestamp_column)
            .reindex(complete_dates)
            .rename_axis(cfg.timestamp_column)
            .reset_index()
        )

        daily[cfg.target_column] = (
            pd.to_numeric(
                daily[cfg.target_column],
                errors="coerce",
            )
            .fillna(0.0)
            .astype(float)
        )

        daily[cfg.id_column] = normalized_item_id

        daily = daily[
            [
                cfg.id_column,
                cfg.timestamp_column,
                cfg.target_column,
            ]
        ].copy()

        # ---------------------------------------------------------------------
        # Final validation
        # ---------------------------------------------------------------------

        expected_rows = (
            (end_date - start_date).days + 1
        )

        if len(daily) != expected_rows:
            raise ForecastValidationError(
                f"Calendar construction failed for "
                f"item_id={normalized_item_id!r}: "
                f"expected {expected_rows} rows, "
                f"received {len(daily)}."
            )

        if daily[cfg.timestamp_column].duplicated().any():
            raise ForecastValidationError(
                f"Duplicate dates remain after calendarization "
                f"for item_id={normalized_item_id!r}."
            )

        final_values = (
            daily[cfg.target_column]
            .to_numpy(dtype=float)
        )

        if not np.isfinite(final_values).all():
            raise ForecastValidationError(
                f"Calendarized history contains non-finite demand "
                f"for item_id={normalized_item_id!r}."
            )

        if (final_values < 0.0).any():
            raise ForecastValidationError(
                f"Calendarized history contains negative demand "
                f"for item_id={normalized_item_id!r}."
            )

        if len(daily) >= 3:

            inferred_frequency = pd.infer_freq(
                daily[cfg.timestamp_column]
            )

            if inferred_frequency not in {"D", "1D"}:
                raise ForecastValidationError(
                    f"Daily frequency validation failed for "
                    f"item_id={normalized_item_id!r}: "
                    f"inferred_frequency={inferred_frequency!r}."
                )

        zero_days = int(
            (daily[cfg.target_column] == 0.0).sum()
        )

        logger.info(
            (
                "CHRONOS DAILY HISTORY | item=%s | rows=%d | "
                "start=%s | end=%s | zero_days=%d"
            ),
            normalized_item_id,
            len(daily),
            start_date.date(),
            end_date.date(),
            zero_days,
        )

        return daily

    # =========================================================================
    # SINGLE MEDICINE FORECAST
    # =========================================================================

    def forecast_medicine(
        self,
        item_id: str,
        history_df: pd.DataFrame,
    ) -> MedicineForecastResult:
        """
        Forecast one medicine for the configured prediction horizon.
        """

        cfg = self.config

        normalized_item_id = str(item_id).strip()

        if not normalized_item_id:
            raise ValueError(
                "item_id must not be empty."
            )

        history = self._prepare_medicine_history(
            item_id=normalized_item_id,
            history_df=history_df,
        )

        # ---------------------------------------------------------------------
        # Apply maximum production context
        # ---------------------------------------------------------------------

        history = (
            history
            .tail(cfg.context_length)
            .copy()
        )

        context_used = len(history)

        if context_used < self._MIN_OBSERVATIONS:
            raise InsufficientHistoryError(
                f"item_id={normalized_item_id!r} has only "
                f"{context_used} daily observation(s); "
                f"minimum {self._MIN_OBSERVATIONS} required."
            )

        if context_used < cfg.context_length:
            logger.warning(
                (
                    "SHORT CHRONOS CONTEXT | item=%s | "
                    "actual=%d | configured=%d"
                ),
                normalized_item_id,
                context_used,
                cfg.context_length,
            )

        history_start = history[
            cfg.timestamp_column
        ].iloc[0]

        history_end = history[
            cfg.timestamp_column
        ].iloc[-1]

        logger.info(
            (
                "CHRONOS INPUT | item=%s | configured_context=%d | "
                "actual_rows=%d | start=%s | end=%s"
            ),
            normalized_item_id,
            cfg.context_length,
            context_used,
            history_start.date(),
            history_end.date(),
        )

        # ---------------------------------------------------------------------
        # Execute Chronos inference
        # ---------------------------------------------------------------------

        try:

            with self._get_prediction_lock():

                raw_forecast = self.pipeline.predict_df(
                    history,
                    prediction_length=cfg.prediction_length,
                    quantile_levels=list(
                        cfg.quantile_levels
                    ),
                    id_column=cfg.id_column,
                    timestamp_column=cfg.timestamp_column,
                    target=cfg.target_column,
                )

        except Exception as exc:

            logger.exception(
                "Chronos-2 prediction failed | item=%s",
                normalized_item_id,
            )

            raise RuntimeError(
                "Chronos-2 prediction failed "
                f"for item_id={normalized_item_id!r}."
            ) from exc

        # ---------------------------------------------------------------------
        # Validate Chronos output
        # ---------------------------------------------------------------------

        self._validate_raw_forecast(
            item_id=normalized_item_id,
            raw_forecast=raw_forecast,
            history_end=history_end,
        )

        # ---------------------------------------------------------------------
        # Convert to typed result
        # ---------------------------------------------------------------------

        return self._to_result(
            item_id=normalized_item_id,
            context_used=context_used,
            raw_forecast=raw_forecast,
        )

    # =========================================================================
    # BATCH FORECAST
    # =========================================================================

    def forecast_batch(
        self,
        history_df: pd.DataFrame,
        item_ids: Optional[Sequence[str]] = None,
    ) -> tuple[
        list[MedicineForecastResult],
        list[str],
    ]:
        """
        Forecast multiple medicines.

        Failure is isolated at medicine level.

        Returns
        -------
        results:
            Successfully generated forecasts.

        failed:
            Medicine IDs that could not be forecast.
        """

        self._validate_input_schema(history_df)

        cfg = self.config

        if item_ids is None:

            ids = (
                history_df[cfg.id_column]
                .astype("string")
                .str.strip()
                .dropna()
                .loc[lambda values: values != ""]
                .unique()
                .tolist()
            )

            ids = sorted(
                str(value)
                for value in ids
            )

        else:

            ids = []

            for item_id in item_ids:

                normalized = str(item_id).strip()

                if normalized:
                    ids.append(normalized)

            ids = list(dict.fromkeys(ids))

        if not ids:
            raise ValueError(
                "No medicine IDs available for forecasting."
            )

        logger.info(
            (
                "Starting Chronos-2 batch forecast | medicines=%d | "
                "context=%d | horizon=%d"
            ),
            len(ids),
            cfg.context_length,
            cfg.prediction_length,
        )

        results: list[
            MedicineForecastResult
        ] = []

        failed: list[str] = []

        for item_id in ids:

            try:

                result = self.forecast_medicine(
                    item_id=item_id,
                    history_df=history_df,
                )

                results.append(result)

            except InsufficientHistoryError as exc:

                logger.warning(
                    (
                        "Chronos forecast skipped | "
                        "item=%s | reason=%s"
                    ),
                    item_id,
                    exc,
                )

                failed.append(item_id)

            except Exception as exc:

                logger.exception(
                    (
                        "Chronos forecast failed | "
                        "item=%s | error=%s"
                    ),
                    item_id,
                    exc,
                )

                failed.append(item_id)

        logger.info(
            (
                "Chronos-2 batch forecast completed | "
                "requested=%d | successful=%d | failed=%d"
            ),
            len(ids),
            len(results),
            len(failed),
        )

        return results, failed

    # =========================================================================
    # RAW FORECAST VALIDATION
    # =========================================================================

    def _validate_raw_forecast(
        self,
        item_id: str,
        raw_forecast: pd.DataFrame,
        history_end: pd.Timestamp,
    ) -> None:
        """
        Validate Chronos-2 predict_df() output.

        Expected contract
        -----------------

        Chronos-2 is called using history for a single medicine.

        The production output is expected to contain:

            timestamp
            0.1
            0.2
            ...
            0.9

        The model does not need to echo the medicine ID.

        The configured point forecast is taken directly from the configured
        quantile, normally the "0.5" / P50 column.
        """

        cfg = self.config

        if not isinstance(raw_forecast, pd.DataFrame):
            raise TypeError(
                "Chronos output must be a pandas DataFrame."
            )

        if raw_forecast.empty:
            raise InsufficientHistoryError(
                f"Chronos returned no forecast rows "
                f"for item_id={item_id!r}."
            )

        quantile_columns = [
            str(float(level))
            for level in cfg.quantile_levels
        ]

        required_columns = {
            cfg.timestamp_column,
            *quantile_columns,
        }

        missing_columns = (
            required_columns
            - set(raw_forecast.columns)
        )

        if missing_columns:
            raise ForecastValidationError(
                "Chronos output is missing required columns: "
                f"{sorted(missing_columns)}."
            )

        if len(raw_forecast) != cfg.prediction_length:
            raise ForecastValidationError(
                f"Chronos returned {len(raw_forecast)} row(s) "
                f"for item_id={item_id!r}; expected "
                f"{cfg.prediction_length}."
            )

        # ---------------------------------------------------------------------
        # Timestamp validation
        # ---------------------------------------------------------------------

        timestamps = pd.to_datetime(
            raw_forecast[cfg.timestamp_column],
            errors="coerce",
            utc=True)

        if timestamps.isna().any():
            raise ForecastValidationError(
                f"Chronos returned invalid forecast timestamps "
                f"for item_id={item_id!r}.")
            
        timestamps = (
            timestamps
            .dt.tz_convert(None).dt.normalize())
            
        if timestamps.duplicated().any():
            raise ForecastValidationError(
                f"Chronos returned duplicate forecast dates "
                f"for item_id={item_id!r}.")

        if not timestamps.is_monotonic_increasing:
            raise ForecastValidationError(
                f"Chronos forecast dates are not sorted "
                f"for item_id={item_id!r}.")
            
        expected_dates = pd.date_range(
            start=pd.Timestamp(history_end).normalize()
            + pd.Timedelta(days=1),
            periods=cfg.prediction_length,
            freq="D",)
            
        actual_dates = (
            timestamps.reset_index(drop=True).to_numpy(dtype="datetime64[ns]"))
            
        expected_dates_array = (
            expected_dates.to_numpy(dtype="datetime64[ns]"))

        if not np.array_equal(
            actual_dates,
            expected_dates_array):

            raise ForecastValidationError(
                f"Chronos returned an unexpected forecast horizon "
                f"for item_id={item_id!r}. "
                f"Expected {expected_dates[0].date()} through "
                f"{expected_dates[-1].date()}.")

        # ---------------------------------------------------------------------
        # Quantile validation
        # ---------------------------------------------------------------------

        for column in quantile_columns:

            values = pd.to_numeric(
                raw_forecast[column],
                errors="coerce",
            )

            if values.isna().any():
                raise ForecastValidationError(
                    f"Chronos returned non-numeric values in "
                    f"quantile column {column!r} "
                    f"for item_id={item_id!r}."
                )

            array = values.to_numpy(dtype=float)

            if not np.isfinite(array).all():
                raise ForecastValidationError(
                    f"Chronos returned non-finite values in "
                    f"quantile column {column!r} "
                    f"for item_id={item_id!r}."
                )

            if (array < 0.0).any():
                logger.warning(
                    (
                        "Negative Chronos quantile values detected | "
                        "item=%s | quantile=%s | policy=clamp_to_zero"
                    ),
                    item_id,
                    column,
                )

        # ---------------------------------------------------------------------
        # Quantile ordering validation
        # ---------------------------------------------------------------------

        quantile_matrix = np.column_stack(
            [
                pd.to_numeric(
                    raw_forecast[column],
                    errors="raise",
                ).to_numpy(dtype=float)
                for column in quantile_columns
            ]
        )

        if (
            np.diff(
                quantile_matrix,
                axis=1,
            ) < 0.0
        ).any():

            logger.warning(
                (
                    "Chronos quantile crossing detected | item=%s | "
                    "policy=cumulative_max_repair"
                ),
                item_id,
            )

    # =========================================================================
    # RESULT CONVERSION
    # =========================================================================

    def _to_result(
        self,
        item_id: str,
        context_used: int,
        raw_forecast: pd.DataFrame,
    ) -> MedicineForecastResult:
        """
        Convert validated Chronos output into project forecast schemas.

        Production policy:

        1. Convert values to float.
        2. Clamp negative values to zero.
        3. Repair quantile crossing using cumulative maximum.
        4. Use the configured point quantile as predicted_demand.
        """

        cfg = self.config

        quantile_columns = [
            str(float(level))
            for level in cfg.quantile_levels
        ]

        point_column = str(
            float(cfg.point_quantile)
        )

        if point_column not in raw_forecast.columns:
            raise ForecastValidationError(
                f"Configured point quantile column "
                f"{point_column!r} is missing."
            )

        days: list[ForecastDayResult] = []

        for _, row in raw_forecast.iterrows():

            forecast_date = (
                pd.Timestamp(
                    row[cfg.timestamp_column]
                )
                .normalize()
            )

            raw_values = np.asarray(
                [
                    float(row[column])
                    for column in quantile_columns
                ],
                dtype=float,
            )

            # Non-negative demand policy.
            sanitized_values = np.maximum(
                raw_values,
                0.0,
            )

            # Monotonic quantile policy.
            corrected_values = np.maximum.accumulate(
                sanitized_values
            )

            if not np.array_equal(
                raw_values,
                corrected_values,
            ):
                logger.warning(
                    (
                        "Forecast quantiles corrected | item=%s | "
                        "date=%s | raw=%s | corrected=%s"
                    ),
                    item_id,
                    forecast_date.date(),
                    raw_values.tolist(),
                    corrected_values.tolist(),
                )

            quantiles = QuantileForecast(
                p10=float(corrected_values[0]),
                p20=float(corrected_values[1]),
                p30=float(corrected_values[2]),
                p40=float(corrected_values[3]),
                p50=float(corrected_values[4]),
                p60=float(corrected_values[5]),
                p70=float(corrected_values[6]),
                p80=float(corrected_values[7]),
                p90=float(corrected_values[8]),
            )

            point_attribute = (
                f"p{int(float(cfg.point_quantile) * 100)}"
            )

            try:
                predicted_demand = float(
                    getattr(
                        quantiles,
                        point_attribute,
                    )
                )

            except AttributeError as exc:
                raise ForecastValidationError(
                    f"QuantileForecast does not expose "
                    f"{point_attribute!r}."
                ) from exc

            days.append(
                ForecastDayResult(
                    forecast_date=forecast_date.date(),
                    predicted_demand=round(
                        predicted_demand,
                        2,
                    ),
                    quantiles=quantiles,
                )
            )

        if len(days) != cfg.prediction_length:
            raise ForecastValidationError(
                f"Result conversion produced {len(days)} forecast "
                f"day(s); expected {cfg.prediction_length}."
            )

        return MedicineForecastResult(
            medicine_id=str(item_id),
            generated_at=datetime.now(
                timezone.utc
            ),
            context_length_used=int(context_used),
            prediction_length=int(
                cfg.prediction_length
            ),
            model_id=cfg.model_id,
            days=days,
        )