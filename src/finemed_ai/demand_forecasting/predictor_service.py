from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import List, Optional

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


# ============================================================================
# ERRORS
# ============================================================================


class InsufficientHistoryError(ValueError):
    """Raised when a medicine has no usable history to forecast from."""


# ============================================================================
# PREDICTOR SERVICE
# ============================================================================


class PredictorService:
    """
    Production Chronos-2 predictor service.

    Responsibilities
    ----------------
    1. Validate the incoming dataframe contract.
    2. Filter history for one medicine.
    3. Normalize timestamps.
    4. Validate demand values.
    5. Aggregate duplicate observations on the same day.
    6. Calendarize the series to continuous daily frequency.
    7. Apply the validated 730-observation context limit.
    8. Execute Chronos-2 probabilistic forecasting.
    9. Validate the returned forecast.
    10. Convert the result into the project's typed forecast schema.

    Production configuration is supplied through ForecastConfig.

    Validated production configuration:

        model_id:
            amazon/chronos-2

        context_length:
            730

        prediction_length:
            30

        point forecast:
            P50

        quantiles:
            P10 ... P90

        bias correction:
            disabled
    """

    _instance: Optional["PredictorService"] = None

    # ------------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------------

    def __init__(
        self,
        config: ForecastConfig = DEFAULT_CONFIG,
        device: Optional[str] = None,
    ) -> None:
        """
        Initialize the Chronos-2 predictor.

        The Chronos model is loaded once per PredictorService instance.
        """

        if config is None:
            raise ValueError(
                "config must not be None."
            )

        self.config = config

        self.device = self._resolve_device(
            device
        )

        logger.info(
            "Loading Chronos-2 | model=%s | device=%s | "
            "context=%d | horizon=%d | point=P%d",
            config.model_id,
            self.device,
            config.context_length,
            config.prediction_length,
            int(config.point_quantile * 100),
        )

        t0 = datetime.now(
            timezone.utc
        )

        try:
            import torch

            from chronos import Chronos2Pipeline

        except ImportError as exc:
            raise ImportError(
                "Chronos-2 dependencies are not available. "
                "Ensure torch and the chronos package are installed "
                "in the active environment."
            ) from exc

        # --------------------------------------------------------------------
        # Torch dtype
        # --------------------------------------------------------------------
        #
        # CUDA:
        #     bfloat16 is used as in the validated implementation.
        #
        # CPU:
        #     float32 is used for compatibility.
        # --------------------------------------------------------------------

        torch_dtype = (
            torch.bfloat16
            if self.device == "cuda"
            else torch.float32
        )

        try:
            self.pipeline = (
                Chronos2Pipeline.from_pretrained(
                    config.model_id,
                    device_map=self.device,
                    torch_dtype=torch_dtype,
                )
            )

        except Exception as exc:
            logger.exception(
                "Failed to load Chronos-2 model=%s on device=%s.",
                config.model_id,
                self.device,
            )

            raise RuntimeError(
                "Failed to initialize Chronos-2 pipeline "
                f"for model={config.model_id!r} "
                f"on device={self.device!r}."
            ) from exc

        elapsed = (
            datetime.now(timezone.utc) - t0
        ).total_seconds()

        logger.info(
            "Chronos-2 loaded successfully | model=%s | "
            "device=%s | elapsed=%.1fs",
            config.model_id,
            self.device,
            elapsed,
        )

    # ------------------------------------------------------------------------
    # Device
    # ------------------------------------------------------------------------

    @staticmethod
    def _resolve_device(
        device: Optional[str],
    ) -> str:
        """
        Resolve the execution device.

        If no device is explicitly supplied:

            CUDA available -> cuda
            otherwise       -> cpu
        """

        if device is not None:

            normalized = (
                str(device)
                .strip()
                .lower()
            )

            if not normalized:
                raise ValueError(
                    "device must not be empty."
                )

            supported_prefixes = (
                "cpu",
                "cuda",
                "mps",
            )

            if not normalized.startswith(
                supported_prefixes
            ):
                raise ValueError(
                    f"Unsupported device={device!r}. "
                    "Expected cpu, cuda, or mps."
                )

            return normalized

        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"

        except ImportError:
            raise ImportError(
                "PyTorch is required to initialize "
                "PredictorService."
            )

        return "cpu"

    # ------------------------------------------------------------------------
    # Singleton
    # ------------------------------------------------------------------------

    @classmethod
    def get_instance(
        cls,
        config: ForecastConfig = DEFAULT_CONFIG,
        device: Optional[str] = None,
    ) -> "PredictorService":
        """
        Return the process-wide PredictorService instance.

        The singleton is configuration-safe.

        If an instance already exists, requesting a different configuration
        or device raises an error instead of silently reusing the existing
        model with incompatible settings.
        """

        if config is None:
            raise ValueError(
                "config must not be None."
            )

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
            existing._resolve_device(device)
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
        Reset the singleton.

        Intended primarily for tests and controlled process lifecycle
        management.
        """

        instance = cls._instance

        cls._instance = None

        if instance is not None:

            # Explicitly release the pipeline reference.
            instance.pipeline = None

            # Release CUDA memory when applicable.
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception:
                # Resetting the singleton must not fail merely because
                # optional GPU cleanup is unavailable.
                logger.debug(
                    "Unable to perform CUDA cleanup during "
                    "PredictorService reset.",
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
        Validate the dataframe contract before filtering.
        """

        if not isinstance(
            history_df,
            pd.DataFrame,
        ):
            raise TypeError(
                "history_df must be a pandas DataFrame."
            )

        required = {
            self.config.id_column,
            self.config.timestamp_column,
            self.config.target_column,
        }

        missing = (
            required
            - set(history_df.columns)
        )

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

    # =========================================================================
    # HISTORY PREPARATION
    # =========================================================================

    def _prepare_medicine_history(
        self,
        item_id: str,
        history_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Prepare one medicine as a complete daily time series.

        Processing:

            raw transaction history
                    |
                    v
            filter medicine
                    |
                    v
            validate timestamps/demand
                    |
                    v
            normalize timestamps
                    |
                    v
            aggregate same-day transactions
                    |
                    v
            construct continuous daily calendar
                    |
                    v
            fill missing dates with zero demand
                    |
                    v
            Chronos-ready daily series
        """

        cfg = self.config

        item_id = str(
            item_id
        ).strip()

        if not item_id:
            raise ValueError(
                "item_id must not be empty."
            )

        self._validate_input_schema(
            history_df
        )

        # --------------------------------------------------------------------
        # Filter medicine
        # --------------------------------------------------------------------

        normalized_ids = (
            history_df[
                cfg.id_column
            ]
            .astype(str)
            .str.strip()
        )

        history = history_df[
            normalized_ids == item_id
        ].copy()

        if history.empty:
            raise InsufficientHistoryError(
                f"No history for item_id={item_id}"
            )

        # --------------------------------------------------------------------
        # Normalize ID
        # --------------------------------------------------------------------

        history[cfg.id_column] = (
            history[cfg.id_column]
            .astype(str)
            .str.strip()
        )

        # --------------------------------------------------------------------
        # Parse timestamps
        # --------------------------------------------------------------------

        history[cfg.timestamp_column] = (
            pd.to_datetime(
                history[cfg.timestamp_column],
                errors="coerce",
            )
        )

        invalid_dates = (
            history[cfg.timestamp_column]
            .isna()
        )

        if invalid_dates.any():

            count = int(
                invalid_dates.sum()
            )

            raise ValueError(
                f"item_id={item_id} contains "
                f"{count} invalid timestamp(s)."
            )

        # --------------------------------------------------------------------
        # Parse demand
        # --------------------------------------------------------------------

        history[cfg.target_column] = (
            pd.to_numeric(
                history[cfg.target_column],
                errors="coerce",
            )
        )

        invalid_targets = (
            history[cfg.target_column]
            .isna()
        )

        if invalid_targets.any():

            count = int(
                invalid_targets.sum()
            )

            raise ValueError(
                f"item_id={item_id} contains "
                f"{count} non-numeric demand value(s)."
            )

        # --------------------------------------------------------------------
        # Validate finite demand
        # --------------------------------------------------------------------

        demand_array = (
            history[cfg.target_column]
            .to_numpy(dtype=float)
        )

        if not np.isfinite(
            demand_array
        ).all():

            raise ValueError(
                f"item_id={item_id} contains "
                "non-finite demand values."
            )

        # --------------------------------------------------------------------
        # Validate non-negative demand
        # --------------------------------------------------------------------

        if (
            history[cfg.target_column] < 0
        ).any():

            raise ValueError(
                f"item_id={item_id} contains "
                "negative demand values."
            )

        # --------------------------------------------------------------------
        # Normalize timestamps to daily frequency
        # --------------------------------------------------------------------

        history[cfg.timestamp_column] = (
            history[cfg.timestamp_column]
            .dt.normalize()
        )

        # --------------------------------------------------------------------
        # Aggregate duplicate same-day observations
        # --------------------------------------------------------------------

        history = (
            history
            .groupby(
                cfg.timestamp_column,
                as_index=False,
            )[cfg.target_column]
            .sum()
        )

        if history.empty:
            raise InsufficientHistoryError(
                f"item_id={item_id} has no usable observations."
            )

        # --------------------------------------------------------------------
        # Validate aggregated demand
        # --------------------------------------------------------------------

        aggregated_demand = (
            history[cfg.target_column]
            .to_numpy(dtype=float)
        )

        if not np.isfinite(
            aggregated_demand
        ).all():

            raise ValueError(
                f"item_id={item_id} contains "
                "non-finite demand after daily aggregation."
            )

        if (
            history[cfg.target_column] < 0
        ).any():

            raise ValueError(
                f"item_id={item_id} contains "
                "negative demand after daily aggregation."
            )

        # --------------------------------------------------------------------
        # Sort
        # --------------------------------------------------------------------

        history = history.sort_values(
            cfg.timestamp_column
        )

        start_date = (
            history[cfg.timestamp_column]
            .min()
        )

        end_date = (
            history[cfg.timestamp_column]
            .max()
        )

        if pd.isna(
            start_date
        ) or pd.isna(
            end_date
        ):

            raise ValueError(
                f"item_id={item_id} has an invalid "
                "history date range."
            )

        if end_date < start_date:
            raise ValueError(
                f"item_id={item_id} has an invalid "
                f"date range: {start_date} > {end_date}."
            )

        # --------------------------------------------------------------------
        # Construct complete daily calendar
        # --------------------------------------------------------------------

        complete_dates = pd.date_range(
            start=start_date,
            end=end_date,
            freq="D",
        )

        daily = (
            history
            .set_index(
                cfg.timestamp_column
            )
            .reindex(
                complete_dates
            )
            .rename_axis(
                cfg.timestamp_column
            )
            .reset_index()
        )

        # --------------------------------------------------------------------
        # Missing calendar days represent zero observed demand.
        # --------------------------------------------------------------------

        daily[cfg.target_column] = (
            pd.to_numeric(
                daily[cfg.target_column],
                errors="coerce",
            )
            .fillna(0.0)
        )

        daily[cfg.id_column] = item_id

        daily = daily[
            [
                cfg.id_column,
                cfg.timestamp_column,
                cfg.target_column,
            ]
        ]

        # --------------------------------------------------------------------
        # Final daily validation
        # --------------------------------------------------------------------

        expected_rows = (
            end_date - start_date
        ).days + 1

        if len(daily) != expected_rows:

            raise ValueError(
                f"Daily calendar construction failed for "
                f"item_id={item_id}: expected "
                f"{expected_rows} rows, got {len(daily)}."
            )

        if (
            daily[cfg.timestamp_column]
            .duplicated()
            .any()
        ):

            raise ValueError(
                f"Duplicate dates remain after calendarization "
                f"for item_id={item_id}."
            )

        final_demand = (
            daily[cfg.target_column]
            .to_numpy(dtype=float)
        )

        if not np.isfinite(
            final_demand
        ).all():

            raise ValueError(
                f"Calendarized history contains non-finite "
                f"demand for item_id={item_id}."
            )

        if (
            daily[cfg.target_column] < 0
        ).any():

            raise ValueError(
                f"Calendarized history contains negative "
                f"demand for item_id={item_id}."
            )

        if len(daily) >= 3:

            inferred_freq = pd.infer_freq(
                daily[cfg.timestamp_column]
            )

            if inferred_freq not in {
                "D",
                "1D",
            }:

                raise ValueError(
                    f"Daily frequency validation failed for "
                    f"item_id={item_id}: "
                    f"inferred_freq={inferred_freq!r}"
                )

        zero_days = int(
            (
                daily[cfg.target_column]
                == 0
            ).sum()
        )

        logger.info(
            "CHRONOS DAILY HISTORY | item=%s | rows=%d | "
            "start=%s | end=%s | zero_days=%d",
            item_id,
            len(daily),
            start_date,
            end_date,
            zero_days,
        )

        return daily

    # =========================================================================
    # CORE FORECASTING
    # =========================================================================

    def forecast_medicine(
        self,
        item_id: str,
        history_df: pd.DataFrame,
    ) -> MedicineForecastResult:
        """
        Forecast one medicine for the configured prediction horizon.

        The input does not need to be calendarized.

        This method guarantees that Chronos receives:

            item_id
            timestamp
            target

        with a continuous daily timestamp sequence.
        """

        cfg = self.config

        item_id = str(
            item_id
        ).strip()

        if not item_id:
            raise ValueError(
                "item_id must not be empty."
            )

        history = (
            self._prepare_medicine_history(
                item_id=item_id,
                history_df=history_df,
            )
        )

        # --------------------------------------------------------------------
        # Apply validated production context.
        #
        # Important:
        # We use the most recent observations because the production
        # configuration is frozen at 730 daily observations.
        # --------------------------------------------------------------------

        history = history.tail(
            cfg.context_length
        ).copy()

        actual_context = len(history)

        if actual_context == 0:
            raise InsufficientHistoryError(
                f"item_id={item_id} has no observations "
                "after context preparation."
            )

        # --------------------------------------------------------------------
        # Minimum Chronos context
        # --------------------------------------------------------------------

        min_observations = 3

        if actual_context < min_observations:

            raise InsufficientHistoryError(
                f"item_id={item_id} has only "
                f"{actual_context} daily observation(s); "
                f"minimum {min_observations} required "
                "for Chronos-2."
            )

        logger.info(
            "CHRONOS INPUT | item=%s | configured_context=%d | "
            "actual_rows=%d | start=%s | end=%s",
            item_id,
            cfg.context_length,
            actual_context,
            history[cfg.timestamp_column].min(),
            history[cfg.timestamp_column].max(),
        )

        # --------------------------------------------------------------------
        # Short-context warning
        # --------------------------------------------------------------------
        #
        # We do NOT reject shorter histories here.
        #
        # Some medicines legitimately have less than 730 days of history.
        # The validated production context is therefore a MAXIMUM context,
        # not a requirement that every medicine must have 730 observations.
        # --------------------------------------------------------------------

        if actual_context < cfg.context_length:

            logger.warning(
                "SHORT CHRONOS CONTEXT | item=%s | "
                "actual_context=%d | configured_context=%d",
                item_id,
                actual_context,
                cfg.context_length,
            )

        # --------------------------------------------------------------------
        # Execute Chronos-2
        # --------------------------------------------------------------------

        try:

            raw = self.pipeline.predict_df(
                history,
                prediction_length=(
                    cfg.prediction_length
                ),
                quantile_levels=list(
                    cfg.quantile_levels
                ),
                id_column=cfg.id_column,
                timestamp_column=(
                    cfg.timestamp_column
                ),
                target=cfg.target_column,
            )

        except Exception as exc:

            logger.exception(
                "Chronos-2 prediction failed for item_id=%s.",
                item_id,
            )

            raise RuntimeError(
                f"Chronos-2 prediction failed "
                f"for item_id={item_id}."
            ) from exc

        # --------------------------------------------------------------------
        # Validate raw result
        # --------------------------------------------------------------------

        self._validate_raw_forecast(
            item_id=item_id,
            raw_forecast=raw,
        )

        # --------------------------------------------------------------------
        # Convert to project schema
        # --------------------------------------------------------------------

        return self._to_result(
            item_id=item_id,
            context_used=actual_context,
            raw_forecast=raw,
        )

    # =========================================================================
    # BATCH FORECASTING
    # =========================================================================

    def forecast_batch(
        self,
        history_df: pd.DataFrame,
        item_ids: Optional[List[str]] = None,
    ) -> tuple[
        List[MedicineForecastResult],
        List[str],
    ]:
        """
        Forecast multiple medicines.

        Failures are isolated at medicine level.

        Returns:

            results:
                Successfully forecast medicines.

            failed:
                Medicine IDs whose forecast failed.
        """

        self._validate_input_schema(
            history_df
        )

        cfg = self.config

        # --------------------------------------------------------------------
        # Determine medicine IDs
        # --------------------------------------------------------------------

        if item_ids is None:

            ids = sorted(
                history_df[
                    cfg.id_column
                ]
                .astype(str)
                .str.strip()
                .unique()
            )

        else:

            ids = [
                str(item_id).strip()
                for item_id in item_ids
                if str(item_id).strip()
            ]

            # Preserve order while removing duplicates.
            ids = list(
                dict.fromkeys(ids)
            )

        if not ids:

            raise ValueError(
                "No medicine IDs available for forecasting."
            )

        logger.info(
            "Starting Chronos-2 batch forecast | "
            "medicines=%d | context=%d | horizon=%d",
            len(ids),
            cfg.context_length,
            cfg.prediction_length,
        )

        results: List[
            MedicineForecastResult
        ] = []

        failed: List[str] = []

        for item_id in ids:

            try:

                result = (
                    self.forecast_medicine(
                        item_id=item_id,
                        history_df=history_df,
                    )
                )

                results.append(
                    result
                )

            except InsufficientHistoryError as exc:

                logger.warning(
                    "Chronos forecast skipped | item=%s | reason=%s",
                    item_id,
                    exc,
                )

                failed.append(
                    item_id
                )

            except Exception:

                logger.exception(
                    "Chronos forecast failed | item=%s",
                    item_id,
                )

                failed.append(
                    item_id
                )

        logger.info(
            "Chronos-2 batch forecast completed | "
            "requested=%d | successful=%d | failed=%d",
            len(ids),
            len(results),
            len(failed),
        )

        return results, failed

    # =========================================================================
    # RAW CHRONOS OUTPUT VALIDATION
    # =========================================================================

    def _validate_raw_forecast(
        self,
        item_id: str,
        raw_forecast: pd.DataFrame,
    ) -> None:
        """
        Validate the raw dataframe returned by Chronos-2.

        The method deliberately validates the result before conversion so
        malformed model output cannot silently enter the production schema.
        """

        cfg = self.config

        if not isinstance(
            raw_forecast,
            pd.DataFrame,
        ):

            raise TypeError(
                "Chronos output must be a pandas DataFrame."
            )

        if raw_forecast.empty:

            raise InsufficientHistoryError(
                f"Chronos returned no forecast rows "
                f"for item_id={item_id}."
            )

        required_columns = {
            cfg.id_column,
            cfg.timestamp_column,
            cfg.target_column,
        }

        # Chronos quantile columns are represented as strings such as
        # "0.1", "0.2", ..., "0.9".
        expected_quantile_columns = [
            str(level)
            for level in cfg.quantile_levels
        ]

        required_columns.update(
            expected_quantile_columns
        )

        missing = (
            required_columns
            - set(raw_forecast.columns)
        )

        if missing:

            raise ValueError(
                "Chronos output is missing required columns: "
                f"{sorted(missing)}"
            )

        expected_horizon = (
            cfg.prediction_length
        )

        if len(raw_forecast) != expected_horizon:

            raise ValueError(
                f"Chronos returned {len(raw_forecast)} "
                f"forecast row(s) for item_id={item_id}; "
                f"expected {expected_horizon}."
            )

        # --------------------------------------------------------------------
        # Validate timestamps
        # --------------------------------------------------------------------

        timestamps = pd.to_datetime(
            raw_forecast[
                cfg.timestamp_column
            ],
            errors="coerce",
        )

        if timestamps.isna().any():

            raise ValueError(
                f"Chronos returned invalid forecast "
                f"timestamps for item_id={item_id}."
            )

        timestamps = (
            timestamps.dt.normalize()
        )

        if timestamps.duplicated().any():

            raise ValueError(
                f"Chronos returned duplicate forecast "
                f"dates for item_id={item_id}."
            )

        if not timestamps.is_monotonic_increasing:

            raise ValueError(
                f"Chronos forecast dates are not sorted "
                f"for item_id={item_id}."
            )

        # --------------------------------------------------------------------
        # Validate target column
        # --------------------------------------------------------------------

        target_values = pd.to_numeric(
            raw_forecast[
                cfg.target_column
            ],
            errors="coerce",
        )

        if target_values.isna().any():

            raise ValueError(
                f"Chronos returned non-numeric point "
                f"forecast values for item_id={item_id}."
            )

        target_array = (
            target_values
            .to_numpy(dtype=float)
        )

        if not np.isfinite(
            target_array
        ).all():

            raise ValueError(
                f"Chronos returned non-finite point "
                f"forecast values for item_id={item_id}."
            )

        if (
            target_array < 0
        ).any():

            raise ValueError(
                f"Chronos returned negative point "
                f"forecast values for item_id={item_id}."
            )

        # --------------------------------------------------------------------
        # Validate quantiles
        # --------------------------------------------------------------------

        for column in expected_quantile_columns:

            values = pd.to_numeric(
                raw_forecast[column],
                errors="coerce",
            )

            if values.isna().any():

                raise ValueError(
                    f"Chronos returned non-numeric "
                    f"{column} quantile values "
                    f"for item_id={item_id}."
                )

            array = (
                values
                .to_numpy(dtype=float)
            )

            if not np.isfinite(
                array
            ).all():

                raise ValueError(
                    f"Chronos returned non-finite "
                    f"{column} quantile values "
                    f"for item_id={item_id}."
                )

            if (
                array < 0
            ).any():

                raise ValueError(
                    f"Chronos returned negative "
                    f"{column} quantile values "
                    f"for item_id={item_id}."
                )

        # --------------------------------------------------------------------
        # Validate quantile ordering
        # --------------------------------------------------------------------

        quantile_arrays = [
            pd.to_numeric(
                raw_forecast[column],
                errors="raise",
            )
            .to_numpy(dtype=float)
            for column in expected_quantile_columns
        ]

        for quantile_index in range(
            len(quantile_arrays) - 1
        ):

            lower = (
                quantile_arrays[
                    quantile_index
                ]
            )

            upper = (
                quantile_arrays[
                    quantile_index + 1
                ]
            )

            if (
                lower > upper
            ).any():

                lower_name = (
                    expected_quantile_columns[
                        quantile_index
                    ]
                )

                upper_name = (
                    expected_quantile_columns[
                        quantile_index + 1
                    ]
                )

                raise ValueError(
                    f"Chronos quantile ordering violation "
                    f"for item_id={item_id}: "
                    f"{lower_name} exceeds {upper_name}."
                )

        # --------------------------------------------------------------------
        # Validate point forecast against P50
        # --------------------------------------------------------------------
        #
        # Production policy explicitly defines the point forecast as P50.
        #
        # Chronos may expose its point forecast in `target`, while P50 is
        # separately exposed as the 0.5 quantile. They should therefore agree
        # within numerical precision.
        # --------------------------------------------------------------------

        p50_column = "0.5"

        p50_values = pd.to_numeric(
            raw_forecast[p50_column],
            errors="raise",
        ).to_numpy(
            dtype=float
        )

        point_values = target_array

        if not np.allclose(
            point_values,
            p50_values,
            rtol=1e-5,
            atol=1e-5,
        ):

            logger.warning(
                "Chronos point/P50 mismatch | item=%s | "
                "max_abs_diff=%s. Production point forecast "
                "will use configured P50.",
                item_id,
                float(
                    np.max(
                        np.abs(
                            point_values
                            - p50_values
                        )
                    )
                ),
            )

        # --------------------------------------------------------------------
        # Validate returned medicine ID
        # --------------------------------------------------------------------

        returned_ids = (
            raw_forecast[
                cfg.id_column
            ]
            .astype(str)
            .str.strip()
        )

        if (
            returned_ids != item_id
        ).any():

            mismatched = (
                returned_ids[
                    returned_ids != item_id
                ]
                .unique()
                .tolist()
            )

            raise ValueError(
                f"Chronos returned unexpected medicine IDs "
                f"for requested item_id={item_id}: "
                f"{mismatched}"
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
        Convert validated Chronos output into the project's typed schema.

        The production point forecast is explicitly P50.
        """

        cfg = self.config

        expected_quantile_columns = [
            str(level)
            for level in cfg.quantile_levels
        ]

        # --------------------------------------------------------------------
        # P50 must exist because it is the production point forecast.
        # --------------------------------------------------------------------

        p50_column = str(
            cfg.point_quantile
        )

        if p50_column not in (
            raw_forecast.columns
        ):

            raise ValueError(
                f"Configured point quantile column "
                f"{p50_column!r} is missing from Chronos output."
            )

        days: List[
            ForecastDayResult
        ] = []

        for _, row in raw_forecast.iterrows():

            # ----------------------------------------------------------------
            # Extract quantiles
            # ----------------------------------------------------------------

            raw_values = [
                max(
                    float(
                        row[column]
                    ),
                    0.0,
                )
                for column in expected_quantile_columns
            ]

            # ----------------------------------------------------------------
            # Defensive monotonicity enforcement
            # ----------------------------------------------------------------
            #
            # Chronos output is validated above. This remains as a defensive
            # final boundary so that tiny numerical irregularities cannot
            # produce an invalid production interval.
            # ----------------------------------------------------------------

            monotonic_values: List[
                float
            ] = []

            running_max = 0.0
            corrected = False

            for value in raw_values:

                if value < running_max:

                    corrected = True
                    value = running_max

                running_max = value

                monotonic_values.append(
                    value
                )

            if corrected:

                logger.warning(
                    "Non-monotonic Chronos quantiles detected "
                    "for item_id=%s on %s. "
                    "Corrected using cumulative maximum. "
                    "Raw=%s",
                    item_id,
                    row[cfg.timestamp_column],
                    raw_values,
                )

            # ----------------------------------------------------------------
            # Build typed quantile object
            # ----------------------------------------------------------------

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

            # ----------------------------------------------------------------
            # Explicitly use configured point quantile.
            # ----------------------------------------------------------------

            point_quantile_name = (
                f"p{int(cfg.point_quantile * 100)}"
            )

            try:

                point = float(
                    getattr(
                        quantiles,
                        point_quantile_name,
                    )
                )

            except AttributeError as exc:

                raise ValueError(
                    f"QuantileForecast does not contain "
                    f"configured point quantile "
                    f"{point_quantile_name!r}."
                ) from exc

            # ----------------------------------------------------------------
            # Forecast date
            # ----------------------------------------------------------------

            forecast_date = pd.Timestamp(
                row[cfg.timestamp_column]
            ).normalize()

            days.append(
                ForecastDayResult(
                    forecast_date=forecast_date.date(),
                    predicted_demand=round(
                        point,
                        2,
                    ),
                    quantiles=quantiles,
                )
            )

        # --------------------------------------------------------------------
        # Final horizon validation
        # --------------------------------------------------------------------

        if len(days) != (
            cfg.prediction_length
        ):

            raise ValueError(
                f"Result conversion produced {len(days)} "
                f"forecast day(s); expected "
                f"{cfg.prediction_length}."
            )

        return MedicineForecastResult(
            medicine_id=str(
                item_id
            ),
            generated_at=datetime.now(
                timezone.utc
            ),
            context_length_used=int(
                context_used
            ),
            prediction_length=int(
                cfg.prediction_length
            ),
            model_id=cfg.model_id,
            days=days,
        )