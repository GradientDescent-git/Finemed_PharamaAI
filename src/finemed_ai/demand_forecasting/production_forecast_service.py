from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional

import numpy as np
import pandas as pd

from finemed_ai.demand_forecasting.config import (
    DEFAULT_CONFIG,
    ForecastConfig,
)
from finemed_ai.demand_forecasting.predictor_service import (
    InsufficientHistoryError,
    PredictorService,
)
from finemed_ai.demand_forecasting.production_forecast_router import (
    ProductionForecastRouter,
)


logger = logging.getLogger(__name__)


# ============================================================================
# Constants
# ============================================================================

MEDICINE_ID_COLUMN = "Medicine_ID"
TIMESTAMP_COLUMN = "timestamp"
TARGET_COLUMN = "target"

MODEL_CHRONOS = "chronos-2-P50"
MODEL_TSB = "tsb"

ROUTING_THRESHOLD_PCT = 30.0

REQUIRED_ROUTING_COLUMNS = {
    MEDICINE_ID_COLUMN,
    "Validation_Advantage_Pct",
}

REQUIRED_HISTORY_COLUMNS = {
    MEDICINE_ID_COLUMN,
    TIMESTAMP_COLUMN,
    TARGET_COLUMN,
}


# ============================================================================
# Result schema
# ============================================================================

@dataclass
class ProductionForecastResult:
    """
    Unified production forecast result.

    Chronos-selected medicines contain P10/P50/P90.

    TSB-selected medicines do not have probabilistic quantiles, so
    p10/p50/p90 remain None.
    """

    medicine_id: str

    selected_model: str

    routing_advantage_pct: float
    routing_reason: str

    forecast_dates: List[pd.Timestamp]
    predicted_demand: List[float]

    p10: Optional[List[float]]
    p50: Optional[List[float]]
    p90: Optional[List[float]]

    context_length_used: int
    prediction_length: int

    generated_at: datetime


# ============================================================================
# TSB
# ============================================================================

def tsb_forecast(
    history: pd.Series,
    horizon: int,
    alpha_demand: float = 0.1,
    alpha_probability: float = 0.1,
) -> np.ndarray:
    """
    Teunter-Syntetos-Babai forecast.

    This matches the validated TSB formulation used during
    model comparison/backtesting.
    """

    if horizon <= 0:
        raise ValueError(
            f"horizon must be positive, got {horizon}"
        )

    if not 0.0 < alpha_demand <= 1.0:
        raise ValueError(
            "alpha_demand must be in (0, 1]."
        )

    if not 0.0 < alpha_probability <= 1.0:
        raise ValueError(
            "alpha_probability must be in (0, 1]."
        )

    if not isinstance(history, pd.Series):
        raise TypeError(
            "history must be a pandas Series."
        )

    y = history.to_numpy(dtype=float)

    if len(y) == 0:
        return np.zeros(horizon, dtype=float)

    if not np.isfinite(y).all():
        raise ValueError(
            "TSB history contains non-finite values."
        )

    y = np.maximum(y, 0.0)

    non_zero = np.flatnonzero(y > 0)

    if len(non_zero) == 0:
        return np.zeros(horizon, dtype=float)

    first = int(non_zero[0])

    demand_estimate = float(y[first])

    probability = 1.0 / float(first + 1)

    for demand in y:

        occurrence = (
            1.0
            if demand > 0
            else 0.0
        )

        probability += (
            alpha_probability
            * (occurrence - probability)
        )

        if occurrence > 0:

            demand_estimate += (
                alpha_demand
                * (
                    float(demand)
                    - demand_estimate
                )
            )

    forecast_value = (
        probability
        * demand_estimate
    )

    forecast_value = max(
        float(forecast_value),
        0.0,
    )

    if not np.isfinite(forecast_value):
        raise ValueError(
            "TSB produced a non-finite forecast."
        )

    return np.full(
        horizon,
        forecast_value,
        dtype=float,
    )


# ============================================================================
# Production service
# ============================================================================

class ProductionForecastService:
    """
    Production forecasting orchestration layer.

    Responsibilities:

    1. Validate source history.
    2. Validate routing policy.
    3. Normalize transaction history into a daily calendar.
    4. Apply frozen model routing.
    5. Execute Chronos-2 P50.
    6. Execute TSB.
    7. Normalize results into one production schema.
    8. Isolate medicine-level failures.
    9. Validate final production output.
    """

    def __init__(
        self,
        router: ProductionForecastRouter,
        forecast_config: ForecastConfig = DEFAULT_CONFIG,
        predictor: Optional[PredictorService] = None,
    ):
        if router is None:
            raise ValueError(
                "router must not be None."
            )

        if forecast_config is None:
            raise ValueError(
                "forecast_config must not be None."
            )

        self.router = router
        self.config = forecast_config

        self.predictor = (
            predictor
            if predictor is not None
            else PredictorService.get_instance(
                forecast_config
            )
        )

    # ========================================================================
    # Input validation
    # ========================================================================

    @staticmethod
    def _validate_history(
        history_df: pd.DataFrame,
    ) -> None:

        if not isinstance(history_df, pd.DataFrame):
            raise TypeError(
                "history_df must be a pandas DataFrame."
            )

        missing = (
            REQUIRED_HISTORY_COLUMNS
            - set(history_df.columns)
        )

        if missing:
            raise ValueError(
                "History dataframe is missing required columns: "
                f"{sorted(missing)}"
            )

        if history_df.empty:
            raise ValueError(
                "History dataframe is empty."
            )

        if history_df[MEDICINE_ID_COLUMN].isna().any():
            raise ValueError(
                "History contains null Medicine_ID values."
            )

        medicine_ids = (
            history_df[MEDICINE_ID_COLUMN]
            .astype(str)
            .str.strip()
        )

        if medicine_ids.eq("").any():
            raise ValueError(
                "History contains empty Medicine_ID values."
            )

        parsed_timestamps = pd.to_datetime(
            history_df[TIMESTAMP_COLUMN],
            errors="coerce",
        )

        if parsed_timestamps.isna().any():
            raise ValueError(
                f"{TIMESTAMP_COLUMN} contains invalid timestamps."
            )

        target_values = pd.to_numeric(
            history_df[TARGET_COLUMN],
            errors="coerce",
        )

        if target_values.isna().any():
            raise ValueError(
                f"{TARGET_COLUMN} contains non-numeric values."
            )

        target_array = target_values.to_numpy(
            dtype=float
        )

        if not np.isfinite(target_array).all():
            raise ValueError(
                f"{TARGET_COLUMN} contains non-finite values."
            )

        if (target_values < 0).any():
            raise ValueError(
                f"{TARGET_COLUMN} contains negative demand."
            )

    @staticmethod
    def _validate_routing_table(
        routing_table: pd.DataFrame,
    ) -> None:

        if not isinstance(
            routing_table,
            pd.DataFrame,
        ):
            raise TypeError(
                "routing_table must be a pandas DataFrame."
            )

        missing = (
            REQUIRED_ROUTING_COLUMNS
            - set(routing_table.columns)
        )

        if missing:
            raise ValueError(
                "Routing table is missing required columns: "
                f"{sorted(missing)}"
            )

        if routing_table.empty:
            raise ValueError(
                "Routing table is empty."
            )

        if routing_table[MEDICINE_ID_COLUMN].isna().any():
            raise ValueError(
                "Routing table contains null Medicine_ID values."
            )

        medicine_ids = (
            routing_table[MEDICINE_ID_COLUMN]
            .astype(str)
            .str.strip()
        )

        if medicine_ids.eq("").any():
            raise ValueError(
                "Routing table contains empty Medicine_ID values."
            )

        duplicate_ids = medicine_ids.duplicated()

        if duplicate_ids.any():

            duplicated = (
                medicine_ids[
                    duplicate_ids
                ]
                .tolist()
            )

            raise ValueError(
                "Routing table contains duplicate Medicine_ID values: "
                f"{duplicated}"
            )

        advantages = pd.to_numeric(
            routing_table[
                "Validation_Advantage_Pct"
            ],
            errors="coerce",
        )

        finite_advantages = advantages.dropna()

        if not np.isfinite(
            finite_advantages.to_numpy(
                dtype=float
            )
        ).all():
            raise ValueError(
                "Routing table contains non-finite "
                "Validation_Advantage_Pct values."
            )

    # ========================================================================
    # Daily history normalization
    # ========================================================================

    @staticmethod
    def _prepare_daily_history(
        medicine_id: str,
        history_df: pd.DataFrame,
    ) -> pd.DataFrame:

        medicine_id = str(
            medicine_id
        ).strip()

        if not medicine_id:
            raise ValueError(
                "medicine_id must not be empty."
            )

        normalized_ids = (
            history_df[MEDICINE_ID_COLUMN]
            .astype(str)
            .str.strip()
        )

        history = history_df[
            normalized_ids == medicine_id
        ].copy()

        if history.empty:
            raise InsufficientHistoryError(
                f"No history for item_id={medicine_id}"
            )

        history[TIMESTAMP_COLUMN] = pd.to_datetime(
            history[TIMESTAMP_COLUMN],
            errors="raise",
        )

        history[TARGET_COLUMN] = pd.to_numeric(
            history[TARGET_COLUMN],
            errors="raise",
        )

        history[MEDICINE_ID_COLUMN] = (
            history[MEDICINE_ID_COLUMN]
            .astype(str)
            .str.strip()
        )

        history[TIMESTAMP_COLUMN] = (
            history[TIMESTAMP_COLUMN]
            .dt.normalize()
        )

        # Aggregate same-day transactions.
        history = (
            history
            .groupby(
                [
                    MEDICINE_ID_COLUMN,
                    TIMESTAMP_COLUMN,
                ],
                as_index=False,
            )[TARGET_COLUMN]
            .sum()
        )

        history = history.sort_values(
            TIMESTAMP_COLUMN
        )

        start_date = history[TIMESTAMP_COLUMN].min()
        end_date = history[TIMESTAMP_COLUMN].max()

        if pd.isna(start_date) or pd.isna(end_date):
            raise ValueError(
                f"Invalid date range for medicine={medicine_id}."
            )

        if end_date < start_date:
            raise ValueError(
                f"Invalid history date range for medicine="
                f"{medicine_id}: {start_date} > {end_date}"
            )

        # Continuous daily calendar.
        calendar = pd.date_range(
            start=start_date,
            end=end_date,
            freq="D",
        )

        daily = (
            history
            .set_index(TIMESTAMP_COLUMN)[TARGET_COLUMN]
            .reindex(
                calendar,
                fill_value=0.0,
            )
            .rename(TARGET_COLUMN)
            .rename_axis(TIMESTAMP_COLUMN)
            .reset_index()
        )

        daily[MEDICINE_ID_COLUMN] = medicine_id

        daily = daily[
            [
                MEDICINE_ID_COLUMN,
                TIMESTAMP_COLUMN,
                TARGET_COLUMN,
            ]
        ]

        daily[TARGET_COLUMN] = (
            pd.to_numeric(
                daily[TARGET_COLUMN],
                errors="raise",
            )
            .fillna(0.0)
            .clip(lower=0.0)
        )

        expected_rows = (
            end_date - start_date
        ).days + 1

        if len(daily) != expected_rows:
            raise ValueError(
                f"Daily calendar construction failed for "
                f"medicine={medicine_id}: expected "
                f"{expected_rows} rows, got {len(daily)}."
            )

        if daily[TIMESTAMP_COLUMN].duplicated().any():
            raise ValueError(
                f"Duplicate dates remain after normalization "
                f"for medicine={medicine_id}."
            )

        if len(daily) >= 3:

            inferred_freq = pd.infer_freq(
                daily[TIMESTAMP_COLUMN]
            )

            if inferred_freq not in {"D", "1D"}:
                raise ValueError(
                    f"Daily frequency validation failed for "
                    f"medicine={medicine_id}: "
                    f"inferred_freq={inferred_freq!r}"
                )

        demand_array = daily[
            TARGET_COLUMN
        ].to_numpy(dtype=float)

        if not np.isfinite(demand_array).all():
            raise ValueError(
                f"Daily history contains non-finite demand "
                f"for medicine={medicine_id}."
            )

        logger.info(
            "DAILY HISTORY | medicine=%s | rows=%d | "
            "start=%s | end=%s | zero_days=%d",
            medicine_id,
            len(daily),
            start_date,
            end_date,
            int(
                (
                    daily[TARGET_COLUMN] == 0
                ).sum()
            ),
        )

        return daily

    # ========================================================================
    # Routing
    # ========================================================================

    def _get_route(
        self,
        medicine_id: str,
        routing_table: pd.DataFrame,
    ) -> tuple[str, float, str]:

        medicine_id = str(
            medicine_id
        ).strip()

        normalized_ids = (
            routing_table[MEDICINE_ID_COLUMN]
            .astype(str)
            .str.strip()
        )

        rows = routing_table[
            normalized_ids == medicine_id
        ]

        # IMPORTANT:
        # The routing table contains 79 medicines, while production
        # history currently contains 158 medicines.
        #
        # Missing routing records intentionally fall back to TSB.
        if rows.empty:

            logger.warning(
                "No routing record for medicine=%s. "
                "Falling back to TSB.",
                medicine_id,
            )

            return (
                MODEL_TSB,
                float("nan"),
                "missing_routing_record",
            )

        row = rows.iloc[0]

        try:

            advantage = float(
                row["Validation_Advantage_Pct"]
            )

        except (
            TypeError,
            ValueError,
        ):

            advantage = float("nan")

        if not np.isfinite(advantage):

            logger.warning(
                "Invalid routing advantage for medicine=%s. "
                "Falling back to TSB.",
                medicine_id,
            )

            return (
                MODEL_TSB,
                float("nan"),
                "invalid_or_missing_advantage",
            )

        selected_model = self.router.select_model(
            advantage
        )

        if selected_model == MODEL_CHRONOS:

            reason = (
                "validation_advantage_ge_30pct"
            )

        elif selected_model == MODEL_TSB:

            reason = (
                "validation_advantage_lt_30pct"
            )

        else:

            raise ValueError(
                "Production router returned unsupported "
                f"model {selected_model!r} "
                f"for medicine={medicine_id}."
            )

        return (
            selected_model,
            advantage,
            reason,
        )

    # ========================================================================
    # Chronos
    # ========================================================================

    def _forecast_chronos(
        self,
        medicine_id: str,
        history_df: pd.DataFrame,
    ) -> ProductionForecastResult:

        predictor_config = self.predictor.config

        chronos_history = history_df.copy()

        rename_map = {}

        if (
            MEDICINE_ID_COLUMN
            != predictor_config.id_column
        ):
            rename_map[
                MEDICINE_ID_COLUMN
            ] = predictor_config.id_column

        if (
            TIMESTAMP_COLUMN
            != predictor_config.timestamp_column
        ):
            rename_map[
                TIMESTAMP_COLUMN
            ] = predictor_config.timestamp_column

        if (
            TARGET_COLUMN
            != predictor_config.target_column
        ):
            rename_map[
                TARGET_COLUMN
            ] = predictor_config.target_column

        if rename_map:

            chronos_history = (
                chronos_history.rename(
                    columns=rename_map
                )
            )

        required_predictor_columns = {
            predictor_config.id_column,
            predictor_config.timestamp_column,
            predictor_config.target_column,
        }

        missing = (
            required_predictor_columns
            - set(chronos_history.columns)
        )

        if missing:
            raise ValueError(
                "Chronos history adapter produced "
                "missing columns: "
                f"{sorted(missing)}"
            )

        result = self.predictor.forecast_medicine(
            medicine_id,
            chronos_history,
        )

        expected_horizon = (
            self.config.prediction_length
        )

        if len(result.days) != expected_horizon:
            raise ValueError(
                f"Chronos returned {len(result.days)} forecast days "
                f"for medicine={medicine_id}; "
                f"expected {expected_horizon}."
            )

        dates = [
            pd.Timestamp(day.forecast_date)
            for day in result.days
        ]

        predicted = [
            float(day.predicted_demand)
            for day in result.days
        ]

        p10 = [
            float(day.quantiles.p10)
            for day in result.days
        ]

        p50 = [
            float(day.quantiles.p50)
            for day in result.days
        ]

        p90 = [
            float(day.quantiles.p90)
            for day in result.days
        ]

        for name, values in {
            "predicted_demand": predicted,
            "p10": p10,
            "p50": p50,
            "p90": p90,
        }.items():

            array = np.asarray(
                values,
                dtype=float,
            )

            if not np.isfinite(array).all():
                raise ValueError(
                    f"Chronos returned non-finite {name} "
                    f"for medicine={medicine_id}."
                )

            if (array < 0).any():
                raise ValueError(
                    f"Chronos returned negative {name} "
                    f"for medicine={medicine_id}."
                )

        if any(
            low > median
            for low, median in zip(p10, p50)
        ):
            raise ValueError(
                f"Chronos P10 exceeds P50 "
                f"for medicine={medicine_id}."
            )

        if any(
            median > high
            for median, high in zip(p50, p90)
        ):
            raise ValueError(
                f"Chronos P50 exceeds P90 "
                f"for medicine={medicine_id}."
            )

        if len(set(dates)) != len(dates):
            raise ValueError(
                f"Chronos returned duplicate forecast dates "
                f"for medicine={medicine_id}."
            )

        if dates != sorted(dates):
            raise ValueError(
                f"Chronos forecast dates are not sorted "
                f"for medicine={medicine_id}."
            )

        return ProductionForecastResult(
            medicine_id=str(medicine_id),
            selected_model=MODEL_CHRONOS,
            routing_advantage_pct=float("nan"),
            routing_reason="",
            forecast_dates=dates,
            predicted_demand=predicted,
            p10=p10,
            p50=p50,
            p90=p90,
            context_length_used=(
                result.context_length_used
            ),
            prediction_length=(
                result.prediction_length
            ),
            generated_at=result.generated_at,
        )

    # ========================================================================
    # TSB
    # ========================================================================

    def _forecast_tsb(
        self,
        medicine_id: str,
        history_df: pd.DataFrame,
    ) -> ProductionForecastResult:

        normalized_ids = (
            history_df[MEDICINE_ID_COLUMN]
            .astype(str)
            .str.strip()
        )

        history = (
            history_df[
                normalized_ids
                == str(medicine_id).strip()
            ]
            .sort_values(TIMESTAMP_COLUMN)
            .copy()
        )

        if history.empty:
            raise InsufficientHistoryError(
                f"No history for item_id={medicine_id}"
            )

        target = (
            history[TARGET_COLUMN]
            .astype(float)
            .clip(lower=0.0)
        )

        horizon = self.config.prediction_length

        prediction = tsb_forecast(
            target,
            horizon,
        )

        if len(prediction) != horizon:
            raise ValueError(
                f"TSB returned {len(prediction)} values "
                f"for medicine={medicine_id}; "
                f"expected {horizon}."
            )

        if not np.isfinite(prediction).all():
            raise ValueError(
                f"TSB returned non-finite predictions "
                f"for medicine={medicine_id}."
            )

        last_date = pd.Timestamp(
            history[TIMESTAMP_COLUMN].max()
        ).normalize()

        forecast_dates = list(
            pd.date_range(
                start=(
                    last_date
                    + pd.Timedelta(days=1)
                ),
                periods=horizon,
                freq="D",
            )
        )

        prediction = np.maximum(
            prediction,
            0.0,
        )

        return ProductionForecastResult(
            medicine_id=str(medicine_id),
            selected_model=MODEL_TSB,
            routing_advantage_pct=float("nan"),
            routing_reason="",
            forecast_dates=forecast_dates,
            predicted_demand=[
                round(float(x), 2)
                for x in prediction
            ],
            p10=None,
            p50=None,
            p90=None,
            context_length_used=len(history),
            prediction_length=horizon,
            generated_at=datetime.now(timezone.utc),
        )

    # ========================================================================
    # Single medicine
    # ========================================================================

    def forecast_medicine(
        self,
        medicine_id: str,
        history_df: pd.DataFrame,
        routing_table: pd.DataFrame,
    ) -> ProductionForecastResult:

        self._validate_history(history_df)
        self._validate_routing_table(routing_table)

        medicine_id = str(
            medicine_id
        ).strip()

        if not medicine_id:
            raise ValueError(
                "medicine_id must not be empty."
            )

        daily_history = (
            self._prepare_daily_history(
                medicine_id,
                history_df,
            )
        )

        (
            selected_model,
            advantage,
            reason,
        ) = self._get_route(
            medicine_id,
            routing_table,
        )

        logger.info(
            "PRODUCTION ROUTE | medicine=%s | model=%s | "
            "advantage=%s | reason=%s",
            medicine_id,
            selected_model,
            advantage,
            reason,
        )

        if selected_model == MODEL_CHRONOS:

            result = self._forecast_chronos(
                medicine_id,
                daily_history,
            )

        elif selected_model == MODEL_TSB:

            result = self._forecast_tsb(
                medicine_id,
                daily_history,
            )

        else:

            raise ValueError(
                f"Unsupported production model: "
                f"{selected_model!r}"
            )

        result.routing_advantage_pct = advantage
        result.routing_reason = reason

        return result

    # ========================================================================
    # Batch forecasting
    # ========================================================================

    def forecast_batch(
        self,
        history_df: pd.DataFrame,
        routing_table: pd.DataFrame,
        item_ids: Optional[List[str]] = None,
    ) -> tuple[pd.DataFrame, List[str]]:

        self._validate_history(history_df)
        self._validate_routing_table(routing_table)

        history_ids = set(
            history_df[MEDICINE_ID_COLUMN]
            .astype(str)
            .str.strip()
            .unique()
        )

        if item_ids is None:

            # Production default:
            # forecast every medicine in Silver history.
            ids = sorted(history_ids)

        else:

            ids = [
                str(x).strip()
                for x in item_ids
                if str(x).strip()
            ]

            ids = list(dict.fromkeys(ids))

            unknown_ids = [
                x
                for x in ids
                if x not in history_ids
            ]

            if unknown_ids:
                logger.warning(
                    "Requested medicine IDs not found in history: %s",
                    unknown_ids[:20],
                )

            ids = [
                x
                for x in ids
                if x in history_ids
            ]

        if not ids:
            raise ValueError(
                "No medicine IDs available for production forecasting."
            )

        logger.info(
            "Starting production forecast batch: %d medicines",
            len(ids),
        )

        records = []
        failed = []

        for medicine_id in ids:

            try:

                result = self.forecast_medicine(
                    medicine_id,
                    history_df,
                    routing_table,
                )

                horizon = result.prediction_length

                if len(result.forecast_dates) != horizon:
                    raise ValueError(
                        f"Forecast date count mismatch "
                        f"for medicine={medicine_id}."
                    )

                if len(result.predicted_demand) != horizon:
                    raise ValueError(
                        f"Prediction count mismatch "
                        f"for medicine={medicine_id}."
                    )

                if result.selected_model == MODEL_CHRONOS:

                    if (
                        result.p10 is None
                        or result.p50 is None
                        or result.p90 is None
                    ):
                        raise ValueError(
                            f"Chronos result missing quantiles "
                            f"for medicine={medicine_id}."
                        )

                    if not (
                        len(result.p10)
                        == len(result.p50)
                        == len(result.p90)
                        == horizon
                    ):
                        raise ValueError(
                            f"Chronos quantile length mismatch "
                            f"for medicine={medicine_id}."
                        )

                elif result.selected_model == MODEL_TSB:

                    if any(
                        value is not None
                        for value in (
                            result.p10,
                            result.p50,
                            result.p90,
                        )
                    ):
                        raise ValueError(
                            f"TSB result unexpectedly contains "
                            f"quantiles for medicine={medicine_id}."
                        )

                else:

                    raise ValueError(
                        f"Unsupported result model "
                        f"{result.selected_model!r} "
                        f"for medicine={medicine_id}."
                    )

                for i, forecast_date in enumerate(
                    result.forecast_dates
                ):

                    records.append(
                        {
                            MEDICINE_ID_COLUMN:
                                result.medicine_id,

                            "Forecast_Date":
                                forecast_date,

                            "Selected_Model":
                                result.selected_model,

                            "Predicted_Demand":
                                result.predicted_demand[i],

                            "P10":
                                (
                                    result.p10[i]
                                    if result.p10 is not None
                                    else np.nan
                                ),

                            "P50":
                                (
                                    result.p50[i]
                                    if result.p50 is not None
                                    else np.nan
                                ),

                            "P90":
                                (
                                    result.p90[i]
                                    if result.p90 is not None
                                    else np.nan
                                ),

                            "Routing_Advantage_Pct":
                                result.routing_advantage_pct,

                            "Routing_Reason":
                                result.routing_reason,

                            "Context_Length_Used":
                                result.context_length_used,

                            "Prediction_Length":
                                result.prediction_length,

                            "Generated_At":
                                result.generated_at,
                        }
                    )

            except InsufficientHistoryError as exc:

                logger.warning(
                    "Skipping medicine=%s: %s",
                    medicine_id,
                    exc,
                )

                failed.append(medicine_id)

            except Exception:

                logger.exception(
                    "Production forecast failed "
                    "for medicine=%s",
                    medicine_id,
                )

                failed.append(medicine_id)

        if not records:
            raise RuntimeError(
                "Production forecast generated "
                "no successful results."
            )

        result_df = pd.DataFrame(records)

        result_df = (
            result_df
            .sort_values(
                [
                    MEDICINE_ID_COLUMN,
                    "Forecast_Date",
                ]
            )
            .reset_index(drop=True)
        )

        self._validate_output(result_df)

        logger.info(
            "Production forecast complete | "
            "successful=%d | failed=%d | rows=%d",
            result_df[
                MEDICINE_ID_COLUMN
            ].nunique(),
            len(failed),
            len(result_df),
        )

        return result_df, failed

    # ========================================================================
    # Output validation
    # ========================================================================

    def _validate_output(
        self,
        result_df: pd.DataFrame,
    ) -> None:

        if not isinstance(
            result_df,
            pd.DataFrame,
        ):
            raise TypeError(
                "result_df must be a pandas DataFrame."
            )

        required = {
            MEDICINE_ID_COLUMN,
            "Forecast_Date",
            "Selected_Model",
            "Predicted_Demand",
            "Routing_Advantage_Pct",
            "Routing_Reason",
            "Context_Length_Used",
            "Prediction_Length",
            "Generated_At",
        }

        missing = required - set(result_df.columns)

        if missing:
            raise ValueError(
                "Production output missing columns: "
                f"{sorted(missing)}"
            )

        if result_df.empty:
            raise ValueError(
                "Production output is empty."
            )

        # Medicine IDs
        if result_df[MEDICINE_ID_COLUMN].isna().any():
            raise ValueError(
                "Production output contains null Medicine_ID values."
            )

        output_ids = (
            result_df[MEDICINE_ID_COLUMN]
            .astype(str)
            .str.strip()
        )

        if output_ids.eq("").any():
            raise ValueError(
                "Production output contains empty Medicine_ID values."
            )

        # Predicted demand
        predicted = pd.to_numeric(
            result_df["Predicted_Demand"],
            errors="coerce",
        )

        if predicted.isna().any():
            raise ValueError(
                "Production output contains null/non-numeric "
                "predicted demand."
            )

        if not np.isfinite(
            predicted.to_numpy(dtype=float)
        ).all():
            raise ValueError(
                "Production output contains non-finite "
                "predicted demand."
            )

        if (predicted < 0).any():
            raise ValueError(
                "Production output contains negative demand."
            )

        # Models
        invalid_models = (
            set(result_df["Selected_Model"].unique())
            - {
                MODEL_CHRONOS,
                MODEL_TSB,
            }
        )

        if invalid_models:
            raise ValueError(
                "Unexpected production models: "
                f"{sorted(invalid_models)}"
            )

        # Dates
        forecast_dates = pd.to_datetime(
            result_df["Forecast_Date"],
            errors="coerce",
        )

        if forecast_dates.isna().any():
            raise ValueError(
                "Production output contains invalid Forecast_Date values."
            )

        duplicate_dates = result_df.duplicated(
            subset=[
                MEDICINE_ID_COLUMN,
                "Forecast_Date",
            ],
            keep=False,
        )

        if duplicate_dates.any():
            raise ValueError(
                "Production output contains duplicate "
                "Medicine_ID + Forecast_Date combinations."
            )

        # Chronos
        chronos = result_df[
            result_df["Selected_Model"]
            == MODEL_CHRONOS
        ]

        if not chronos.empty:

            quantiles = chronos[
                ["P10", "P50", "P90"]
            ].apply(
                pd.to_numeric,
                errors="coerce",
            )

            if quantiles.isna().any().any():
                raise ValueError(
                    "Chronos forecasts must contain numeric "
                    "P10, P50 and P90."
                )

            if not np.isfinite(
                quantiles.to_numpy(dtype=float)
            ).all():
                raise ValueError(
                    "Chronos forecasts contain "
                    "non-finite quantiles."
                )

            if (quantiles < 0).any().any():
                raise ValueError(
                    "Chronos forecasts contain "
                    "negative quantiles."
                )

            if (
                chronos["P10"]
                > chronos["P50"]
            ).any():
                raise ValueError(
                    "Chronos P10 exceeds P50."
                )

            if (
                chronos["P50"]
                > chronos["P90"]
            ).any():
                raise ValueError(
                    "Chronos P50 exceeds P90."
                )

        # TSB
        tsb = result_df[
            result_df["Selected_Model"]
            == MODEL_TSB
        ]

        if not tsb.empty:

            if tsb[
                ["P10", "P50", "P90"]
            ].notna().any().any():

                raise ValueError(
                    "TSB forecasts must not contain "
                    "P10/P50/P90 values."
                )

        # Horizon
        horizon_counts = (
            result_df
            .groupby(MEDICINE_ID_COLUMN)
            .size()
        )

        invalid_horizon = (
            horizon_counts
            != self.config.prediction_length
        )

        if invalid_horizon.any():

            bad = (
                horizon_counts[
                    invalid_horizon
                ]
                .to_dict()
            )

            raise ValueError(
                "Invalid forecast horizon for medicines: "
                f"{bad}"
            )

        # Prediction length
        prediction_lengths = pd.to_numeric(
            result_df["Prediction_Length"],
            errors="coerce",
        )

        if prediction_lengths.isna().any():
            raise ValueError(
                "Prediction_Length contains invalid values."
            )

        if (
            prediction_lengths
            != self.config.prediction_length
        ).any():
            raise ValueError(
                "Production output contains unexpected "
                "Prediction_Length values."
            )

        # Context length
        context_lengths = pd.to_numeric(
            result_df["Context_Length_Used"],
            errors="coerce",
        )

        if context_lengths.isna().any():
            raise ValueError(
                "Context_Length_Used contains invalid values."
            )

        if (context_lengths <= 0).any():
            raise ValueError(
                "Context_Length_Used must be positive."
            )

        # Routing advantage
        advantages = pd.to_numeric(
            result_df["Routing_Advantage_Pct"],
            errors="coerce",
        )

        finite_advantages = advantages.dropna()

        if not np.isfinite(
            finite_advantages.to_numpy(dtype=float)
        ).all():
            raise ValueError(
                "Routing_Advantage_Pct contains "
                "non-finite values."
            )

        # Chronos routing consistency
        chronos_advantages = pd.to_numeric(
            chronos["Routing_Advantage_Pct"],
            errors="coerce",
        ).dropna()

        if (
            not chronos_advantages.empty
            and (
                chronos_advantages
                < ROUTING_THRESHOLD_PCT
            ).any()
        ):
            raise ValueError(
                "Chronos production forecasts contain routing "
                "advantages below the frozen "
                f"{ROUTING_THRESHOLD_PCT:.0f}% threshold."
            )

        # TSB routing consistency
        tsb_advantages = pd.to_numeric(
            tsb["Routing_Advantage_Pct"],
            errors="coerce",
        ).dropna()

        if (
            not tsb_advantages.empty
            and (
                tsb_advantages
                >= ROUTING_THRESHOLD_PCT
            ).any()
        ):
            raise ValueError(
                "TSB production forecasts contain routing "
                "advantages at or above the frozen "
                f"{ROUTING_THRESHOLD_PCT:.0f}% threshold."
            )


# ============================================================================
# Smoke test
# ============================================================================

def main() -> None:

    print("=" * 80)
    print("PRODUCTION FORECAST SERVICE")
    print("=" * 80)

    print()
    print("ProductionForecastService import successful.")
    print()

    print("Frozen routing policy:")
    print(
        "Chronos-2 P50 if validation advantage >= "
        f"{ROUTING_THRESHOLD_PCT:.0f}%"
    )
    print("Otherwise: TSB")

    print()
    print("TSB parameters:")
    print("alpha_demand=0.1")
    print("alpha_probability=0.1")

    print()
    print("Forecast configuration:")
    print(
        f"prediction_length="
        f"{DEFAULT_CONFIG.prediction_length}"
    )
    print(
        f"context_length="
        f"{DEFAULT_CONFIG.context_length}"
    )

    print()
    print("ProductionForecastService is ready.")


if __name__ == "__main__":
    main()