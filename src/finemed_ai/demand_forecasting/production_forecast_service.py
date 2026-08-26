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
# CONSTANTS
# ============================================================================

MEDICINE_ID_COLUMN = "Medicine_ID"
TIMESTAMP_COLUMN = "timestamp"
TARGET_COLUMN = "target"

SELECTED_MODEL_COLUMN = "Selected_Model"
ROUTING_RULE_COLUMN = "Routing_Rule"
ROUTING_REASON_COLUMN = "Routing_Reason"
VALIDATION_ADVANTAGE_COLUMN = "Validation_Advantage_Pct"

MODEL_CHRONOS = "chronos-2-P50"
MODEL_TSB = "tsb"

REQUIRED_HISTORY_COLUMNS = {
    MEDICINE_ID_COLUMN,
    TIMESTAMP_COLUMN,
    TARGET_COLUMN,
}

REQUIRED_ROUTING_COLUMNS = {
    MEDICINE_ID_COLUMN,
    SELECTED_MODEL_COLUMN,
    ROUTING_RULE_COLUMN,
    ROUTING_REASON_COLUMN,
    VALIDATION_ADVANTAGE_COLUMN,
}

OUTPUT_COLUMNS = [
    MEDICINE_ID_COLUMN,
    "Forecast_Date",
    SELECTED_MODEL_COLUMN,
    "Predicted_Demand",
    "P10",
    "P50",
    "P90",
    VALIDATION_ADVANTAGE_COLUMN,
    ROUTING_REASON_COLUMN,
    "Context_Length_Used",
    "Prediction_Length",
    "Generated_At",
]


# ============================================================================
# RESULT SCHEMA
# ============================================================================

@dataclass
class ProductionForecastResult:
    """
    Unified production forecast result.

    Chronos-selected medicines contain P10/P50/P90.

    For Chronos-2-P50:

        predicted_demand == p50

    TSB-selected medicines do not expose probabilistic quantiles.

    validation_advantage_pct is NaN when the routing table explicitly
    contains a non-validation fallback medicine.
    """

    medicine_id: str

    selected_model: str

    validation_advantage_pct: float
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
# TSB FORECAST
# ============================================================================

def tsb_forecast(
    history: pd.Series,
    horizon: int,
    alpha_demand: float = 0.1,
    alpha_probability: float = 0.1,
) -> np.ndarray:
    """
    Teunter-Syntetos-Babai forecast.

    This implementation must remain aligned with the validated TSB
    formulation used during model comparison and production routing.
    """

    if not isinstance(history, pd.Series):
        raise TypeError(
            "history must be a pandas Series."
        )

    if horizon <= 0:
        raise ValueError(
            f"horizon must be positive, got {horizon}."
        )

    if not 0.0 < alpha_demand <= 1.0:
        raise ValueError(
            "alpha_demand must be in (0, 1]."
        )

    if not 0.0 < alpha_probability <= 1.0:
        raise ValueError(
            "alpha_probability must be in (0, 1]."
        )

    y = history.to_numpy(
        dtype=float
    )

    if len(y) == 0:
        return np.zeros(
            horizon,
            dtype=float,
        )

    if not np.isfinite(y).all():
        raise ValueError(
            "TSB history contains non-finite values."
        )

    y = np.maximum(y,0.0)

    non_zero = np.flatnonzero(
        y > 0
    )

    if len(non_zero) == 0:
        return np.zeros(
            horizon,
            dtype=float,
        )

    first_non_zero = int(
        non_zero[0]
    )

    demand_estimate = float(
        y[first_non_zero]
    )

    probability = (
        1.0
        / float(first_non_zero + 1)
    )

    for demand in y:

        occurrence = (
            1.0
            if demand > 0
            else 0.0
        )

        probability += (
            alpha_probability
            * (
                occurrence
                - probability
            )
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

    if not np.isfinite(
        forecast_value
    ):
        raise ValueError(
            "TSB produced a non-finite forecast."
        )

    return np.full(
        horizon,
        forecast_value,
        dtype=float,
    )


# ============================================================================
# PRODUCTION FORECAST SERVICE
# ============================================================================

class ProductionForecastService:
    """
    Production demand forecasting orchestration layer.

    Responsibilities:

    1. Validate source history.
    2. Validate production routing table.
    3. Build continuous daily medicine history.
    4. Preserve the frozen routing decision.
    5. Execute Chronos-2 P50.
    6. Execute TSB.
    7. Enforce production forecast invariants.
    8. Isolate medicine-level failures.
    9. Return one normalized production output schema.

    IMPORTANT ARCHITECTURAL RULE
    ----------------------------

    The production routing table is the source of truth.

    For an existing routing record, this service MUST NOT recompute:

        - Selected_Model
        - Routing_Rule
        - Routing_Reason
        - Validation_Advantage_Pct

    This is essential because some medicines legitimately have:

        Validation_Advantage_Pct = NaN

    while still having an explicit routing decision such as:

        Selected_Model = "tsb"
        Routing_Rule = "fallback_to_tsb_when_no_validation_record"
        Routing_Reason = "not_in_validation_population"

    Missing routing records are the only case where this service creates
    its own fallback provenance.
    """

    def __init__(
        self,
        router: ProductionForecastRouter,
        forecast_config: ForecastConfig = DEFAULT_CONFIG,
        predictor: Optional[PredictorService] = None,
    ) -> None:

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
    # INPUT VALIDATION
    # ========================================================================

    @staticmethod
    def _validate_history(
        history_df: pd.DataFrame,
    ) -> None:

        if not isinstance(
            history_df,
            pd.DataFrame,
        ):
            raise TypeError(
                "history_df must be a pandas DataFrame."
            )

        if history_df.empty:
            raise ValueError(
                "History dataframe is empty."
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

        if history_df[
            MEDICINE_ID_COLUMN
        ].isna().any():

            raise ValueError(
                "History contains null Medicine_ID values."
            )

        medicine_ids = (
            history_df[
                MEDICINE_ID_COLUMN
            ]
            .astype(str)
            .str.strip()
        )

        if medicine_ids.eq(
            ""
        ).any():

            raise ValueError(
                "History contains empty Medicine_ID values."
            )

        timestamps = pd.to_datetime(
            history_df[
                TIMESTAMP_COLUMN
            ],
            errors="coerce",
        )

        if timestamps.isna().any():

            raise ValueError(
                f"{TIMESTAMP_COLUMN} contains invalid timestamps."
            )

        target = pd.to_numeric(
            history_df[
                TARGET_COLUMN
            ],
            errors="coerce",
        )

        if target.isna().any():

            raise ValueError(
                f"{TARGET_COLUMN} contains non-numeric values."
            )

        target_array = target.to_numpy(
            dtype=float
        )

        if not np.isfinite(
            target_array
        ).all():

            raise ValueError(
                f"{TARGET_COLUMN} contains non-finite values."
            )

        if (
            target < 0
        ).any():

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

        if routing_table.empty:
            raise ValueError(
                "Routing table is empty."
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

        if routing_table[
            MEDICINE_ID_COLUMN
        ].isna().any():

            raise ValueError(
                "Routing table contains null Medicine_ID values."
            )

        medicine_ids = (
            routing_table[
                MEDICINE_ID_COLUMN
            ]
            .astype(str)
            .str.strip()
        )

        if medicine_ids.eq(
            ""
        ).any():

            raise ValueError(
                "Routing table contains empty Medicine_ID values."
            )

        duplicate_mask = medicine_ids.duplicated(
            keep=False
        )

        if duplicate_mask.any():

            duplicates = (
                medicine_ids[
                    duplicate_mask
                ]
                .unique()
                .tolist()
            )

            raise ValueError(
                "Routing table contains duplicate Medicine_ID values: "
                f"{duplicates}"
            )

        selected_models = (
            routing_table[
                SELECTED_MODEL_COLUMN
            ]
            .astype(str)
            .str.strip()
        )

        invalid_models = (
            set(
                selected_models.unique()
            )
            - {
                MODEL_CHRONOS,
                MODEL_TSB,
            }
        )

        if invalid_models:

            raise ValueError(
                "Routing table contains unsupported Selected_Model values: "
                f"{sorted(invalid_models)}"
            )

        routing_rules = (
            routing_table[
                ROUTING_RULE_COLUMN
            ]
            .astype(str)
            .str.strip()
        )

        if routing_rules.eq(
            ""
        ).any():

            raise ValueError(
                "Routing table contains empty Routing_Rule values."
            )

        routing_reasons = (
            routing_table[
                ROUTING_REASON_COLUMN
            ]
            .astype(str)
            .str.strip()
        )

        if routing_reasons.eq(
            ""
        ).any():

            raise ValueError(
                "Routing table contains empty Routing_Reason values."
            )

        advantages = pd.to_numeric(
            routing_table[
                VALIDATION_ADVANTAGE_COLUMN
            ],
            errors="coerce",
        )

        finite_advantages = (
            advantages.dropna()
        )

        if (
            not finite_advantages.empty
            and not np.isfinite(
                finite_advantages.to_numpy(
                    dtype=float
                )
            ).all()
        ):

            raise ValueError(
                "Routing table contains non-finite "
                f"{VALIDATION_ADVANTAGE_COLUMN} values."
            )

        chronos_mask = (
            selected_models
            == MODEL_CHRONOS
        )

        if chronos_mask.any():

            chronos_advantages = (
                advantages[
                    chronos_mask
                ]
            )

            if chronos_advantages.isna().any():

                raise ValueError(
                    "Chronos routing records must contain a finite "
                    f"{VALIDATION_ADVANTAGE_COLUMN}."
                )

    # ========================================================================
    # DAILY HISTORY NORMALIZATION
    # ========================================================================

    @staticmethod
    def _prepare_daily_history(
        medicine_id: str,
        history_df: pd.DataFrame,
        forecast_origin: Optional[pd.Timestamp] = None,
    ) -> pd.DataFrame:

        medicine_id = str(
            medicine_id
        ).strip()

        if not medicine_id:
            raise ValueError(
                "medicine_id must not be empty."
            )

        normalized_ids = (
            history_df[
                MEDICINE_ID_COLUMN
            ]
            .astype(str)
            .str.strip()
        )

        history = (
            history_df[
                normalized_ids
                == medicine_id
            ]
            .copy()
        )

        if history.empty:

            raise InsufficientHistoryError(
                f"No history for medicine_id={medicine_id}"
            )

        history[
            MEDICINE_ID_COLUMN
        ] = (
            history[
                MEDICINE_ID_COLUMN
            ]
            .astype(str)
            .str.strip()
        )

        history[
            TIMESTAMP_COLUMN
        ] = pd.to_datetime(
            history[
                TIMESTAMP_COLUMN
            ],
            errors="raise",
        ).dt.normalize()

        history[
            TARGET_COLUMN
        ] = pd.to_numeric(
            history[
                TARGET_COLUMN
            ],
            errors="raise",
        )

        if (
            history[
                TARGET_COLUMN
            ]
            < 0
        ).any():

            raise ValueError(
                f"Negative demand found for medicine={medicine_id}."
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
            )[
                TARGET_COLUMN
            ]
            .sum()
            .sort_values(
                TIMESTAMP_COLUMN
            )
            .reset_index(
                drop=True
            )
        )

        start_date = (
            history[
                TIMESTAMP_COLUMN
            ].min()
        )

        observed_end_date = (
            history[
                TIMESTAMP_COLUMN
            ].max()
        )

        if (
            pd.isna(start_date)
            or pd.isna(observed_end_date)
        ):

            raise ValueError(
                f"Invalid date range for medicine={medicine_id}."
            )

        if forecast_origin is None:

            aligned_end_date = (
                observed_end_date
            )

        else:

            aligned_end_date = (
                pd.Timestamp(
                    forecast_origin
                )
                .normalize()
            )

            if (
                aligned_end_date
                < observed_end_date
            ):

                raise ValueError(
                    "forecast_origin cannot be earlier than "
                    "the latest observed medicine date. "
                    f"medicine={medicine_id}, "
                    f"forecast_origin={aligned_end_date}, "
                    f"observed_end={observed_end_date}"
                )

        calendar = pd.date_range(
            start=start_date,
            end=aligned_end_date,
            freq="D",
        )

        daily = (
            history
            .set_index(
                TIMESTAMP_COLUMN
            )[
                TARGET_COLUMN
            ]
            .reindex(
                calendar,
                fill_value=0.0,
            )
            .rename(
                TARGET_COLUMN
            )
            .rename_axis(
                TIMESTAMP_COLUMN
            )
            .reset_index()
        )

        daily[
            MEDICINE_ID_COLUMN
        ] = medicine_id

        daily = daily[
            [
                MEDICINE_ID_COLUMN,
                TIMESTAMP_COLUMN,
                TARGET_COLUMN,
            ]
        ]

        daily[
            TARGET_COLUMN
        ] = (
            pd.to_numeric(
                daily[
                    TARGET_COLUMN
                ],
                errors="raise",
            )
            .fillna(0.0)
            .clip(lower=0.0)
        )

        expected_rows = (
            aligned_end_date
            - start_date
        ).days + 1

        if len(daily) != expected_rows:

            raise ValueError(
                "Daily calendar construction failed for "
                f"medicine={medicine_id}: expected "
                f"{expected_rows} rows, got {len(daily)}."
            )

        if daily[
            TIMESTAMP_COLUMN
        ].duplicated().any():

            raise ValueError(
                "Duplicate dates remain after normalization "
                f"for medicine={medicine_id}."
            )

        demand_array = daily[
            TARGET_COLUMN
        ].to_numpy(
            dtype=float
        )

        if not np.isfinite(
            demand_array
        ).all():

            raise ValueError(
                "Daily history contains non-finite demand "
                f"for medicine={medicine_id}."
            )

        logger.info(
            "DAILY HISTORY | medicine=%s | rows=%d | "
            "start=%s | observed_end=%s | aligned_end=%s | "
            "zero_days=%d",
            medicine_id,
            len(daily),
            start_date,
            observed_end_date,
            aligned_end_date,
            int(
                (
                    daily[
                        TARGET_COLUMN
                    ]
                    == 0
                ).sum()
            ),
        )

        return daily

    # ========================================================================
    # ROUTING
    # ========================================================================

    def _get_route(
        self,
        medicine_id: str,
        routing_table: pd.DataFrame,
    ) -> tuple[str, float, str]:

        """
        Return the frozen routing decision.

        IMPORTANT:

        Existing routing records are preserved exactly.

        This method does NOT recompute model selection from
        Validation_Advantage_Pct.

        That prevents valid unvalidated medicines with NaN advantage
        from losing their explicit provenance.
        """

        medicine_id = str(
            medicine_id
        ).strip()

        normalized_ids = (
            routing_table[
                MEDICINE_ID_COLUMN
            ]
            .astype(str)
            .str.strip()
        )

        rows = routing_table[
            normalized_ids
            == medicine_id
        ]

        # Missing routing record:
        #
        # This is different from an explicit routing record containing
        # Validation_Advantage_Pct = NaN.
        if rows.empty:

            logger.warning(
                "No routing record for medicine=%s. "
                "Using explicit TSB service fallback.",
                medicine_id,
            )

            return (
                MODEL_TSB,
                float("nan"),
                "missing_routing_record",
            )

        if len(rows) != 1:

            raise ValueError(
                "Expected exactly one routing record for "
                f"medicine={medicine_id}, got {len(rows)}."
            )

        row = rows.iloc[0]

        selected_model = str(
            row[
                SELECTED_MODEL_COLUMN
            ]
        ).strip()

        if selected_model not in {
            MODEL_CHRONOS,
            MODEL_TSB,
        }:

            raise ValueError(
                "Routing record contains unsupported model "
                f"for medicine={medicine_id}: "
                f"{selected_model!r}"
            )

        routing_reason = str(
            row[
                ROUTING_REASON_COLUMN
            ]
        ).strip()

        if not routing_reason:

            raise ValueError(
                "Routing record contains empty Routing_Reason "
                f"for medicine={medicine_id}."
            )

        raw_advantage = row[
            VALIDATION_ADVANTAGE_COLUMN
        ]

        if pd.isna(
            raw_advantage
        ):

            advantage = float("nan")

        else:

            try:

                advantage = float(
                    raw_advantage
                )

            except (
                TypeError,
                ValueError,
            ) as exc:

                raise ValueError(
                    "Routing record contains invalid "
                    f"{VALIDATION_ADVANTAGE_COLUMN} "
                    f"for medicine={medicine_id}: "
                    f"{raw_advantage!r}"
                ) from exc

            if not np.isfinite(
                advantage
            ):

                raise ValueError(
                    "Routing record contains non-finite "
                    f"{VALIDATION_ADVANTAGE_COLUMN} "
                    f"for medicine={medicine_id}: "
                    f"{raw_advantage!r}"
                )

        if (
            selected_model
            == MODEL_CHRONOS
            and not np.isfinite(
                advantage
            )
        ):

            raise ValueError(
                "Chronos routing record requires a finite "
                f"{VALIDATION_ADVANTAGE_COLUMN} "
                f"for medicine={medicine_id}."
            )

        logger.info(
            "FROZEN ROUTE | medicine=%s | model=%s | "
            "advantage=%s | reason=%s",
            medicine_id,
            selected_model,
            advantage,
            routing_reason,
        )

        return (
            selected_model,
            advantage,
            routing_reason,
        )

    # ========================================================================
    # CHRONOS
    # ========================================================================

    def _forecast_chronos(
        self,
        medicine_id: str,
        history_df: pd.DataFrame,
    ) -> ProductionForecastResult:

        predictor_config = (
            self.predictor.config
        )

        chronos_history = (
            history_df.copy()
        )

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
            - set(
                chronos_history.columns
            )
        )

        if missing:

            raise ValueError(
                "Chronos history adapter produced missing columns: "
                f"{sorted(missing)}"
            )

        result = (
            self.predictor.forecast_medicine(
                medicine_id,
                chronos_history,
            )
        )

        expected_horizon = int(
            self.config.prediction_length
        )

        if (
            len(result.days)
            != expected_horizon
        ):

            raise ValueError(
                f"Chronos returned {len(result.days)} forecast days "
                f"for medicine={medicine_id}; "
                f"expected {expected_horizon}."
            )

        dates = [
            pd.Timestamp(
                day.forecast_date
            ).normalize()
            for day in result.days
        ]

        p10 = [
            float(
                day.quantiles.p10
            )
            for day in result.days
        ]

        p50 = [
            float(
                day.quantiles.p50
            )
            for day in result.days
        ]

        p90 = [
            float(
                day.quantiles.p90
            )
            for day in result.days
        ]

        predicted = list(
            p50
        )

        if len(set(dates)) != len(dates):

            raise ValueError(
                "Chronos returned duplicate forecast dates "
                f"for medicine={medicine_id}."
            )

        if dates != sorted(dates):

            raise ValueError(
                "Chronos returned unsorted forecast dates "
                f"for medicine={medicine_id}."
            )

        for name, values in {
            "Predicted_Demand": predicted,
            "P10": p10,
            "P50": p50,
            "P90": p90,
        }.items():

            values_array = np.asarray(
                values,
                dtype=float,
            )

            if not np.isfinite(
                values_array
            ).all():

                raise ValueError(
                    f"Chronos returned non-finite {name} "
                    f"for medicine={medicine_id}."
                )

            if (
                values_array < 0
            ).any():

                raise ValueError(
                    f"Chronos returned negative {name} "
                    f"for medicine={medicine_id}."
                )

        if (
            np.asarray(p10)
            > np.asarray(p50)
        ).any():

            raise ValueError(
                "Chronos P10 exceeds P50 "
                f"for medicine={medicine_id}."
            )

        if (
            np.asarray(p50)
            > np.asarray(p90)
        ).any():

            raise ValueError(
                "Chronos P50 exceeds P90 "
                f"for medicine={medicine_id}."
            )

        if not np.array_equal(
            np.asarray(
                predicted,
                dtype=float,
            ),
            np.asarray(
                p50,
                dtype=float,
            ),
        ):

            raise ValueError(
                "Chronos production invariant violated: "
                "Predicted_Demand must equal P50 exactly."
            )

        context_length_used = int(
            result.context_length_used
        )

        prediction_length = int(
            result.prediction_length
        )

        if context_length_used <= 0:

            raise ValueError(
                "Chronos returned invalid context_length_used "
                f"for medicine={medicine_id}."
            )

        if (
            prediction_length
            != expected_horizon
        ):

            raise ValueError(
                "Chronos returned invalid prediction_length "
                f"for medicine={medicine_id}: "
                f"{prediction_length}."
            )

        return ProductionForecastResult(
            medicine_id=str(
                medicine_id
            ),
            selected_model=MODEL_CHRONOS,
            validation_advantage_pct=float(
                "nan"
            ),
            routing_reason="",
            forecast_dates=dates,
            predicted_demand=predicted,
            p10=p10,
            p50=p50,
            p90=p90,
            context_length_used=context_length_used,
            prediction_length=prediction_length,
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
            history_df[
                MEDICINE_ID_COLUMN
            ]
            .astype(str)
            .str.strip()
        )

        medicine_id = str(
            medicine_id
        ).strip()

        history = (
            history_df[
                normalized_ids
                == medicine_id
            ]
            .sort_values(
                TIMESTAMP_COLUMN
            )
            .copy()
        )

        if history.empty:

            raise InsufficientHistoryError(
                f"No history for medicine_id={medicine_id}"
            )

        target = (
            pd.to_numeric(
                history[
                    TARGET_COLUMN
                ],
                errors="raise",
            )
            .astype(float)
            .clip(lower=0.0)
        )

        horizon = int(
            self.config.prediction_length
        )

        if horizon <= 0:

            raise ValueError(
                f"prediction_length must be positive, got {horizon}."
            )

        prediction = tsb_forecast(
            target,
            horizon,
        )

        if (
            len(prediction)
            != horizon
        ):

            raise ValueError(
                f"TSB returned {len(prediction)} predictions "
                f"for medicine={medicine_id}; "
                f"expected {horizon}."
            )

        if not np.isfinite(
            prediction
        ).all():

            raise ValueError(
                "TSB returned non-finite predictions "
                f"for medicine={medicine_id}."
            )

        if (
            prediction < 0
        ).any():

            raise ValueError(
                "TSB returned negative predictions "
                f"for medicine={medicine_id}."
            )

        last_date = (
            pd.Timestamp(
                history[
                    TIMESTAMP_COLUMN
                ].max()
            )
            .normalize()
        )

        forecast_dates = list(
            pd.date_range(
                start=(
                    last_date
                    + pd.Timedelta(
                        days=1
                    )
                ),
                periods=horizon,
                freq="D",
            )
        )

        return ProductionForecastResult(
            medicine_id=medicine_id,
            selected_model=MODEL_TSB,
            validation_advantage_pct=float(
                "nan"
            ),
            routing_reason="",
            forecast_dates=forecast_dates,
            predicted_demand=[
                float(value)
                for value in prediction
            ],
            p10=None,
            p50=None,
            p90=None,
            context_length_used=len(
                history
            ),
            prediction_length=horizon,
            generated_at=datetime.now(
                timezone.utc
            ),
        )

    # ========================================================================
    # SINGLE MEDICINE FORECAST
    # ========================================================================

    def forecast_medicine(
        self,
        medicine_id: str,
        history_df: pd.DataFrame,
        routing_table: pd.DataFrame,
        forecast_origin: Optional[pd.Timestamp] = None,
    ) -> ProductionForecastResult:

        self._validate_history(
            history_df
        )

        self._validate_routing_table(
            routing_table
        )

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
                forecast_origin=forecast_origin,
            )
        )

        (
            selected_model,
            advantage,
            routing_reason,
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
            routing_reason,
        )

        if (
            selected_model
            == MODEL_CHRONOS
        ):

            result = self._forecast_chronos(
                medicine_id,
                daily_history,
            )

        elif (
            selected_model
            == MODEL_TSB
        ):

            result = self._forecast_tsb(
                medicine_id,
                daily_history,
            )

        else:

            raise ValueError(
                f"Unsupported production model: "
                f"{selected_model!r}"
            )

        # --------------------------------------------------------------------
        # Preserve routing provenance from the frozen routing decision.
        # --------------------------------------------------------------------

        result.selected_model = selected_model

        result.validation_advantage_pct = (
            advantage
        )

        result.routing_reason = (
            routing_reason
        )

        return result

    # ========================================================================
    # BATCH FORECASTING
    # ========================================================================

    def forecast_batch(
        self,
        history_df: pd.DataFrame,
        routing_table: pd.DataFrame,
        item_ids: Optional[List[str]] = None,
    ) -> tuple[pd.DataFrame, List[str]]:

        self._validate_history(
            history_df
        )

        self._validate_routing_table(
            routing_table
        )

        normalized_history = (
            history_df.copy()
        )

        normalized_history[
            MEDICINE_ID_COLUMN
        ] = (
            normalized_history[
                MEDICINE_ID_COLUMN
            ]
            .astype(str)
            .str.strip()
        )

        normalized_history[
            TIMESTAMP_COLUMN
        ] = pd.to_datetime(
            normalized_history[
                TIMESTAMP_COLUMN
            ],
            errors="raise",
        ).dt.normalize()

        history_ids = set(
            normalized_history[
                MEDICINE_ID_COLUMN
            ]
            .unique()
        )

        if item_ids is None:

            ids = sorted(
                history_ids
            )

        else:

            ids = [
                str(
                    medicine_id
                ).strip()
                for medicine_id in item_ids
                if str(
                    medicine_id
                ).strip()
            ]

            ids = list(
                dict.fromkeys(
                    ids
                )
            )

            unknown_ids = [
                medicine_id
                for medicine_id in ids
                if medicine_id
                not in history_ids
            ]

            if unknown_ids:

                logger.warning(
                    "Requested medicine IDs not found in history: %s",
                    unknown_ids[:20],
                )

            ids = [
                medicine_id
                for medicine_id in ids
                if medicine_id
                in history_ids
            ]

        if not ids:

            raise ValueError(
                "No medicine IDs available for production forecasting."
            )

        global_forecast_origin = (
            normalized_history[
                TIMESTAMP_COLUMN
            ]
            .max()
        )

        if pd.isna(
            global_forecast_origin
        ):

            raise ValueError(
                "Unable to determine global forecast origin."
            )

        global_forecast_origin = (
            pd.Timestamp(
                global_forecast_origin
            )
            .normalize()
        )

        horizon = int(
            self.config.prediction_length
        )

        if horizon <= 0:

            raise ValueError(
                f"prediction_length must be positive, got {horizon}."
            )

        expected_forecast_dates = list(
            pd.date_range(
                start=(
                    global_forecast_origin
                    + pd.Timedelta(
                        days=1
                    )
                ),
                periods=horizon,
                freq="D",
            )
        )

        logger.info(
            "Starting production forecast batch | medicines=%d | "
            "origin=%s | window=%s -> %s",
            len(ids),
            global_forecast_origin.date(),
            expected_forecast_dates[0].date(),
            expected_forecast_dates[-1].date(),
        )

        records: list[dict] = []

        failed: list[str] = []

        for medicine_id in ids:

            try:

                result = self.forecast_medicine(
                    medicine_id,
                    normalized_history,
                    routing_table,
                    forecast_origin=global_forecast_origin,
                )

                if (
                    result.selected_model
                    not in {
                        MODEL_CHRONOS,
                        MODEL_TSB,
                    }
                ):

                    raise ValueError(
                        "Unsupported result model "
                        f"{result.selected_model!r} "
                        f"for medicine={medicine_id}."
                    )

                if (
                    result.prediction_length
                    != horizon
                ):

                    raise ValueError(
                        "Prediction length mismatch for "
                        f"medicine={medicine_id}: "
                        f"{result.prediction_length} != {horizon}."
                    )

                if (
                    len(
                        result.forecast_dates
                    )
                    != horizon
                ):

                    raise ValueError(
                        "Forecast date count mismatch for "
                        f"medicine={medicine_id}."
                    )

                if (
                    len(
                        result.predicted_demand
                    )
                    != horizon
                ):

                    raise ValueError(
                        "Prediction count mismatch for "
                        f"medicine={medicine_id}."
                    )

                actual_forecast_dates = [
                    pd.Timestamp(
                        date
                    ).normalize()
                    for date in result.forecast_dates
                ]

                if (
                    actual_forecast_dates
                    != expected_forecast_dates
                ):

                    raise ValueError(
                        "Forecast dates are not aligned to the "
                        "global forecast origin for "
                        f"medicine={medicine_id}. "
                        f"Expected "
                        f"{expected_forecast_dates[0].date()} -> "
                        f"{expected_forecast_dates[-1].date()}, "
                        f"got "
                        f"{actual_forecast_dates[0].date()} -> "
                        f"{actual_forecast_dates[-1].date()}."
                    )

                if (
                    result.selected_model
                    == MODEL_CHRONOS
                ):

                    if (
                        result.p10 is None
                        or result.p50 is None
                        or result.p90 is None
                    ):

                        raise ValueError(
                            "Chronos result missing quantiles "
                            f"for medicine={medicine_id}."
                        )

                    if not (
                        len(
                            result.p10
                        )
                        == len(
                            result.p50
                        )
                        == len(
                            result.p90
                        )
                        == horizon
                    ):

                        raise ValueError(
                            "Chronos quantile length mismatch "
                            f"for medicine={medicine_id}."
                        )

                    if not np.array_equal(
                        np.asarray(
                            result.predicted_demand,
                            dtype=float,
                        ),
                        np.asarray(
                            result.p50,
                            dtype=float,
                        ),
                    ):

                        raise ValueError(
                            "Chronos production invariant violated: "
                            "Predicted_Demand must equal P50 exactly "
                            f"for medicine={medicine_id}."
                        )

                else:

                    if any(
                        values is not None
                        for values in (
                            result.p10,
                            result.p50,
                            result.p90,
                        )
                    ):

                        raise ValueError(
                            "TSB result unexpectedly contains "
                            f"quantiles for medicine={medicine_id}."
                        )

                for index, forecast_date in enumerate(
                    result.forecast_dates
                ):

                    records.append(
                        {
                            MEDICINE_ID_COLUMN:
                                result.medicine_id,

                            "Forecast_Date":
                                pd.Timestamp(
                                    forecast_date
                                ).normalize(),

                            SELECTED_MODEL_COLUMN:
                                result.selected_model,

                            "Predicted_Demand":
                                float(
                                    result.predicted_demand[
                                        index
                                    ]
                                ),

                            "P10":
                                (
                                    float(
                                        result.p10[index]
                                    )
                                    if result.p10
                                    is not None
                                    else np.nan
                                ),

                            "P50":
                                (
                                    float(
                                        result.p50[index]
                                    )
                                    if result.p50
                                    is not None
                                    else np.nan
                                ),

                            "P90":
                                (
                                    float(
                                        result.p90[index]
                                    )
                                    if result.p90
                                    is not None
                                    else np.nan
                                ),

                            VALIDATION_ADVANTAGE_COLUMN:
                                result.validation_advantage_pct,

                            ROUTING_REASON_COLUMN:
                                result.routing_reason,

                            "Context_Length_Used":
                                int(
                                    result.context_length_used
                                ),

                            "Prediction_Length":
                                int(
                                    result.prediction_length
                                ),

                            "Generated_At":
                                result.generated_at,
                        }
                    )

            except InsufficientHistoryError as exc:

                logger.warning(
                    "Skipping medicine=%s due to insufficient history: %s",
                    medicine_id,
                    exc,
                )

                failed.append(
                    medicine_id
                )

            except Exception:

                logger.exception(
                    "Production forecast failed for medicine=%s",
                    medicine_id,
                )

                failed.append(
                    medicine_id
                )

        if not records:

            raise RuntimeError(
                "Production forecast generated no successful results."
            )

        result_df = pd.DataFrame(
            records,
            columns=OUTPUT_COLUMNS,
        )

        result_df = (
            result_df
            .sort_values(
                [
                    MEDICINE_ID_COLUMN,
                    "Forecast_Date",
                ]
            )
            .reset_index(
                drop=True
            )
        )

        self._validate_output(
            result_df
        )

        self._validate_global_forecast_window(
            result_df,
            expected_forecast_dates,
        )

        logger.info(
            "Production forecast complete | "
            "successful=%d | failed=%d | rows=%d",
            result_df[
                MEDICINE_ID_COLUMN
            ].nunique(),
            len(failed),
            len(result_df),
        )

        return (
            result_df,
            failed,
        )

    # ========================================================================
    # GLOBAL FORECAST WINDOW VALIDATION
    # ========================================================================

    @staticmethod
    def _validate_global_forecast_window(
        result_df: pd.DataFrame,
        expected_forecast_dates: list[pd.Timestamp],
    ) -> None:

        if result_df.empty:
            raise ValueError(
                "Cannot validate forecast window on empty output."
            )

        expected = [
            pd.Timestamp(
                date
            ).normalize()
            for date in expected_forecast_dates
        ]

        actual_unique_dates = sorted(
            pd.to_datetime(
                result_df[
                    "Forecast_Date"
                ],
                errors="raise",
            )
            .dt.normalize()
            .unique()
        )

        if list(
            actual_unique_dates
        ) != expected:

            raise ValueError(
                "Final production output does not contain exactly "
                "the expected shared forecast window."
            )

        grouped_dates = (
            result_df
            .groupby(
                MEDICINE_ID_COLUMN
            )[
                "Forecast_Date"
            ]
            .apply(
                lambda values: [
                    pd.Timestamp(
                        value
                    ).normalize()
                    for value in values
                ]
            )
        )

        for (
            medicine_id,
            dates,
        ) in grouped_dates.items():

            if dates != expected:

                raise ValueError(
                    "Final production output has inconsistent "
                    "forecast dates for "
                    f"medicine={medicine_id}."
                )

    # ========================================================================
    # OUTPUT VALIDATION
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

        if result_df.empty:
            raise ValueError(
                "Production output is empty."
            )

        missing = (
            set(
                OUTPUT_COLUMNS
            )
            - set(
                result_df.columns
            )
        )

        if missing:

            raise ValueError(
                "Production output missing columns: "
                f"{sorted(missing)}"
            )

        # --------------------------------------------------------------------
        # MEDICINE IDs
        # --------------------------------------------------------------------

        if result_df[
            MEDICINE_ID_COLUMN
        ].isna().any():

            raise ValueError(
                "Production output contains null Medicine_ID values."
            )

        medicine_ids = (
            result_df[
                MEDICINE_ID_COLUMN
            ]
            .astype(str)
            .str.strip()
        )

        if medicine_ids.eq(
            ""
        ).any():

            raise ValueError(
                "Production output contains empty Medicine_ID values."
            )

        # --------------------------------------------------------------------
        # MODELS
        # --------------------------------------------------------------------

        models = (
            result_df[
                SELECTED_MODEL_COLUMN
            ]
            .astype(str)
            .str.strip()
        )

        invalid_models = (
            set(
                models.unique()
            )
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

        # --------------------------------------------------------------------
        # FORECAST DATES
        # --------------------------------------------------------------------

        forecast_dates = pd.to_datetime(
            result_df[
                "Forecast_Date"
            ],
            errors="coerce",
        )

        if forecast_dates.isna().any():

            raise ValueError(
                "Production output contains invalid Forecast_Date values."
            )

        duplicate_rows = (
            result_df.duplicated(
                subset=[
                    MEDICINE_ID_COLUMN,
                    "Forecast_Date",
                ],
                keep=False,
            )
        )

        if duplicate_rows.any():

            raise ValueError(
                "Production output contains duplicate "
                "Medicine_ID + Forecast_Date combinations."
            )

        # --------------------------------------------------------------------
        # PREDICTED DEMAND
        # --------------------------------------------------------------------

        predicted = pd.to_numeric(
            result_df[
                "Predicted_Demand"
            ],
            errors="coerce",
        )

        if predicted.isna().any():

            raise ValueError(
                "Production output contains null or non-numeric "
                "Predicted_Demand values."
            )

        predicted_array = predicted.to_numpy(
            dtype=float
        )

        if not np.isfinite(
            predicted_array
        ).all():

            raise ValueError(
                "Production output contains non-finite "
                "Predicted_Demand values."
            )

        if (
            predicted_array < 0
        ).any():

            raise ValueError(
                "Production output contains negative demand."
            )

        # --------------------------------------------------------------------
        # ROUTING REASON
        # --------------------------------------------------------------------

        routing_reasons = (
            result_df[
                ROUTING_REASON_COLUMN
            ]
            .astype(str)
            .str.strip()
        )

        if routing_reasons.eq(
            ""
        ).any():

            raise ValueError(
                "Production output contains empty Routing_Reason values."
            )

        # --------------------------------------------------------------------
        # VALIDATION ADVANTAGE
        # --------------------------------------------------------------------

        advantages = pd.to_numeric(
            result_df[
                VALIDATION_ADVANTAGE_COLUMN
            ],
            errors="coerce",
        )

        finite_advantages = (
            advantages.dropna()
        )

        if (
            not finite_advantages.empty
            and not np.isfinite(
                finite_advantages.to_numpy(
                    dtype=float
                )
            ).all()
        ):

            raise ValueError(
                f"{VALIDATION_ADVANTAGE_COLUMN} contains "
                "non-finite values."
            )

        # --------------------------------------------------------------------
        # CHRONOS VALIDATION
        # --------------------------------------------------------------------

        chronos = result_df[
            models
            == MODEL_CHRONOS
        ]

        if not chronos.empty:

            chronos_quantiles = chronos[
                [
                    "P10",
                    "P50",
                    "P90",
                ]
            ].apply(
                pd.to_numeric,
                errors="coerce",
            )

            if chronos_quantiles.isna().any().any():

                raise ValueError(
                    "Chronos forecasts must contain numeric "
                    "P10, P50 and P90."
                )

            quantile_array = (
                chronos_quantiles.to_numpy(
                    dtype=float
                )
            )

            if not np.isfinite(
                quantile_array
            ).all():

                raise ValueError(
                    "Chronos forecasts contain non-finite quantiles."
                )

            if (
                quantile_array < 0
            ).any():

                raise ValueError(
                    "Chronos forecasts contain negative quantiles."
                )

            if (
                chronos_quantiles[
                    "P10"
                ]
                > chronos_quantiles[
                    "P50"
                ]
            ).any():

                raise ValueError(
                    "Chronos P10 exceeds P50."
                )

            if (
                chronos_quantiles[
                    "P50"
                ]
                > chronos_quantiles[
                    "P90"
                ]
            ).any():

                raise ValueError(
                    "Chronos P50 exceeds P90."
                )

            if not np.array_equal(
                chronos[
                    "Predicted_Demand"
                ].to_numpy(
                    dtype=float
                ),
                chronos[
                    "P50"
                ].to_numpy(
                    dtype=float
                ),
            ):

                raise ValueError(
                    "Chronos production invariant violated: "
                    "Predicted_Demand must equal P50 exactly."
                )

            chronos_advantages = pd.to_numeric(
                chronos[
                    VALIDATION_ADVANTAGE_COLUMN
                ],
                errors="coerce",
            )

            if chronos_advantages.isna().any():

                raise ValueError(
                    "Chronos production forecasts must contain "
                    "finite validation advantage values."
                )

        # --------------------------------------------------------------------
        # TSB VALIDATION
        # --------------------------------------------------------------------

        tsb = result_df[
            models
            == MODEL_TSB
        ]

        if not tsb.empty:

            if tsb[
                [
                    "P10",
                    "P50",
                    "P90",
                ]
            ].notna().any().any():

                raise ValueError(
                    "TSB forecasts must not contain "
                    "P10/P50/P90 values."
                )

        # --------------------------------------------------------------------
        # HORIZON
        # --------------------------------------------------------------------

        horizon_counts = (
            result_df
            .groupby(
                MEDICINE_ID_COLUMN
            )
            .size()
        )

        invalid_horizons = (
            horizon_counts
            != int(
                self.config.prediction_length
            )
        )

        if invalid_horizons.any():

            raise ValueError(
                "Invalid forecast horizon for medicines: "
                f"{horizon_counts[invalid_horizons].to_dict()}"
            )

        # --------------------------------------------------------------------
        # PREDICTION LENGTH
        # --------------------------------------------------------------------

        prediction_lengths = pd.to_numeric(
            result_df[
                "Prediction_Length"
            ],
            errors="coerce",
        )

        if prediction_lengths.isna().any():

            raise ValueError(
                "Prediction_Length contains invalid values."
            )

        if (
            prediction_lengths
            != int(
                self.config.prediction_length
            )
        ).any():

            raise ValueError(
                "Production output contains unexpected "
                "Prediction_Length values."
            )

        # --------------------------------------------------------------------
        # CONTEXT LENGTH
        # --------------------------------------------------------------------

        context_lengths = pd.to_numeric(
            result_df[
                "Context_Length_Used"
            ],
            errors="coerce",
        )

        if context_lengths.isna().any():

            raise ValueError(
                "Context_Length_Used contains invalid values."
            )

        if (
            context_lengths <= 0
        ).any():

            raise ValueError(
                "Context_Length_Used must be positive."
            )


# ============================================================================
# SMOKE TEST
# ============================================================================

def main() -> None:

    print("=" * 80)
    print(
        "PRODUCTION FORECAST SERVICE"
    )
    print("=" * 80)

    print()
    print(
        "ProductionForecastService import successful."
    )

    print()
    print(
        "Routing source of truth:"
    )
    print(
        "production_routing_table.parquet"
    )

    print()
    print(
        "Routing policy:"
    )
    print(
        "Existing routing records are preserved exactly."
    )
    print(
        "Missing routing record -> TSB fallback."
    )

    print()
    print(
        "Chronos invariant:"
    )
    print(
        "Predicted_Demand == P50 exactly."
    )

    print()
    print(
        "TSB parameters:"
    )
    print(
        "alpha_demand=0.1"
    )
    print(
        "alpha_probability=0.1"
    )

    print()
    print(
        "Configuration:"
    )
    print(
        f"prediction_length="
        f"{DEFAULT_CONFIG.prediction_length}"
    )
    print(
        f"context_length="
        f"{DEFAULT_CONFIG.context_length}"
    )

    print()
    print(
        "ProductionForecastService is ready."
    )


if __name__ == "__main__":
    main()