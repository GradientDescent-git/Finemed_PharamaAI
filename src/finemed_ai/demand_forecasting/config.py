from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


# ============================================================================
# FROZEN PRODUCTION FORECASTING POLICY
# ============================================================================
#
# IMPORTANT
# ---------
# These values were selected through the validated demand-forecasting
# experiments and must not be changed without re-running the evaluation
# suite.
#
# Validated production configuration:
#
#   Model:
#       amazon/chronos-2
#
#   Context:
#       730 daily observations
#
#   Prediction horizon:
#       30 days
#
#   Point forecast:
#       P50
#
#   Quantiles:
#       P10 ... P90
#
#   Bias correction:
#       Disabled
#
# TSB routing parameters are intentionally NOT stored here because TSB is
# a separate production routing/model policy. They belong to the production
# router/model implementation.
#
# ============================================================================


# ============================================================================
# VALIDATED CONSTANTS
# ============================================================================

PRODUCTION_MODEL_ID = "amazon/chronos-2"

PRODUCTION_CONTEXT_LENGTH = 730

PRODUCTION_PREDICTION_LENGTH = 30

PRODUCTION_POINT_QUANTILE = 0.5

PRODUCTION_QUANTILE_LEVELS: Tuple[float, ...] = (
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

PRODUCTION_APPLY_BIAS_CORRECTION = False


# ============================================================================
# CHRONOS INPUT SCHEMA
# ============================================================================

CHRONOS_ID_COLUMN = "item_id"

CHRONOS_TIMESTAMP_COLUMN = "timestamp"

CHRONOS_TARGET_COLUMN = "target"


# ============================================================================
# PRODUCTION OUTPUT SCHEMA
# ============================================================================

PRODUCTION_OUTPUT_ID_COLUMN = "Medicine_ID"

PRODUCTION_OUTPUT_DATE_COLUMN = "Forecast_Date"

PRODUCTION_OUTPUT_POINT_COLUMN = "Predicted_Demand"


# ============================================================================
# CONFIGURATION
# ============================================================================


@dataclass(frozen=True)
class ForecastConfig:
    """
    Immutable production configuration for demand forecasting.

    This configuration represents the validated Chronos-2 production
    settings.

    Validated production configuration:

        Model:
            amazon/chronos-2

        Context:
            730 daily observations

        Horizon:
            30 days

        Point forecast:
            P50

        Quantiles:
            P10 ... P90

        Bias correction:
            Disabled

    Chronos input schema:

        item_id
        timestamp
        target

    Production output schema:

        Medicine_ID
        Forecast_Date
        Predicted_Demand

    The configuration is frozen so production code cannot mutate it at
    runtime.
    """

    # ------------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------------

    model_id: str = PRODUCTION_MODEL_ID

    # ------------------------------------------------------------------------
    # Forecasting configuration
    # ------------------------------------------------------------------------

    context_length: int = PRODUCTION_CONTEXT_LENGTH

    prediction_length: int = PRODUCTION_PREDICTION_LENGTH

    quantile_levels: Tuple[float, ...] = (
        PRODUCTION_QUANTILE_LEVELS
    )

    point_quantile: float = PRODUCTION_POINT_QUANTILE

    # Validation showed that bias correction worsened WAPE.
    apply_bias_correction: bool = (
        PRODUCTION_APPLY_BIAS_CORRECTION
    )

    # ------------------------------------------------------------------------
    # Chronos input schema
    # ------------------------------------------------------------------------

    id_column: str = CHRONOS_ID_COLUMN

    timestamp_column: str = CHRONOS_TIMESTAMP_COLUMN

    target_column: str = CHRONOS_TARGET_COLUMN

    # ------------------------------------------------------------------------
    # Production output schema
    # ------------------------------------------------------------------------

    output_id_column: str = (
        PRODUCTION_OUTPUT_ID_COLUMN
    )

    output_date_column: str = (
        PRODUCTION_OUTPUT_DATE_COLUMN
    )

    output_point_column: str = (
        PRODUCTION_OUTPUT_POINT_COLUMN
    )

    # ------------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------------

    def __post_init__(self) -> None:
        """
        Validate the immutable production configuration.

        These checks protect the production pipeline from accidentally
        receiving an internally inconsistent ForecastConfig.
        """

        # --------------------------------------------------------------------
        # Model
        # --------------------------------------------------------------------

        if not isinstance(self.model_id, str):
            raise TypeError(
                "model_id must be a string."
            )

        if not self.model_id.strip():
            raise ValueError(
                "model_id must not be empty."
            )

        # --------------------------------------------------------------------
        # Context
        # --------------------------------------------------------------------

        if not isinstance(
            self.context_length,
            int,
        ):
            raise TypeError(
                "context_length must be an integer."
            )

        if self.context_length <= 0:
            raise ValueError(
                "context_length must be positive."
            )

        # --------------------------------------------------------------------
        # Prediction horizon
        # --------------------------------------------------------------------

        if not isinstance(
            self.prediction_length,
            int,
        ):
            raise TypeError(
                "prediction_length must be an integer."
            )

        if self.prediction_length <= 0:
            raise ValueError(
                "prediction_length must be positive."
            )

        # --------------------------------------------------------------------
        # Quantile levels
        # --------------------------------------------------------------------

        if not isinstance(
            self.quantile_levels,
            tuple,
        ):
            raise TypeError(
                "quantile_levels must be a tuple."
            )

        if len(self.quantile_levels) == 0:
            raise ValueError(
                "quantile_levels must not be empty."
            )

        for level in self.quantile_levels:

            if not isinstance(
                level,
                (int, float),
            ):
                raise TypeError(
                    "Every quantile level must be numeric."
                )

            if not 0.0 < float(level) < 1.0:
                raise ValueError(
                    "Every quantile level must be strictly "
                    "between 0 and 1."
                )

        if tuple(
            sorted(self.quantile_levels)
        ) != self.quantile_levels:

            raise ValueError(
                "quantile_levels must be sorted "
                "in ascending order."
            )

        if len(
            set(self.quantile_levels)
        ) != len(self.quantile_levels):

            raise ValueError(
                "quantile_levels must not contain duplicates."
            )

        # --------------------------------------------------------------------
        # Point quantile
        # --------------------------------------------------------------------

        if not isinstance(
            self.point_quantile,
            (int, float),
        ):
            raise TypeError(
                "point_quantile must be numeric."
            )

        if not 0.0 < float(
            self.point_quantile
        ) < 1.0:

            raise ValueError(
                "point_quantile must be strictly "
                "between 0 and 1."
            )

        if self.point_quantile not in (
            self.quantile_levels
        ):

            raise ValueError(
                "point_quantile must be present "
                "in quantile_levels."
            )

        # --------------------------------------------------------------------
        # Bias correction
        # --------------------------------------------------------------------

        if not isinstance(
            self.apply_bias_correction,
            bool,
        ):
            raise TypeError(
                "apply_bias_correction must be a boolean."
            )

        # --------------------------------------------------------------------
        # Column names
        # --------------------------------------------------------------------

        column_fields = {
            "id_column": self.id_column,
            "timestamp_column": self.timestamp_column,
            "target_column": self.target_column,
            "output_id_column": self.output_id_column,
            "output_date_column": self.output_date_column,
            "output_point_column": self.output_point_column,
        }

        for field_name, value in column_fields.items():

            if not isinstance(value, str):
                raise TypeError(
                    f"{field_name} must be a string."
                )

            if not value.strip():
                raise ValueError(
                    f"{field_name} must not be empty."
                )

        # --------------------------------------------------------------------
        # Chronos schema columns must be distinct.
        # --------------------------------------------------------------------

        chronos_columns = (
            self.id_column,
            self.timestamp_column,
            self.target_column,
        )

        if len(
            set(chronos_columns)
        ) != len(chronos_columns):

            raise ValueError(
                "Chronos input column names must be distinct."
            )

        # --------------------------------------------------------------------
        # Production output columns must be distinct.
        # --------------------------------------------------------------------

        output_columns = (
            self.output_id_column,
            self.output_date_column,
            self.output_point_column,
        )

        if len(
            set(output_columns)
        ) != len(output_columns):

            raise ValueError(
                "Production output column names must be distinct."
            )


# ============================================================================
# DEFAULT PRODUCTION CONFIGURATION
# ============================================================================

DEFAULT_CONFIG = ForecastConfig()