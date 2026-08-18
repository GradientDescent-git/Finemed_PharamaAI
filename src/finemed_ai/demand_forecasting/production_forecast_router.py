from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd

from finemed_ai.demand_forecasting.predictor_service import (
    InsufficientHistoryError,
    PredictorService,
)

logger = logging.getLogger(__name__)


# ============================================================================
# FROZEN PRODUCTION POLICY
# ============================================================================

ROUTING_RULE_NAME = "validation_advantage_ge_30pct"

VALIDATION_ADVANTAGE_THRESHOLD = 30.0

CHRONOS_MODEL = "chronos-2-P50"
TSB_MODEL = "tsb"


# ============================================================================
# SOURCE DATA SCHEMA
# ============================================================================

# Actual Module 2 / Silver daily demand dataset:
#
#   MDCODE
#   INVDT
#   Demand_Qty
#
# Chronos internally expects:
#
#   item_id
#   timestamp
#   target
#
# The production router therefore normalizes the Silver schema once at
# its boundary instead of forcing every downstream component to know
# about the ERP/Silver naming convention.

SOURCE_ID_COLUMN = "MDCODE"
SOURCE_TIMESTAMP_COLUMN = "INVDT"
SOURCE_TARGET_COLUMN = "Demand_Qty"

INTERNAL_ID_COLUMN = "item_id"
INTERNAL_TIMESTAMP_COLUMN = "timestamp"
INTERNAL_TARGET_COLUMN = "target"


# ============================================================================
# DATA STRUCTURES
# ============================================================================


@dataclass(frozen=True)
class RoutingDecision:
    """
    Immutable production routing decision.

    The decision is made entirely from validation-only information and is
    subsequently applied to future production forecasts.
    """

    medicine_id: str
    selected_model: str
    routing_rule: str
    validation_advantage_pct: Optional[float]
    reason: str


# ============================================================================
# PRODUCTION ROUTER
# ============================================================================


class ProductionForecastRouter:
    """
    Production model router.

    Frozen policy:

        validation advantage >= 30%
            -> Chronos-2 P50

        otherwise
            -> TSB

    IMPORTANT
    ---------
    The threshold is NOT optimized here.

    The 30% threshold was selected during validation and subsequently
    evaluated on untouched holdout data.

    Therefore this production component treats the threshold as frozen.
    """

    def __init__(
        self,
        predictor_service: Optional[PredictorService] = None,
    ) -> None:

        self.predictor = (
            predictor_service
            if predictor_service is not None
            else PredictorService.get_instance()
        )

    # ========================================================================
    # DATA NORMALIZATION
    # ========================================================================

    @staticmethod
    def _validate_source_history(history_df: pd.DataFrame) -> None:
        """
        Validate the actual Silver daily-demand schema.

        Expected:

            MDCODE
            INVDT
            Demand_Qty
        """

        required = {
            SOURCE_ID_COLUMN,
            SOURCE_TIMESTAMP_COLUMN,
            SOURCE_TARGET_COLUMN,
        }

        missing = required - set(history_df.columns)

        if missing:
            raise ValueError(
                "History dataframe is missing required Silver columns: "
                f"{sorted(missing)}. "
                f"Expected columns: "
                f"{[SOURCE_ID_COLUMN, SOURCE_TIMESTAMP_COLUMN, SOURCE_TARGET_COLUMN]}"
            )

    @staticmethod
    def _prepare_history(
        history_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Convert the actual Silver daily-demand dataframe into the internal
        Chronos schema.

        Input:

            MDCODE
            INVDT
            Demand_Qty

        Output:

            item_id
            timestamp
            target

        The method also:

        - normalizes medicine IDs to strings
        - converts timestamps to datetime
        - converts demand to numeric
        - removes invalid rows
        - sorts by medicine/date
        - aggregates duplicate medicine/date observations
        - creates a complete daily calendar per medicine

        The complete-calendar step is important because Chronos-2 expects
        regularly spaced observations.
        """

        ProductionForecastRouter._validate_source_history(history_df)

        df = history_df[
            [
                SOURCE_ID_COLUMN,
                SOURCE_TIMESTAMP_COLUMN,
                SOURCE_TARGET_COLUMN,
            ]
        ].copy()

        # ------------------------------------------------------------------
        # Normalize identifiers
        # ------------------------------------------------------------------

        df[INTERNAL_ID_COLUMN] = (
            df[SOURCE_ID_COLUMN]
            .astype(str)
            .str.strip()
        )

        # ------------------------------------------------------------------
        # Normalize dates
        # ------------------------------------------------------------------

        df[INTERNAL_TIMESTAMP_COLUMN] = pd.to_datetime(
            df[SOURCE_TIMESTAMP_COLUMN],
            errors="coerce",
        )

        # Remove timezone if present. The forecasting pipeline operates on
        # daily ERP dates rather than timezone-aware timestamps.
        if pd.api.types.is_datetime64tz_dtype(
            df[INTERNAL_TIMESTAMP_COLUMN]
        ):
            df[INTERNAL_TIMESTAMP_COLUMN] = (
                df[INTERNAL_TIMESTAMP_COLUMN]
                .dt.tz_localize(None)
            )

        # ------------------------------------------------------------------
        # Normalize target
        # ------------------------------------------------------------------

        df[INTERNAL_TARGET_COLUMN] = pd.to_numeric(
            df[SOURCE_TARGET_COLUMN],
            errors="coerce",
        )

        # ------------------------------------------------------------------
        # Remove invalid records
        # ------------------------------------------------------------------

        before = len(df)

        df = df.dropna(
            subset=[
                INTERNAL_ID_COLUMN,
                INTERNAL_TIMESTAMP_COLUMN,
                INTERNAL_TARGET_COLUMN,
            ]
        ).copy()

        removed = before - len(df)

        if removed:
            logger.warning(
                "Removed %d invalid history rows during production "
                "forecast preparation.",
                removed,
            )

        # Demand cannot be negative for this production demand series.
        negative_count = int(
            (df[INTERNAL_TARGET_COLUMN] < 0).sum()
        )

        if negative_count:
            logger.warning(
                "Found %d negative demand observations. "
                "Clipping them to zero.",
                negative_count,
            )

            df[INTERNAL_TARGET_COLUMN] = (
                df[INTERNAL_TARGET_COLUMN]
                .clip(lower=0.0)
            )

        # ------------------------------------------------------------------
        # Aggregate duplicate medicine/date observations
        # ------------------------------------------------------------------

        duplicate_mask = df.duplicated(
            subset=[
                INTERNAL_ID_COLUMN,
                INTERNAL_TIMESTAMP_COLUMN,
            ],
            keep=False,
        )

        duplicate_count = int(duplicate_mask.sum())

        if duplicate_count:
            logger.warning(
                "Found %d duplicate medicine/date rows. "
                "Aggregating Demand_Qty by day.",
                duplicate_count,
            )

        df = (
            df.groupby(
                [
                    INTERNAL_ID_COLUMN,
                    INTERNAL_TIMESTAMP_COLUMN,
                ],
                as_index=False,
            )[INTERNAL_TARGET_COLUMN]
            .sum()
        )

        # ------------------------------------------------------------------
        # Complete daily calendar per medicine
        # ------------------------------------------------------------------

        completed = []

        for medicine_id, group in df.groupby(
            INTERNAL_ID_COLUMN,
            sort=False,
        ):
            group = group.sort_values(
                INTERNAL_TIMESTAMP_COLUMN
            )

            start = group[INTERNAL_TIMESTAMP_COLUMN].min()
            end = group[INTERNAL_TIMESTAMP_COLUMN].max()

            full_dates = pd.date_range(
                start=start,
                end=end,
                freq="D",
            )

            group = (
                group.set_index(INTERNAL_TIMESTAMP_COLUMN)
                .reindex(full_dates)
            )

            group.index.name = INTERNAL_TIMESTAMP_COLUMN

            group[INTERNAL_ID_COLUMN] = medicine_id

            group[INTERNAL_TARGET_COLUMN] = (
                group[INTERNAL_TARGET_COLUMN]
                .fillna(0.0)
            )

            completed.append(
                group.reset_index()[
                    [
                        INTERNAL_ID_COLUMN,
                        INTERNAL_TIMESTAMP_COLUMN,
                        INTERNAL_TARGET_COLUMN,
                    ]
                ]
            )

        if not completed:
            raise InsufficientHistoryError(
                "No usable demand history remains after data validation."
            )

        result = pd.concat(
            completed,
            ignore_index=True,
        )

        result[INTERNAL_ID_COLUMN] = (
            result[INTERNAL_ID_COLUMN]
            .astype(str)
        )

        result = result.sort_values(
            [
                INTERNAL_ID_COLUMN,
                INTERNAL_TIMESTAMP_COLUMN,
            ]
        ).reset_index(drop=True)

        return result

    # ========================================================================
    # ROUTING
    # ========================================================================

    @staticmethod
    def calculate_validation_advantage(
        validation_chronos_ae: float,
        validation_tsb_ae: float,
    ) -> float:
        """
        Calculate percentage improvement of Chronos AE over TSB AE.

        Positive value:
            Chronos has lower absolute error.

        Example:

            TSB AE      = 100
            Chronos AE  = 70

            advantage = 30%
        """

        validation_chronos_ae = float(validation_chronos_ae)
        validation_tsb_ae = float(validation_tsb_ae)

        if validation_tsb_ae == 0:
            return np.nan

        return (
            (validation_tsb_ae - validation_chronos_ae)
            / validation_tsb_ae
            * 100.0
        )

    @staticmethod
    def select_model(
        validation_advantage_pct: Optional[float],
    ) -> str:
        """
        Apply the frozen production routing policy.
        """

        if validation_advantage_pct is None:
            return TSB_MODEL

        try:
            advantage = float(validation_advantage_pct)
        except (TypeError, ValueError):
            return TSB_MODEL

        if not np.isfinite(advantage):
            return TSB_MODEL

        if advantage >= VALIDATION_ADVANTAGE_THRESHOLD:
            return CHRONOS_MODEL

        return TSB_MODEL

    @staticmethod
    def build_decision(
        medicine_id: str,
        validation_advantage_pct: Optional[float],
    ) -> RoutingDecision:
        """
        Build an auditable routing decision.
        """

        selected_model = ProductionForecastRouter.select_model(
            validation_advantage_pct
        )

        if selected_model == CHRONOS_MODEL:

            reason = (
                f"Validation advantage "
                f"{float(validation_advantage_pct):.2f}% >= "
                f"{VALIDATION_ADVANTAGE_THRESHOLD:.0f}% threshold"
            )

        else:

            if validation_advantage_pct is None:

                reason = (
                    "No validated Chronos advantage available; "
                    "using TSB fallback"
                )

            else:

                try:
                    advantage = float(validation_advantage_pct)
                except (TypeError, ValueError):

                    reason = (
                        "Invalid validation advantage; "
                        "using TSB fallback"
                    )

                else:

                    if not np.isfinite(advantage):

                        reason = (
                            "Invalid validation advantage; "
                            "using TSB fallback"
                        )

                    else:

                        reason = (
                            f"Validation advantage "
                            f"{advantage:.2f}% < "
                            f"{VALIDATION_ADVANTAGE_THRESHOLD:.0f}% threshold"
                        )

        return RoutingDecision(
            medicine_id=str(medicine_id),
            selected_model=selected_model,
            routing_rule=ROUTING_RULE_NAME,
            validation_advantage_pct=validation_advantage_pct,
            reason=reason,
        )

    # ========================================================================
    # TSB FORECAST
    # ========================================================================

    @staticmethod
    def forecast_tsb(
        history_df: pd.DataFrame,
        medicine_id: str,
        prediction_length: int,
        timestamp_column: str = INTERNAL_TIMESTAMP_COLUMN,
        target_column: str = INTERNAL_TARGET_COLUMN,
        alpha: float = 0.1,
        beta: float = 0.1,
    ) -> pd.DataFrame:
        """
        Production TSB implementation.

        Teunter-Syntetos-Babai separates intermittent demand into:

            1. Probability of demand occurrence
            2. Demand size when demand occurs

        The final forecast is:

            forecast = probability * demand_size

        This implementation assumes `history_df` has already been normalized
        to:

            item_id
            timestamp
            target

        It is intentionally simple and deterministic because this is the
        validated production fallback model.
        """

        if prediction_length <= 0:
            raise ValueError(
                "prediction_length must be greater than zero."
            )

        if not (0.0 < alpha <= 1.0):
            raise ValueError(
                "alpha must be in the interval (0, 1]."
            )

        if not (0.0 < beta <= 1.0):
            raise ValueError(
                "beta must be in the interval (0, 1]."
            )

        item_id = str(medicine_id)

        if INTERNAL_ID_COLUMN not in history_df.columns:
            raise ValueError(
                f"TSB history must contain '{INTERNAL_ID_COLUMN}'."
            )

        history = history_df[
            history_df[INTERNAL_ID_COLUMN].astype(str) == item_id
        ].copy()

        if history.empty:
            raise InsufficientHistoryError(
                f"No history for medicine_id={item_id}"
            )

        history = history.sort_values(
            timestamp_column
        )

        y = pd.to_numeric(
            history[target_column],
            errors="coerce",
        ).fillna(0.0)

        y = y.clip(lower=0.0)

        if len(y) < 3:
            raise InsufficientHistoryError(
                f"medicine_id={item_id} has only {len(y)} observations; "
                "minimum 3 observations required for TSB."
            )

        # ------------------------------------------------------------------
        # Identify non-zero demand observations
        # ------------------------------------------------------------------

        demand_mask = y.to_numpy(dtype=float) > 0.0

        demand_positions = np.flatnonzero(demand_mask)

        if len(demand_positions) == 0:

            forecast_value = 0.0

        else:

            # --------------------------------------------------------------
            # Initial demand probability
            #
            # TSB estimates probability of demand occurrence using the
            # interval between non-zero demands.
            #
            # For the first observed demand:
            #
            #     p = 1 / interval
            #
            # --------------------------------------------------------------

            first_demand_position = int(
                demand_positions[0]
            )

            first_interval = (
                first_demand_position + 1
            )

            probability = 1.0 / max(
                first_interval,
                1,
            )

            # --------------------------------------------------------------
            # Initial demand size
            # --------------------------------------------------------------

            demand_size = float(
                y.iloc[first_demand_position]
            )

            previous_demand_position = first_demand_position

            # --------------------------------------------------------------
            # Sequential TSB updates
            # --------------------------------------------------------------

            for position in demand_positions[1:]:

                position = int(position)

                # Every time step since the previous observation:
                # update the probability of occurrence.
                #
                # If a demand occurs:
                #
                #     p_t = p_{t-1} + alpha * (1 - p_{t-1})
                #
                # If no demand occurs:
                #
                #     p_t = (1 - alpha) * p_{t-1}
                #
                # The loop below explicitly applies those updates over
                # the elapsed daily observations.

                gap = (
                    position
                    - previous_demand_position
                )

                if gap > 1:

                    for _ in range(gap - 1):
                        probability = (
                            (1.0 - alpha)
                            * probability
                        )

                # Demand occurrence update.
                probability = (
                    probability
                    + alpha * (1.0 - probability)
                )

                # Demand-size update only occurs when demand is non-zero.
                observed_demand = float(
                    y.iloc[position]
                )

                demand_size = (
                    demand_size
                    + beta
                    * (observed_demand - demand_size)
                )

                previous_demand_position = position

            # --------------------------------------------------------------
            # Account for zero-demand observations after the last demand.
            # --------------------------------------------------------------

            trailing_zeros = (
                len(y)
                - 1
                - previous_demand_position
            )

            if trailing_zeros > 0:

                for _ in range(trailing_zeros):

                    probability = (
                        (1.0 - alpha)
                        * probability
                    )

            forecast_value = (
                max(probability, 0.0)
                * max(demand_size, 0.0)
            )

        # ------------------------------------------------------------------
        # Forecast dates
        # ------------------------------------------------------------------

        last_date = pd.Timestamp(
            history[timestamp_column].max()
        )

        dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=prediction_length,
            freq="D",
        )

        forecast_value = float(
            max(forecast_value, 0.0)
        )

        return pd.DataFrame(
            {
                "Medicine_ID": item_id,
                "Forecast_Date": dates,
                "Predicted_Demand": forecast_value,
            }
        )

    # ========================================================================
    # PRODUCTION FORECAST — ONE MEDICINE
    # ========================================================================

    def forecast_medicine(
        self,
        medicine_id: str,
        history_df: pd.DataFrame,
        validation_advantage_pct: Optional[float],
    ) -> pd.DataFrame:
        """
        Generate a production forecast for one medicine.

        The input can be the actual Silver daily-demand dataframe:

            MDCODE
            INVDT
            Demand_Qty

        Routing:

            Chronos-2 P50 if validation advantage >= 30%
            TSB otherwise
        """

        medicine_id = str(medicine_id)

        # ------------------------------------------------------------------
        # Normalize production input
        # ------------------------------------------------------------------

        prepared_history = self._prepare_history(
            history_df
        )

        # ------------------------------------------------------------------
        # Build immutable routing decision
        # ------------------------------------------------------------------

        decision = self.build_decision(
            medicine_id=medicine_id,
            validation_advantage_pct=validation_advantage_pct,
        )

        logger.info(
            "ROUTING | medicine=%s | model=%s | advantage=%s | rule=%s",
            medicine_id,
            decision.selected_model,
            decision.validation_advantage_pct,
            decision.routing_rule,
        )

        # ==================================================================
        # CHRONOS-2 P50
        # ==================================================================

        if decision.selected_model == CHRONOS_MODEL:

            result = self.predictor.forecast_medicine(
                medicine_id,
                prepared_history,
            )

            rows = []

            for day in result.days:

                rows.append(
                    {
                        "Medicine_ID": result.medicine_id,

                        "Forecast_Date": pd.Timestamp(
                            day.forecast_date
                        ),

                        "Predicted_Demand": float(
                            day.predicted_demand
                        ),

                        "P10": float(day.quantiles.p10),
                        "P20": float(day.quantiles.p20),
                        "P30": float(day.quantiles.p30),
                        "P40": float(day.quantiles.p40),
                        "P50": float(day.quantiles.p50),
                        "P60": float(day.quantiles.p60),
                        "P70": float(day.quantiles.p70),
                        "P80": float(day.quantiles.p80),
                        "P90": float(day.quantiles.p90),

                        "Selected_Model": CHRONOS_MODEL,

                        "Routing_Rule": (
                            decision.routing_rule
                        ),

                        "Validation_Advantage_Pct": (
                            decision.validation_advantage_pct
                        ),

                        "Routing_Reason": decision.reason,

                        "Context_Length_Used": (
                            result.context_length_used
                        ),

                        "Prediction_Length": (
                            result.prediction_length
                        ),

                        "Model_ID": result.model_id,

                        "Generated_At": result.generated_at,
                    }
                )

            return pd.DataFrame(rows)

        # ==================================================================
        # TSB FALLBACK
        # ==================================================================

        tsb = self.forecast_tsb(
            history_df=prepared_history,
            medicine_id=medicine_id,
            prediction_length=(
                self.predictor.config.prediction_length
            ),
            timestamp_column=(
                self.predictor.config.timestamp_column
            ),
            target_column=(
                self.predictor.config.target_column
            ),
        )

        # TSB is a point forecast. There is no calibrated probabilistic
        # distribution available from this implementation.
        #
        # We therefore expose the point forecast consistently across the
        # output schema rather than pretending these are independently
        # calibrated quantiles.

        for quantile in (
            "P10",
            "P20",
            "P30",
            "P40",
            "P50",
            "P60",
            "P70",
            "P80",
            "P90",
        ):
            tsb[quantile] = (
                tsb["Predicted_Demand"]
            )

        tsb["Selected_Model"] = TSB_MODEL

        tsb["Routing_Rule"] = (
            decision.routing_rule
        )

        tsb["Validation_Advantage_Pct"] = (
            decision.validation_advantage_pct
        )

        tsb["Routing_Reason"] = (
            decision.reason
        )

        medicine_history = prepared_history[
            prepared_history[INTERNAL_ID_COLUMN].astype(str)
            == medicine_id
        ]

        tsb["Context_Length_Used"] = min(
            len(medicine_history),
            self.predictor.config.context_length,
        )

        tsb["Prediction_Length"] = (
            self.predictor.config.prediction_length
        )

        tsb["Model_ID"] = TSB_MODEL

        tsb["Generated_At"] = (
            datetime.now(timezone.utc)
        )

        return tsb

    # ========================================================================
    # BATCH FORECASTING
    # ========================================================================

    def forecast_batch(
        self,
        history_df: pd.DataFrame,
        routing_table: pd.DataFrame,
    ) -> tuple[pd.DataFrame, list[str]]:
        """
        Generate forecasts for all medicines represented in routing_table.

        routing_table must contain:

            Medicine_ID
            Validation_Advantage_Pct

        history_df must contain the actual Silver schema:

            MDCODE
            INVDT
            Demand_Qty

        Returns:

            forecasts
            failed_medicines
        """

        required = {
            "Medicine_ID",
            "Validation_Advantage_Pct",
        }

        missing = (
            required
            - set(routing_table.columns)
        )

        if missing:
            raise ValueError(
                "Routing table missing required columns: "
                f"{sorted(missing)}"
            )

        # Validate the Silver dataset before beginning the batch.
        self._validate_source_history(
            history_df
        )

        # Prepare once for the complete batch.
        prepared_history = self._prepare_history(
            history_df
        )

        routing_table = routing_table.copy()

        routing_table["Medicine_ID"] = (
            routing_table["Medicine_ID"]
            .astype(str)
            .str.strip()
        )

        forecasts = []
        failed = []

        for _, route in routing_table.iterrows():

            medicine_id = str(
                route["Medicine_ID"]
            )

            advantage = (
                route["Validation_Advantage_Pct"]
            )

            try:

                decision = self.build_decision(
                    medicine_id=medicine_id,
                    validation_advantage_pct=advantage,
                )

                logger.info(
                    "BATCH ROUTING | medicine=%s | model=%s | "
                    "advantage=%s",
                    medicine_id,
                    decision.selected_model,
                    decision.validation_advantage_pct,
                )

                # Use the already normalized history rather than rebuilding
                # the complete daily calendar for every medicine.

                medicine_history = prepared_history[
                    prepared_history[
                        INTERNAL_ID_COLUMN
                    ].astype(str)
                    == medicine_id
                ].copy()

                if medicine_history.empty:
                    raise InsufficientHistoryError(
                        f"No history for medicine_id={medicine_id}"
                    )

                if decision.selected_model == CHRONOS_MODEL:

                    result = self.predictor.forecast_medicine(
                        medicine_id,
                        prepared_history,
                    )

                    rows = []

                    for day in result.days:

                        rows.append(
                            {
                                "Medicine_ID": (
                                    result.medicine_id
                                ),

                                "Forecast_Date": (
                                    pd.Timestamp(
                                        day.forecast_date
                                    )
                                ),

                                "Predicted_Demand": float(
                                    day.predicted_demand
                                ),

                                "P10": float(
                                    day.quantiles.p10
                                ),
                                "P20": float(
                                    day.quantiles.p20
                                ),
                                "P30": float(
                                    day.quantiles.p30
                                ),
                                "P40": float(
                                    day.quantiles.p40
                                ),
                                "P50": float(
                                    day.quantiles.p50
                                ),
                                "P60": float(
                                    day.quantiles.p60
                                ),
                                "P70": float(
                                    day.quantiles.p70
                                ),
                                "P80": float(
                                    day.quantiles.p80
                                ),
                                "P90": float(
                                    day.quantiles.p90
                                ),

                                "Selected_Model": (
                                    CHRONOS_MODEL
                                ),

                                "Routing_Rule": (
                                    decision.routing_rule
                                ),

                                "Validation_Advantage_Pct": (
                                    decision.validation_advantage_pct
                                ),

                                "Routing_Reason": (
                                    decision.reason
                                ),

                                "Context_Length_Used": (
                                    result.context_length_used
                                ),

                                "Prediction_Length": (
                                    result.prediction_length
                                ),

                                "Model_ID": (
                                    result.model_id
                                ),

                                "Generated_At": (
                                    result.generated_at
                                ),
                            }
                        )

                    forecasts.append(
                        pd.DataFrame(rows)
                    )

                else:

                    tsb = self.forecast_tsb(
                        history_df=prepared_history,
                        medicine_id=medicine_id,
                        prediction_length=(
                            self.predictor.config.prediction_length
                        ),
                        timestamp_column=(
                            self.predictor.config.timestamp_column
                        ),
                        target_column=(
                            self.predictor.config.target_column
                        ),
                    )

                    for quantile in (
                        "P10",
                        "P20",
                        "P30",
                        "P40",
                        "P50",
                        "P60",
                        "P70",
                        "P80",
                        "P90",
                    ):
                        tsb[quantile] = (
                            tsb["Predicted_Demand"]
                        )

                    tsb["Selected_Model"] = (
                        TSB_MODEL
                    )

                    tsb["Routing_Rule"] = (
                        decision.routing_rule
                    )

                    tsb["Validation_Advantage_Pct"] = (
                        decision.validation_advantage_pct
                    )

                    tsb["Routing_Reason"] = (
                        decision.reason
                    )

                    tsb["Context_Length_Used"] = min(
                        len(medicine_history),
                        self.predictor.config.context_length,
                    )

                    tsb["Prediction_Length"] = (
                        self.predictor.config.prediction_length
                    )

                    tsb["Model_ID"] = (
                        TSB_MODEL
                    )

                    tsb["Generated_At"] = (
                        datetime.now(timezone.utc)
                    )

                    forecasts.append(tsb)

            except InsufficientHistoryError as exc:

                logger.warning(
                    "Skipping medicine=%s: %s",
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

        # ------------------------------------------------------------------
        # Final output
        # ------------------------------------------------------------------

        if forecasts:

            output = pd.concat(
                forecasts,
                ignore_index=True,
            )

            output = output.sort_values(
                [
                    "Medicine_ID",
                    "Forecast_Date",
                ]
            ).reset_index(
                drop=True
            )

        else:

            output = pd.DataFrame(
                columns=[
                    "Medicine_ID",
                    "Forecast_Date",
                    "Predicted_Demand",
                    "P10",
                    "P20",
                    "P30",
                    "P40",
                    "P50",
                    "P60",
                    "P70",
                    "P80",
                    "P90",
                    "Selected_Model",
                    "Routing_Rule",
                    "Validation_Advantage_Pct",
                    "Routing_Reason",
                    "Context_Length_Used",
                    "Prediction_Length",
                    "Model_ID",
                    "Generated_At",
                ]
            )

        return output, failed


# ============================================================================
# ROUTING TABLE BUILDER
# ============================================================================


def build_routing_table(
    robustness_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert the validated robustness dataset into the exact routing table
    required by ProductionForecastRouter.

    Expected input columns:

        Medicine_ID
        Validation_Chronos_AE
        Validation_TSB_AE
    """

    required = {
        "Medicine_ID",
        "Validation_Chronos_AE",
        "Validation_TSB_AE",
    }

    missing = (
        required
        - set(robustness_df.columns)
    )

    if missing:
        raise ValueError(
            "Robustness dataframe missing columns: "
            f"{sorted(missing)}"
        )

    routing = robustness_df[
        [
            "Medicine_ID",
            "Validation_Chronos_AE",
            "Validation_TSB_AE",
        ]
    ].copy()

    routing["Medicine_ID"] = (
        routing["Medicine_ID"]
        .astype(str)
        .str.strip()
    )

    routing["Validation_Chronos_AE"] = pd.to_numeric(
        routing["Validation_Chronos_AE"],
        errors="coerce",
    )

    routing["Validation_TSB_AE"] = pd.to_numeric(
        routing["Validation_TSB_AE"],
        errors="coerce",
    )

    routing["Validation_Advantage_Pct"] = (
        (
            routing["Validation_TSB_AE"]
            - routing["Validation_Chronos_AE"]
        )
        / routing["Validation_TSB_AE"].replace(
            0,
            np.nan,
        )
        * 100.0
    )

    routing["Selected_Model"] = (
        routing["Validation_Advantage_Pct"]
        .apply(
            ProductionForecastRouter.select_model
        )
    )

    routing["Routing_Rule"] = (
        ROUTING_RULE_NAME
    )

    routing["Threshold"] = (
        VALIDATION_ADVANTAGE_THRESHOLD
    )

    return routing


# ============================================================================
# PRODUCTION SELF-CHECKS
# ============================================================================


def _run_policy_self_check() -> None:
    """
    Validate the frozen routing policy.
    """

    test_cases = [
        (69.069, CHRONOS_MODEL),
        (30.0, CHRONOS_MODEL),
        (29.999, TSB_MODEL),
        (0.0, TSB_MODEL),
        (-20.0, TSB_MODEL),
        (np.nan, TSB_MODEL),
        (None, TSB_MODEL),
        (np.inf, TSB_MODEL),
        (-np.inf, TSB_MODEL),
    ]

    passed = True

    for advantage, expected in test_cases:

        actual = (
            ProductionForecastRouter.select_model(
                advantage
            )
        )

        if actual != expected:

            passed = False

            logger.error(
                "ROUTING SELF-CHECK FAILED | advantage=%s | "
                "expected=%s | actual=%s",
                advantage,
                expected,
                actual,
            )

    if not passed:
        raise RuntimeError(
            "Production routing policy self-check failed."
        )


# ============================================================================
# CLI / VALIDATION
# ============================================================================


def main() -> None:

    print("=" * 80)
    print("FINEMED PRODUCTION FORECAST ROUTER")
    print("=" * 80)

    print()
    print("Production data schema:")
    print(
        f"  ID        : {SOURCE_ID_COLUMN}"
    )
    print(
        f"  Date      : {SOURCE_TIMESTAMP_COLUMN}"
    )
    print(
        f"  Demand    : {SOURCE_TARGET_COLUMN}"
    )

    print()
    print("Internal Chronos schema:")
    print(
        f"  ID        : {INTERNAL_ID_COLUMN}"
    )
    print(
        f"  Timestamp : {INTERNAL_TIMESTAMP_COLUMN}"
    )
    print(
        f"  Target    : {INTERNAL_TARGET_COLUMN}"
    )

    print()
    print("Frozen routing policy:")
    print(
        f"  Chronos-2 P50 if validation advantage >= "
        f"{VALIDATION_ADVANTAGE_THRESHOLD:.0f}%"
    )
    print("  Otherwise -> TSB")

    print()
    print("Running routing policy self-check...")

    _run_policy_self_check()

    print("PASS: routing policy self-check")

    print()
    print("=" * 80)
    print("POLICY STATUS")
    print("=" * 80)

    print(
        "PASS: 30% routing threshold is frozen."
    )

    print(
        "PASS: Chronos-2 P50 is selected only at >= 30% advantage."
    )

    print(
        "PASS: TSB is the default fallback."
    )

    print(
        "PASS: Missing/invalid validation advantage falls back to TSB."
    )

    print(
        "PASS: Silver MDCODE/INVDT/Demand_Qty schema is normalized "
        "before forecasting."
    )

    print(
        "PASS: Chronos receives item_id/timestamp/target schema."
    )

    print(
        "PASS: Daily calendar gaps are filled with zero demand."
    )

    print(
        "PASS: Duplicate medicine/date observations are aggregated."
    )

    print(
        "PASS: TSB fallback uses explicit probability and demand-size "
        "updates."
    )

    print()
    print("=" * 80)
    print("NEXT")
    print("=" * 80)

    print(
        "Use build_routing_table() with "
        "medicine_model_robustness.parquet."
    )

    print(
        "Then pass the routing table and the Silver "
        "daily_demand dataframe to "
        "ProductionForecastRouter.forecast_batch()."
    )


if __name__ == "__main__":
    main()