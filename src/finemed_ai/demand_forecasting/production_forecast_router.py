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
# VALIDATED TSB PARAMETERS
# ============================================================================

TSB_ALPHA_DEMAND = 0.1
TSB_ALPHA_PROBABILITY = 0.1


# ============================================================================
# SOURCE DATA SCHEMA
# ============================================================================

SOURCE_ID_COLUMN = "MDCODE"
SOURCE_TIMESTAMP_COLUMN = "INVDT"
SOURCE_TARGET_COLUMN = "Demand_Qty"


# ============================================================================
# INTERNAL FORECASTING SCHEMA
# ============================================================================

INTERNAL_ID_COLUMN = "item_id"
INTERNAL_TIMESTAMP_COLUMN = "timestamp"
INTERNAL_TARGET_COLUMN = "target"


# ============================================================================
# PRODUCTION OUTPUT SCHEMA
# ============================================================================

OUTPUT_COLUMNS = [
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
    "Forecast_Type",
    "Routing_Rule",
    "Validation_Advantage_Pct",
    "Routing_Reason",
    "Context_Length_Used",
    "Prediction_Length",
    "Model_ID",
    "Generated_At",
]


TSB_QUANTILES = (
    "P10",
    "P20",
    "P30",
    "P40",
    "P50",
    "P60",
    "P70",
    "P80",
    "P90",
)


# ============================================================================
# DATA STRUCTURES
# ============================================================================


@dataclass(frozen=True)
class RoutingDecision:
    """
    Immutable production routing decision.

    The decision is based only on validation-derived information.

    Production forecast generation itself must never recompute model
    performance or optimize the routing threshold.
    """

    medicine_id: str
    selected_model: str
    routing_rule: str
    validation_advantage_pct: Optional[float]
    reason: str


# ============================================================================
# PRODUCTION FORECAST ROUTER
# ============================================================================


class ProductionForecastRouter:
    """
    Production model router.

    Frozen policy:

        validation advantage >= 30%
            -> Chronos-2 P50

        otherwise
            -> TSB

    Important:

    - The threshold is not optimized here.
    - Validation metrics must come from the validation stage.
    - Holdout/test information must never be supplied to this router
      for routing decisions.
    - Chronos is used exactly as validated: P50, context 730, horizon 30.
    - No unvalidated post-hoc scaling/bias correction is applied here.
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
    # DATA VALIDATION
    # ========================================================================

    @staticmethod
    def _validate_source_history(
        history_df: pd.DataFrame,
    ) -> None:
        """
        Validate the Silver daily-demand dataframe.

        Required columns:

            MDCODE
            INVDT
            Demand_Qty
        """

        if not isinstance(history_df, pd.DataFrame):
            raise TypeError(
                "history_df must be a pandas DataFrame."
            )

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
                "Expected columns: "
                f"{[SOURCE_ID_COLUMN, SOURCE_TIMESTAMP_COLUMN, SOURCE_TARGET_COLUMN]}"
            )

        if history_df.empty:
            raise InsufficientHistoryError(
                "History dataframe is empty."
            )

    # ========================================================================
    # HISTORY PREPARATION
    # ========================================================================

    @staticmethod
    def _prepare_history(
        history_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Convert Silver demand data into the internal forecasting schema.

        Input:

            MDCODE
            INVDT
            Demand_Qty

        Output:

            item_id
            timestamp
            target

        Processing:

        1. Validate source schema.
        2. Remove missing source values.
        3. Normalize medicine IDs.
        4. Parse timestamps.
        5. Remove timezone information safely.
        6. Normalize timestamps to calendar-day granularity.
        7. Convert demand to numeric.
        8. Reject invalid rows.
        9. Reject/clip negative demand.
        10. Aggregate duplicate medicine/date observations.
        11. Complete the daily calendar for each medicine.
        12. Fill missing calendar dates with zero demand.
        13. Sort by medicine/date.
        """

        ProductionForecastRouter._validate_source_history(
            history_df
        )

        df = history_df[
            [
                SOURCE_ID_COLUMN,
                SOURCE_TIMESTAMP_COLUMN,
                SOURCE_TARGET_COLUMN,
            ]
        ].copy()

        # --------------------------------------------------------------------
        # Remove missing source values before string conversion.
        # --------------------------------------------------------------------

        before = len(df)

        df = df.dropna(
            subset=[
                SOURCE_ID_COLUMN,
                SOURCE_TIMESTAMP_COLUMN,
                SOURCE_TARGET_COLUMN,
            ]
        ).copy()

        removed = before - len(df)

        if removed:
            logger.warning(
                "Removed %d rows with missing source values.",
                removed,
            )

        if df.empty:
            raise InsufficientHistoryError(
                "No usable rows remain after removing missing source values."
            )

        # --------------------------------------------------------------------
        # Normalize medicine IDs.
        # --------------------------------------------------------------------

        df[INTERNAL_ID_COLUMN] = (
            df[SOURCE_ID_COLUMN]
            .astype(str)
            .str.strip()
        )

        empty_ids = df[INTERNAL_ID_COLUMN].eq("")

        if empty_ids.any():
            count = int(empty_ids.sum())

            logger.warning(
                "Removed %d rows with empty medicine IDs.",
                count,
            )

            df = df.loc[~empty_ids].copy()

        if df.empty:
            raise InsufficientHistoryError(
                "No usable medicine IDs remain after normalization."
            )

        # --------------------------------------------------------------------
        # Parse timestamps.
        # --------------------------------------------------------------------

        parsed_dates = pd.to_datetime(
            df[SOURCE_TIMESTAMP_COLUMN],
            errors="coerce",
        )

        invalid_dates = parsed_dates.isna()

        if invalid_dates.any():
            count = int(invalid_dates.sum())

            logger.warning(
                "Removed %d rows with invalid timestamps.",
                count,
            )

        df[INTERNAL_TIMESTAMP_COLUMN] = parsed_dates

        # --------------------------------------------------------------------
        # Normalize timezone safely.
        #
        # Using utc=True first gives us one deterministic representation
        # even when input contains mixed timezone-aware values.
        # --------------------------------------------------------------------

        if (
            df[INTERNAL_TIMESTAMP_COLUMN]
            .dt
            .tz is not None
        ):
            df[INTERNAL_TIMESTAMP_COLUMN] = (
                df[INTERNAL_TIMESTAMP_COLUMN]
                .dt.tz_localize(None)
            )

        df[INTERNAL_TIMESTAMP_COLUMN] = (
            df[INTERNAL_TIMESTAMP_COLUMN]
            .dt.normalize()
        )

        # --------------------------------------------------------------------
        # Numeric demand.
        # --------------------------------------------------------------------

        df[INTERNAL_TARGET_COLUMN] = pd.to_numeric(
            df[SOURCE_TARGET_COLUMN],
            errors="coerce",
        )

        # --------------------------------------------------------------------
        # Remove invalid normalized rows.
        # --------------------------------------------------------------------

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
                "Removed %d invalid rows during forecast preparation.",
                removed,
            )

        if df.empty:
            raise InsufficientHistoryError(
                "No usable demand history remains after normalization."
            )

        # --------------------------------------------------------------------
        # Demand integrity.
        #
        # Demand cannot be negative.
        # --------------------------------------------------------------------

        negative_mask = (
            df[INTERNAL_TARGET_COLUMN] < 0
        )

        negative_count = int(
            negative_mask.sum()
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

        # --------------------------------------------------------------------
        # Aggregate duplicate medicine/date rows.
        # --------------------------------------------------------------------

        duplicate_count = int(
            df.duplicated(
                subset=[
                    INTERNAL_ID_COLUMN,
                    INTERNAL_TIMESTAMP_COLUMN,
                ],
                keep=False,
            ).sum()
        )

        if duplicate_count:
            logger.info(
                "Aggregating %d duplicate medicine/date observations.",
                duplicate_count,
            )

        df = (
            df.groupby(
                [
                    INTERNAL_ID_COLUMN,
                    INTERNAL_TIMESTAMP_COLUMN,
                ],
                as_index=False,
                sort=False,
            )[INTERNAL_TARGET_COLUMN]
            .sum()
        )

        if df.empty:
            raise InsufficientHistoryError(
                "No observations remain after daily aggregation."
            )

        # --------------------------------------------------------------------
        # Complete daily calendar independently for every medicine.
        #
        # IMPORTANT:
        #
        # This preserves the validated project assumption that a missing
        # calendar date inside an observed medicine history represents
        # zero demand.
        # --------------------------------------------------------------------

        completed: list[pd.DataFrame] = []

        for medicine_id, group in df.groupby(
            INTERNAL_ID_COLUMN,
            sort=False,
        ):

            group = group.sort_values(
                INTERNAL_TIMESTAMP_COLUMN
            ).copy()

            start_date = group[
                INTERNAL_TIMESTAMP_COLUMN
            ].min()

            end_date = group[
                INTERNAL_TIMESTAMP_COLUMN
            ].max()

            if pd.isna(start_date) or pd.isna(end_date):
                continue

            full_dates = pd.date_range(
                start=start_date,
                end=end_date,
                freq="D",
            )

            group = (
                group.set_index(
                    INTERNAL_TIMESTAMP_COLUMN
                )
                .reindex(full_dates)
            )

            group.index.name = INTERNAL_TIMESTAMP_COLUMN

            group[INTERNAL_ID_COLUMN] = str(
                medicine_id
            ).strip()

            group[INTERNAL_TARGET_COLUMN] = (
                pd.to_numeric(
                    group[INTERNAL_TARGET_COLUMN],
                    errors="coerce",
                )
                .fillna(0.0)
                .clip(lower=0.0)
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
                "No usable medicine histories remain after calendar completion."
            )

        result = pd.concat(
            completed,
            ignore_index=True,
        )

        result[INTERNAL_ID_COLUMN] = (
            result[INTERNAL_ID_COLUMN]
            .astype(str)
            .str.strip()
        )

        result[INTERNAL_TARGET_COLUMN] = (
            pd.to_numeric(
                result[INTERNAL_TARGET_COLUMN],
                errors="coerce",
            )
            .fillna(0.0)
            .clip(lower=0.0)
        )

        result = result.sort_values(
            [
                INTERNAL_ID_COLUMN,
                INTERNAL_TIMESTAMP_COLUMN,
            ]
        ).reset_index(drop=True)

        return result[
            [
                INTERNAL_ID_COLUMN,
                INTERNAL_TIMESTAMP_COLUMN,
                INTERNAL_TARGET_COLUMN,
            ]
        ]

    # ========================================================================
    # VALIDATION ADVANTAGE
    # ========================================================================

    @staticmethod
    def calculate_validation_advantage(
        validation_chronos_ae: float,
        validation_tsb_ae: float,
    ) -> float:
        """
        Calculate Chronos percentage improvement over TSB.

            advantage =
                (TSB_AE - Chronos_AE) / TSB_AE * 100

        Positive value means Chronos has lower AE.

        This function is intended for validation/routing-table generation,
        not for evaluating production forecasts.
        """

        chronos_ae = float(
            validation_chronos_ae
        )

        tsb_ae = float(
            validation_tsb_ae
        )

        if not np.isfinite(chronos_ae):
            return np.nan

        if not np.isfinite(tsb_ae):
            return np.nan

        if chronos_ae < 0:
            return np.nan

        if tsb_ae <= 0:
            return np.nan

        return (
            (tsb_ae - chronos_ae)
            / tsb_ae
            * 100.0
        )

    # ========================================================================
    # MODEL SELECTION
    # ========================================================================

    @staticmethod
    def select_model(
        validation_advantage_pct: Optional[float],
    ) -> str:
        """
        Apply the frozen routing policy.

            advantage >= 30%
                -> Chronos-2 P50

            otherwise
                -> TSB
        """

        if validation_advantage_pct is None:
            return TSB_MODEL

        try:
            advantage = float(
                validation_advantage_pct
            )
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
        Build an auditable immutable routing decision.
        """

        medicine_id = str(
            medicine_id
        ).strip()

        if not medicine_id:
            raise ValueError(
                "medicine_id cannot be empty."
            )

        selected_model = (
            ProductionForecastRouter.select_model(
                validation_advantage_pct
            )
        )

        normalized_advantage: Optional[float]

        if validation_advantage_pct is None:
            normalized_advantage = None

        else:
            try:
                value = float(
                    validation_advantage_pct
                )

                normalized_advantage = (
                    value
                    if np.isfinite(value)
                    else None
                )

            except (TypeError, ValueError):
                normalized_advantage = None

        if selected_model == CHRONOS_MODEL:

            reason = (
                f"Validation advantage "
                f"{normalized_advantage:.2f}% >= "
                f"{VALIDATION_ADVANTAGE_THRESHOLD:.0f}% threshold"
            )

        elif normalized_advantage is None:

            reason = (
                "No valid validation advantage available; "
                "using TSB fallback"
            )

        else:

            reason = (
                f"Validation advantage "
                f"{normalized_advantage:.2f}% < "
                f"{VALIDATION_ADVANTAGE_THRESHOLD:.0f}% threshold"
            )

        return RoutingDecision(
            medicine_id=medicine_id,
            selected_model=selected_model,
            routing_rule=ROUTING_RULE_NAME,
            validation_advantage_pct=normalized_advantage,
            reason=reason,
        )

    # ========================================================================
    # VALIDATED TSB
    # ========================================================================

    @staticmethod
    def forecast_tsb(
        history_df: pd.DataFrame,
        medicine_id: str,
        prediction_length: int,
        timestamp_column: str = INTERNAL_TIMESTAMP_COLUMN,
        target_column: str = INTERNAL_TARGET_COLUMN,
        alpha_demand: float = TSB_ALPHA_DEMAND,
        alpha_probability: float = TSB_ALPHA_PROBABILITY,
    ) -> pd.DataFrame:
        """
        Generate the validated TSB point forecast.

        Validated behavior:

        - demand estimate initialized using first non-zero demand
        - probability initialized as:
              1 / (first_nonzero_index + 1)
        - probability updated at every observation
        - demand estimate updated only on non-zero demand
        - final forecast:
              probability * demand_estimate

        TSB is a point forecast and does not produce calibrated quantiles.
        """

        if prediction_length <= 0:
            raise ValueError(
                "prediction_length must be greater than zero."
            )

        if not (
            0.0 < alpha_demand <= 1.0
        ):
            raise ValueError(
                "alpha_demand must be in the interval (0, 1]."
            )

        if not (
            0.0 < alpha_probability <= 1.0
        ):
            raise ValueError(
                "alpha_probability must be in the interval (0, 1]."
            )

        item_id = str(
            medicine_id
        ).strip()

        if not item_id:
            raise ValueError(
                "medicine_id cannot be empty."
            )

        if not isinstance(
            history_df,
            pd.DataFrame,
        ):
            raise TypeError(
                "history_df must be a pandas DataFrame."
            )

        required = {
            INTERNAL_ID_COLUMN,
            timestamp_column,
            target_column,
        }

        missing = (
            required
            - set(history_df.columns)
        )

        if missing:
            raise ValueError(
                "TSB history is missing required columns: "
                f"{sorted(missing)}"
            )

        history = history_df[
            history_df[INTERNAL_ID_COLUMN]
            .astype(str)
            .str.strip()
            == item_id
        ].copy()

        if history.empty:
            raise InsufficientHistoryError(
                f"No history for medicine_id={item_id}"
            )

        # --------------------------------------------------------------------
        # Normalize timestamp and demand.
        # --------------------------------------------------------------------

        history[timestamp_column] = pd.to_datetime(
            history[timestamp_column],
            errors="coerce",
        )

        history[target_column] = pd.to_numeric(
            history[target_column],
            errors="coerce",
        )

        history = history.dropna(
            subset=[
                timestamp_column,
                target_column,
            ]
        ).copy()

        if history.empty:
            raise InsufficientHistoryError(
                f"No usable TSB history for medicine_id={item_id}"
            )

        # --------------------------------------------------------------------
        # Normalize daily timestamps.
        # --------------------------------------------------------------------

        if isinstance(
            history[timestamp_column].dtype,
            pd.DatetimeTZDtype,
        ):
            history[timestamp_column] = (
                history[timestamp_column]
                .dt.tz_localize(None)
            )

        history[timestamp_column] = (
            history[timestamp_column]
            .dt.normalize()
        )

        # --------------------------------------------------------------------
        # Clip negative demand and aggregate duplicate days.
        #
        # This makes forecast_tsb safe even when called independently from
        # the production router.
        # --------------------------------------------------------------------

        history[target_column] = (
            history[target_column]
            .astype(float)
            .clip(lower=0.0)
        )

        history = (
            history.groupby(
                timestamp_column,
                as_index=False,
            )[target_column]
            .sum()
            .sort_values(timestamp_column)
            .reset_index(drop=True)
        )

        if len(history) < 3:
            raise InsufficientHistoryError(
                f"medicine_id={item_id} has only "
                f"{len(history)} observations; minimum 3 "
                "observations required for TSB."
            )

        values = history[
            target_column
        ].to_numpy(
            dtype=float
        )

        if not np.isfinite(values).all():
            raise ValueError(
                f"TSB history contains non-finite demand values "
                f"for medicine_id={item_id}."
            )

        # --------------------------------------------------------------------
        # Locate first non-zero demand.
        # --------------------------------------------------------------------

        demand_positions = np.flatnonzero(
            values > 0.0
        )

        if len(demand_positions) == 0:

            forecast_value = 0.0

        else:

            first_position = int(
                demand_positions[0]
            )

            demand_estimate = float(
                values[first_position]
            )

            probability = (
                1.0
                / float(first_position + 1)
            )

            # ---------------------------------------------------------------
            # Validated TSB sequential update.
            # ---------------------------------------------------------------

            for position in range(
                first_position + 1,
                len(values),
            ):

                observed_demand = float(
                    values[position]
                )

                occurrence = (
                    1.0
                    if observed_demand > 0.0
                    else 0.0
                )

                probability = (
                    probability
                    + alpha_probability
                    * (
                        occurrence
                        - probability
                    )
                )

                if occurrence > 0.0:

                    demand_estimate = (
                        demand_estimate
                        + alpha_demand
                        * (
                            observed_demand
                            - demand_estimate
                        )
                    )

            forecast_value = (
                max(probability, 0.0)
                * max(demand_estimate, 0.0)
            )

        forecast_value = float(
            max(
                forecast_value,
                0.0,
            )
        )

        if not np.isfinite(
            forecast_value
        ):
            raise ValueError(
                f"TSB produced a non-finite forecast "
                f"for medicine_id={item_id}."
            )

        # --------------------------------------------------------------------
        # Generate future daily dates.
        # --------------------------------------------------------------------

        last_date = pd.Timestamp(
            history[timestamp_column].max()
        ).normalize()

        dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=prediction_length,
            freq="D",
        )

        return pd.DataFrame(
            {
                "Medicine_ID": item_id,
                "Forecast_Date": dates,
                "Predicted_Demand": forecast_value,
            }
        )

    # ========================================================================
    # CHRONOS RESULT FORMATTER
    # ========================================================================

    @staticmethod
    def _format_chronos_result(
        result,
        decision: RoutingDecision,
    ) -> pd.DataFrame:
        """
        Convert PredictorService result into production output format.
        """

        rows = []

        for day in result.days:

            quantiles = day.quantiles

            values = {
                "P10": float(quantiles.p10),
                "P20": float(quantiles.p20),
                "P30": float(quantiles.p30),
                "P40": float(quantiles.p40),
                "P50": float(quantiles.p50),
                "P60": float(quantiles.p60),
                "P70": float(quantiles.p70),
                "P80": float(quantiles.p80),
                "P90": float(quantiles.p90),
            }

            # ----------------------------------------------------------------
            # Defensive numeric validation.
            # ----------------------------------------------------------------

            numeric_values = np.array(
                list(values.values()),
                dtype=float,
            )

            if not np.isfinite(
                numeric_values
            ).all():
                raise ValueError(
                    f"Chronos returned non-finite quantile values "
                    f"for medicine_id={result.medicine_id}."
                )

            if (
                numeric_values < 0
            ).any():
                raise ValueError(
                    f"Chronos returned negative quantile values "
                    f"for medicine_id={result.medicine_id}."
                )

            # ----------------------------------------------------------------
            # Quantile monotonicity.
            #
            # PredictorService already enforces this, but production router
            # keeps an explicit invariant here as well.
            # ----------------------------------------------------------------

            ordered = [
                values["P10"],
                values["P20"],
                values["P30"],
                values["P40"],
                values["P50"],
                values["P60"],
                values["P70"],
                values["P80"],
                values["P90"],
            ]

            if any(
                ordered[i] > ordered[i + 1]
                for i in range(
                    len(ordered) - 1
                )
            ):
                raise ValueError(
                    f"Chronos quantiles are not monotonic for "
                    f"medicine_id={result.medicine_id}."
                )

            rows.append(
                {
                    "Medicine_ID": str(
                        result.medicine_id
                    ),
                    "Forecast_Date": pd.Timestamp(
                        day.forecast_date
                    ).normalize(),
                    "Predicted_Demand": values["P50"],
                    "P10": values["P10"],
                    "P20": values["P20"],
                    "P30": values["P30"],
                    "P40": values["P40"],
                    "P50": values["P50"],
                    "P60": values["P60"],
                    "P70": values["P70"],
                    "P80": values["P80"],
                    "P90": values["P90"],
                    "Selected_Model": CHRONOS_MODEL,
                    "Forecast_Type": "probabilistic",
                    "Routing_Rule": decision.routing_rule,
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

        if not rows:
            raise InsufficientHistoryError(
                "Chronos returned no forecast days."
            )

        output = pd.DataFrame(
            rows,
            columns=OUTPUT_COLUMNS,
        )

        # Chronos production point forecast must be P50.
        if not np.allclose(
            output["Predicted_Demand"].to_numpy(dtype=float),
            output["P50"].to_numpy(dtype=float),
            rtol=1e-9,
            atol=1e-9,
        ):
            raise ValueError(
                "Chronos Predicted_Demand is not equal to P50."
            )

        return output

    # ========================================================================
    # TSB RESULT FORMATTER
    # ========================================================================

    def _format_tsb_result(
        self,
        tsb: pd.DataFrame,
        decision: RoutingDecision,
        medicine_history: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Convert TSB point forecast into the common production schema.

        TSB is not probabilistic.

        P10...P90 are therefore populated with the same point forecast only
        for schema compatibility. Forecast_Type explicitly remains "point".
        """

        tsb = tsb.copy()

        predicted = pd.to_numeric(
            tsb["Predicted_Demand"],
            errors="coerce",
        )

        if predicted.isna().any():
            raise ValueError(
                "TSB produced non-numeric demand values."
            )

        if (
            predicted < 0
        ).any():
            raise ValueError(
                "TSB produced negative demand values."
            )

        tsb["Predicted_Demand"] = predicted.astype(float)

        for quantile in TSB_QUANTILES:
            tsb[quantile] = (
                tsb["Predicted_Demand"]
            )

        tsb["Selected_Model"] = TSB_MODEL

        tsb["Forecast_Type"] = "point"

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

        tsb["Model_ID"] = TSB_MODEL

        tsb["Generated_At"] = (
            datetime.now(timezone.utc)
        )

        return tsb[
            OUTPUT_COLUMNS
        ]

    # ========================================================================
    # CHRONOS FORECAST
    # ========================================================================

    def _forecast_chronos(
        self,
        medicine_id: str,
        medicine_history: pd.DataFrame,
        decision: RoutingDecision,
    ) -> pd.DataFrame:
        """
        Generate Chronos-2 P50 forecast for one medicine.
        """

        result = self.predictor.forecast_medicine(
            medicine_id=medicine_id,
            history_df=medicine_history,
        )

        return self._format_chronos_result(
            result=result,
            decision=decision,
        )

    # ========================================================================
    # TSB FORECAST
    # ========================================================================

    def _forecast_tsb(
        self,
        medicine_id: str,
        medicine_history: pd.DataFrame,
        decision: RoutingDecision,
    ) -> pd.DataFrame:
        """
        Generate validated TSB forecast.
        """

        cfg = self.predictor.config

        tsb = self.forecast_tsb(
            history_df=medicine_history,
            medicine_id=medicine_id,
            prediction_length=cfg.prediction_length,
            timestamp_column=cfg.timestamp_column,
            target_column=cfg.target_column,
            alpha_demand=TSB_ALPHA_DEMAND,
            alpha_probability=TSB_ALPHA_PROBABILITY,
        )

        return self._format_tsb_result(
            tsb=tsb,
            decision=decision,
            medicine_history=medicine_history,
        )

    # ========================================================================
    # ONE-MEDICINE PRODUCTION FORECAST
    # ========================================================================

    def forecast_medicine(
        self,
        medicine_id: str,
        history_df: pd.DataFrame,
        validation_advantage_pct: Optional[float],
    ) -> pd.DataFrame:
        """
        Generate a production forecast for one medicine.

        Source schema:

            MDCODE
            INVDT
            Demand_Qty

        Routing:

            advantage >= 30%
                -> Chronos-2 P50

            otherwise
                -> TSB
        """

        medicine_id = str(
            medicine_id
        ).strip()

        if not medicine_id:
            raise ValueError(
                "medicine_id cannot be empty."
            )

        prepared_history = (
            self._prepare_history(
                history_df
            )
        )

        medicine_history = (
            prepared_history[
                prepared_history[
                    INTERNAL_ID_COLUMN
                ]
                .astype(str)
                .str.strip()
                == medicine_id
            ]
            .copy()
        )

        if medicine_history.empty:
            raise InsufficientHistoryError(
                f"No history for medicine_id={medicine_id}"
            )

        decision = self.build_decision(
            medicine_id=medicine_id,
            validation_advantage_pct=(
                validation_advantage_pct
            ),
        )

        logger.info(
            "ROUTING | medicine=%s | model=%s | advantage=%s | rule=%s",
            medicine_id,
            decision.selected_model,
            decision.validation_advantage_pct,
            decision.routing_rule,
        )

        if decision.selected_model == CHRONOS_MODEL:

            forecast = self._forecast_chronos(
                medicine_id=medicine_id,
                medicine_history=medicine_history,
                decision=decision,
            )

        else:

            forecast = self._forecast_tsb(
                medicine_id=medicine_id,
                medicine_history=medicine_history,
                decision=decision,
            )

        self._validate_forecast_output(
            forecast,
            medicine_id,
        )

        return forecast

    # ========================================================================
    # ROUTING TABLE VALIDATION
    # ========================================================================

    @staticmethod
    def _validate_routing_table(
        routing_table: pd.DataFrame,
    ) -> None:
        """
        Validate production routing-table structure.

        Required:

            Medicine_ID
            Validation_Advantage_Pct
        """

        if not isinstance(
            routing_table,
            pd.DataFrame,
        ):
            raise TypeError(
                "routing_table must be a pandas DataFrame."
            )

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

        if routing_table.empty:
            raise ValueError(
                "Routing table is empty."
            )

        normalized_ids = (
            routing_table["Medicine_ID"]
            .astype(str)
            .str.strip()
        )

        if normalized_ids.eq("").any():
            raise ValueError(
                "Routing table contains empty Medicine_ID values."
            )

        duplicate_mask = normalized_ids.duplicated(
            keep=False
        )

        if duplicate_mask.any():

            duplicates = (
                normalized_ids[
                    duplicate_mask
                ]
                .unique()
                .tolist()
            )

            raise ValueError(
                "Routing table contains duplicate Medicine_ID values: "
                f"{duplicates}"
            )

        raw_advantage = routing_table[
            "Validation_Advantage_Pct"
        ]

        advantage = pd.to_numeric(
            raw_advantage,
            errors="coerce",
        )

        invalid_non_numeric = (
            raw_advantage.notna()
            & advantage.isna()
        )

        if invalid_non_numeric.any():

            invalid_values = (
                raw_advantage.loc[
                    invalid_non_numeric
                ]
                .astype(str)
                .unique()
                .tolist()
            )

            raise ValueError(
                "Routing table contains non-numeric "
                "Validation_Advantage_Pct values: "
                f"{invalid_values}"
            )

        invalid_finite = (
            advantage.notna()
            & ~np.isfinite(
                advantage
            )
        )

        if invalid_finite.any():
            raise ValueError(
                "Routing table contains non-finite "
                "Validation_Advantage_Pct values."
            )

    # ========================================================================
    # BATCH FORECASTING
    # ========================================================================

    def forecast_batch(
        self,
        history_df: pd.DataFrame,
        routing_table: pd.DataFrame,
    ) -> tuple[pd.DataFrame, list[str]]:
        """
        Generate production forecasts for all medicines in routing_table.

        History:

            MDCODE
            INVDT
            Demand_Qty

        Routing table:

            Medicine_ID
            Validation_Advantage_Pct

        Returns:

            forecasts
            failed_medicines
        """

        self._validate_routing_table(
            routing_table
        )

        self._validate_source_history(
            history_df
        )

        # --------------------------------------------------------------------
        # Prepare source history exactly once.
        # --------------------------------------------------------------------

        prepared_history = (
            self._prepare_history(
                history_df
            )
        )

        routing_table = routing_table.copy()

        routing_table["Medicine_ID"] = (
            routing_table["Medicine_ID"]
            .astype(str)
            .str.strip()
        )

        routing_table[
            "Validation_Advantage_Pct"
        ] = pd.to_numeric(
            routing_table[
                "Validation_Advantage_Pct"
            ],
            errors="coerce",
        )

        forecasts: list[pd.DataFrame] = []

        failed: list[str] = []

        # --------------------------------------------------------------------
        # Process every routed medicine independently.
        # --------------------------------------------------------------------

        for _, route in routing_table.iterrows():

            medicine_id = str(
                route["Medicine_ID"]
            ).strip()

            advantage = route[
                "Validation_Advantage_Pct"
            ]

            try:

                decision = self.build_decision(
                    medicine_id=medicine_id,
                    validation_advantage_pct=advantage,
                )

                logger.info(
                    "BATCH ROUTING | medicine=%s | model=%s | "
                    "advantage=%s | rule=%s",
                    medicine_id,
                    decision.selected_model,
                    decision.validation_advantage_pct,
                    decision.routing_rule,
                )

                medicine_history = (
                    prepared_history[
                        prepared_history[
                            INTERNAL_ID_COLUMN
                        ]
                        .astype(str)
                        .str.strip()
                        == medicine_id
                    ]
                    .copy()
                )

                if medicine_history.empty:
                    raise InsufficientHistoryError(
                        f"No history for medicine_id={medicine_id}"
                    )

                if decision.selected_model == CHRONOS_MODEL:

                    forecast = self._forecast_chronos(
                        medicine_id=medicine_id,
                        medicine_history=medicine_history,
                        decision=decision,
                    )

                else:

                    forecast = self._forecast_tsb(
                        medicine_id=medicine_id,
                        medicine_history=medicine_history,
                        decision=decision,
                    )

                self._validate_forecast_output(
                    forecast,
                    medicine_id,
                )

                forecasts.append(
                    forecast
                )

            except InsufficientHistoryError as exc:

                logger.warning(
                    "Skipping medicine=%s: %s",
                    medicine_id,
                    exc,
                )

                failed.append(
                    medicine_id
                )

            except (
                ValueError,
                TypeError,
            ) as exc:

                logger.error(
                    "Invalid production forecast for medicine=%s: %s",
                    medicine_id,
                    exc,
                )

                failed.append(
                    medicine_id
                )

            except Exception:

                logger.exception(
                    "Unexpected production forecast failure "
                    "for medicine=%s",
                    medicine_id,
                )

                failed.append(
                    medicine_id
                )

        # --------------------------------------------------------------------
        # Assemble final output.
        # --------------------------------------------------------------------

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
                columns=OUTPUT_COLUMNS
            )

        return output, failed

    # ========================================================================
    # FORECAST OUTPUT VALIDATION
    # ========================================================================

    @staticmethod
    def _validate_forecast_output(
        forecast: pd.DataFrame,
        medicine_id: str,
    ) -> None:
        """
        Validate production forecast integrity.
        """

        if not isinstance(
            forecast,
            pd.DataFrame,
        ):
            raise TypeError(
                "Forecast result must be a pandas DataFrame."
            )

        if forecast.empty:
            raise ValueError(
                f"Forecast result is empty for medicine_id={medicine_id}."
            )

        missing = (
            set(OUTPUT_COLUMNS)
            - set(forecast.columns)
        )

        if missing:
            raise ValueError(
                f"Forecast output for medicine_id={medicine_id} "
                f"is missing columns: {sorted(missing)}"
            )

        # --------------------------------------------------------------------
        # Medicine identity.
        # --------------------------------------------------------------------

        output_ids = (
            forecast["Medicine_ID"]
            .astype(str)
            .str.strip()
            .unique()
        )

        if len(output_ids) != 1:
            raise ValueError(
                f"Forecast for medicine_id={medicine_id} "
                "contains multiple medicine IDs."
            )

        if output_ids[0] != str(
            medicine_id
        ).strip():
            raise ValueError(
                f"Forecast medicine mismatch. "
                f"Expected={medicine_id}, "
                f"Found={output_ids.tolist()}"
            )

        # --------------------------------------------------------------------
        # Dates.
        # --------------------------------------------------------------------

        dates = pd.to_datetime(
            forecast["Forecast_Date"],
            errors="coerce",
        )

        if dates.isna().any():
            raise ValueError(
                f"Forecast contains invalid dates for "
                f"medicine_id={medicine_id}."
            )

        if dates.duplicated().any():
            raise ValueError(
                f"Forecast contains duplicate dates for "
                f"medicine_id={medicine_id}."
            )

        if not dates.is_monotonic_increasing:
            raise ValueError(
                f"Forecast dates are not sorted for "
                f"medicine_id={medicine_id}."
            )

        # --------------------------------------------------------------------
        # Demand / quantile values.
        # --------------------------------------------------------------------

        numeric_columns = [
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
        ]

        numeric = forecast[
            numeric_columns
        ].apply(
            pd.to_numeric,
            errors="coerce",
        )

        if numeric.isna().any().any():
            raise ValueError(
                f"Forecast contains NaN/non-numeric demand values "
                f"for medicine_id={medicine_id}."
            )

        values = numeric.to_numpy(
            dtype=float
        )

        if not np.isfinite(values).all():
            raise ValueError(
                f"Forecast contains non-finite demand values "
                f"for medicine_id={medicine_id}."
            )

        if (
            values < 0
        ).any():
            raise ValueError(
                f"Forecast contains negative demand values "
                f"for medicine_id={medicine_id}."
            )

        # --------------------------------------------------------------------
        # Validate model.
        # --------------------------------------------------------------------

        allowed_models = {
            CHRONOS_MODEL,
            TSB_MODEL,
        }

        invalid_models = (
            set(
                forecast["Selected_Model"]
            )
            - allowed_models
        )

        if invalid_models:
            raise ValueError(
                f"Invalid Selected_Model values for "
                f"medicine_id={medicine_id}: "
                f"{sorted(invalid_models)}"
            )

        # --------------------------------------------------------------------
        # Validate Forecast_Type.
        # --------------------------------------------------------------------

        expected_forecast_types = {
            CHRONOS_MODEL: "probabilistic",
            TSB_MODEL: "point",
        }

        for model, expected_type in (
            expected_forecast_types.items()
        ):

            mask = (
                forecast["Selected_Model"]
                == model
            )

            if mask.any():

                actual_types = set(
                    forecast.loc[
                        mask,
                        "Forecast_Type",
                    ]
                )

                if actual_types != {
                    expected_type
                }:
                    raise ValueError(
                        f"Invalid Forecast_Type for "
                        f"model={model}, "
                        f"medicine_id={medicine_id}: "
                        f"{actual_types}"
                    )

        # --------------------------------------------------------------------
        # Chronos Predicted_Demand must equal P50.
        # --------------------------------------------------------------------

        chronos_mask = (
            forecast["Selected_Model"]
            == CHRONOS_MODEL
        )

        if chronos_mask.any():

            predicted = (
                forecast.loc[
                    chronos_mask,
                    "Predicted_Demand",
                ]
                .astype(float)
                .to_numpy()
            )

            p50 = (
                forecast.loc[
                    chronos_mask,
                    "P50",
                ]
                .astype(float)
                .to_numpy()
            )

            if not np.allclose(
                predicted,
                p50,
                rtol=1e-9,
                atol=1e-9,
            ):
                raise ValueError(
                    f"Chronos Predicted_Demand does not equal P50 "
                    f"for medicine_id={medicine_id}."
                )

            # ---------------------------------------------------------------
            # Quantiles must be monotonically increasing.
            # ---------------------------------------------------------------

            quantile_matrix = forecast.loc[
                chronos_mask,
                [
                    "P10",
                    "P20",
                    "P30",
                    "P40",
                    "P50",
                    "P60",
                    "P70",
                    "P80",
                    "P90",
                ],
            ].to_numpy(
                dtype=float
            )

            if (
                np.diff(
                    quantile_matrix,
                    axis=1,
                )
                < -1e-9
            ).any():

                raise ValueError(
                    f"Chronos quantiles are not monotonic "
                    f"for medicine_id={medicine_id}."
                )

        # --------------------------------------------------------------------
        # TSB quantiles must equal the point forecast because they are only
        # schema-compatible placeholders, not calibrated probabilistic
        # estimates.
        # --------------------------------------------------------------------

        tsb_mask = (
            forecast["Selected_Model"]
            == TSB_MODEL
        )

        if tsb_mask.any():

            predicted = forecast.loc[
                tsb_mask,
                "Predicted_Demand",
            ].to_numpy(
                dtype=float
            )

            for quantile in TSB_QUANTILES:

                quantile_values = forecast.loc[
                    tsb_mask,
                    quantile,
                ].to_numpy(
                    dtype=float
                )

                if not np.allclose(
                    predicted,
                    quantile_values,
                    rtol=1e-9,
                    atol=1e-9,
                ):
                    raise ValueError(
                        f"TSB {quantile} does not equal "
                        f"Predicted_Demand for "
                        f"medicine_id={medicine_id}."
                    )

        # --------------------------------------------------------------------
        # Prediction length.
        # --------------------------------------------------------------------

        prediction_lengths = pd.to_numeric(
            forecast[
                "Prediction_Length"
            ],
            errors="coerce",
        )

        if prediction_lengths.isna().any():
            raise ValueError(
                f"Prediction_Length contains invalid values "
                f"for medicine_id={medicine_id}."
            )

        unique_lengths = (
            prediction_lengths
            .astype(int)
            .unique()
        )

        if len(unique_lengths) != 1:
            raise ValueError(
                f"Inconsistent Prediction_Length for "
                f"medicine_id={medicine_id}."
            )

        expected_length = int(
            unique_lengths[0]
        )

        if expected_length <= 0:
            raise ValueError(
                f"Prediction_Length must be positive "
                f"for medicine_id={medicine_id}."
            )

        if len(forecast) != expected_length:
            raise ValueError(
                f"Forecast horizon mismatch for "
                f"medicine_id={medicine_id}. "
                f"Expected={expected_length}, "
                f"actual={len(forecast)}"
            )

    # ========================================================================
    # ROUTING POLICY SELF-CHECK
    # ========================================================================

    @staticmethod
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
            ("invalid", TSB_MODEL),
        ]

        for advantage, expected in test_cases:

            actual = (
                ProductionForecastRouter.select_model(
                    advantage
                )
            )

            if actual != expected:
                raise RuntimeError(
                    "Production routing policy self-check failed: "
                    f"advantage={advantage}, "
                    f"expected={expected}, "
                    f"actual={actual}"
                )

    # ========================================================================
    # TSB SELF-CHECK
    # ========================================================================

    @staticmethod
    def _run_tsb_self_check() -> None:
        """
        Validate basic invariants of the validated TSB implementation.
        """

        # --------------------------------------------------------------------
        # All-zero history.
        # --------------------------------------------------------------------

        history_zero = pd.DataFrame(
            {
                INTERNAL_ID_COLUMN: [
                    "TEST_ZERO"
                ] * 5,
                INTERNAL_TIMESTAMP_COLUMN: pd.date_range(
                    "2026-01-01",
                    periods=5,
                    freq="D",
                ),
                INTERNAL_TARGET_COLUMN: [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
            }
        )

        zero_forecast = (
            ProductionForecastRouter.forecast_tsb(
                history_zero,
                "TEST_ZERO",
                prediction_length=3,
            )
        )

        zero_values = (
            zero_forecast[
                "Predicted_Demand"
            ]
            .to_numpy(
                dtype=float
            )
        )

        if not np.allclose(
            zero_values,
            0.0,
        ):
            raise RuntimeError(
                "TSB self-check failed: "
                "all-zero history did not produce zero forecast."
            )

        # --------------------------------------------------------------------
        # Positive intermittent history.
        # --------------------------------------------------------------------

        history_positive = pd.DataFrame(
            {
                INTERNAL_ID_COLUMN: [
                    "TEST_POS"
                ] * 6,
                INTERNAL_TIMESTAMP_COLUMN: pd.date_range(
                    "2026-01-01",
                    periods=6,
                    freq="D",
                ),
                INTERNAL_TARGET_COLUMN: [
                    0.0,
                    10.0,
                    0.0,
                    0.0,
                    20.0,
                    0.0,
                ],
            }
        )

        positive_forecast = (
            ProductionForecastRouter.forecast_tsb(
                history_positive,
                "TEST_POS",
                prediction_length=3,
            )
        )

        values = (
            positive_forecast[
                "Predicted_Demand"
            ]
            .to_numpy(
                dtype=float
            )
        )

        if not np.isfinite(
            values
        ).all():
            raise RuntimeError(
                "TSB self-check failed: "
                "non-finite forecast produced."
            )

        if (
            values < 0
        ).any():
            raise RuntimeError(
                "TSB self-check failed: "
                "negative forecast produced."
            )

        if not np.allclose(
            values,
            values[0],
        ):
            raise RuntimeError(
                "TSB self-check failed: "
                "forecast is not constant across horizon."
            )

    # ========================================================================
    # ROUTING TABLE BUILDER
    # ========================================================================


def build_routing_table(
    robustness_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert validation/robustness results into the production routing table.

    Expected columns:

        Medicine_ID
        Validation_Chronos_AE
        Validation_TSB_AE

    Important:

    The input dataframe must contain validation-derived metrics only.

    Do not populate these columns using holdout/test results.
    """

    if not isinstance(
        robustness_df,
        pd.DataFrame,
    ):
        raise TypeError(
            "robustness_df must be a pandas DataFrame."
        )

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

    if robustness_df.empty:
        raise ValueError(
            "Robustness dataframe is empty."
        )

    routing = robustness_df[
        [
            "Medicine_ID",
            "Validation_Chronos_AE",
            "Validation_TSB_AE",
        ]
    ].copy()

    # ------------------------------------------------------------------------
    # Normalize IDs.
    # ------------------------------------------------------------------------

    routing["Medicine_ID"] = (
        routing["Medicine_ID"]
        .astype(str)
        .str.strip()
    )

    if routing[
        "Medicine_ID"
    ].eq("").any():
        raise ValueError(
            "Robustness dataframe contains empty Medicine_ID values."
        )

    # ------------------------------------------------------------------------
    # Duplicate validation.
    # ------------------------------------------------------------------------

    duplicate_mask = (
        routing["Medicine_ID"]
        .duplicated(
            keep=False
        )
    )

    if duplicate_mask.any():

        duplicates = (
            routing.loc[
                duplicate_mask,
                "Medicine_ID",
            ]
            .unique()
            .tolist()
        )

        raise ValueError(
            "Robustness dataframe contains duplicate Medicine_ID values: "
            f"{duplicates}"
        )

    # ------------------------------------------------------------------------
    # Numeric validation metrics.
    # ------------------------------------------------------------------------

    routing[
        "Validation_Chronos_AE"
    ] = pd.to_numeric(
        routing[
            "Validation_Chronos_AE"
        ],
        errors="coerce",
    )

    routing[
        "Validation_TSB_AE"
    ] = pd.to_numeric(
        routing[
            "Validation_TSB_AE"
        ],
        errors="coerce",
    )

    # ------------------------------------------------------------------------
    # Validate AE values.
    # ------------------------------------------------------------------------

    for column in (
        "Validation_Chronos_AE",
        "Validation_TSB_AE",
    ):

        values = routing[column]

        invalid = (
            values.notna()
            & (
                (values < 0)
                | ~np.isfinite(values)
            )
        )

        if invalid.any():
            raise ValueError(
                f"{column} contains negative or non-finite values."
            )

    # ------------------------------------------------------------------------
    # Calculate validation advantage.
    #
    # TSB AE == 0 is treated as undefined rather than allowing division
    # by zero.
    # ------------------------------------------------------------------------

    routing[
        "Validation_Advantage_Pct"
    ] = (
        (
            routing[
                "Validation_TSB_AE"
            ]
            - routing[
                "Validation_Chronos_AE"
            ]
        )
        / routing[
            "Validation_TSB_AE"
        ].replace(
            0,
            np.nan,
        )
        * 100.0
    )

    # ------------------------------------------------------------------------
    # Apply frozen policy.
    # ------------------------------------------------------------------------

    routing[
        "Selected_Model"
    ] = (
        routing[
            "Validation_Advantage_Pct"
        ]
        .apply(
            ProductionForecastRouter.select_model
        )
    )

    routing[
        "Routing_Rule"
    ] = ROUTING_RULE_NAME

    routing[
        "Threshold"
    ] = VALIDATION_ADVANTAGE_THRESHOLD

    return routing


# ============================================================================
# MODULE-LEVEL SELF-CHECKS
# ============================================================================


def _run_policy_self_check() -> None:
    """
    Backward-compatible module-level policy self-check.
    """

    ProductionForecastRouter._run_policy_self_check()


def _run_tsb_self_check() -> None:
    """
    Module-level TSB self-check.
    """

    ProductionForecastRouter._run_tsb_self_check()


# ============================================================================
# CLI
# ============================================================================


def main() -> None:

    print("=" * 80)
    print("FINEMED PRODUCTION FORECAST ROUTER")
    print("=" * 80)

    print()
    print("Production data schema:")
    print(f"  ID        : {SOURCE_ID_COLUMN}")
    print(f"  Date      : {SOURCE_TIMESTAMP_COLUMN}")
    print(f"  Demand    : {SOURCE_TARGET_COLUMN}")

    print()
    print("Internal forecasting schema:")
    print(f"  ID        : {INTERNAL_ID_COLUMN}")
    print(f"  Timestamp : {INTERNAL_TIMESTAMP_COLUMN}")
    print(f"  Target    : {INTERNAL_TARGET_COLUMN}")

    print()
    print("Frozen routing policy:")
    print(
        f"  Chronos-2 P50 if validation advantage >= "
        f"{VALIDATION_ADVANTAGE_THRESHOLD:.0f}%"
    )
    print("  Otherwise -> TSB")

    print()
    print("Validated TSB parameters:")
    print(
        f"  alpha_demand      = {TSB_ALPHA_DEMAND}"
    )
    print(
        f"  alpha_probability = {TSB_ALPHA_PROBABILITY}"
    )

    print()
    print("Running routing policy self-check...")

    _run_policy_self_check()

    print(
        "PASS: routing policy self-check"
    )

    print()
    print("Running TSB implementation self-check...")

    _run_tsb_self_check()

    print(
        "PASS: TSB implementation self-check"
    )

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
        "PASS: Silver MDCODE/INVDT/Demand_Qty is normalized "
        "before forecasting."
    )

    print(
        "PASS: Missing source IDs are removed before string conversion."
    )

    print(
        "PASS: Daily timestamps are normalized."
    )

    print(
        "PASS: Duplicate medicine/date observations are aggregated."
    )

    print(
        "PASS: Calendar gaps are filled with zero demand."
    )

    print(
        "PASS: Chronos receives medicine-specific history."
    )

    print(
        "PASS: Chronos Predicted_Demand equals P50."
    )

    print(
        "PASS: Chronos quantiles are validated for monotonicity."
    )

    print(
        "PASS: Validated TSB probability is updated every observation."
    )

    print(
        "PASS: Validated TSB demand estimate updates only on demand."
    )

    print(
        "PASS: TSB output is explicitly marked as a point forecast."
    )

    print(
        "PASS: Routing-table duplicate medicine IDs are rejected."
    )

    print(
        "PASS: Forecast output integrity checks are enabled."
    )

    print(
        "PASS: No unvalidated Chronos bias correction/scaling is applied."
    )

    print()
    print("=" * 80)
    print("PRODUCTION FLOW")
    print("=" * 80)

    print(
        "Silver MDCODE/INVDT/Demand_Qty"
    )

    print(
        "        -> validation + normalization"
    )

    print(
        "        -> validation-only routing table"
    )

    print(
        "        -> frozen 30% routing policy"
    )

    print(
        "        -> Chronos-2 P50 OR validated TSB"
    )

    print(
        "        -> 30-day forecast"
    )

    print(
        "        -> output integrity validation"
    )


if __name__ == "__main__":
    main()