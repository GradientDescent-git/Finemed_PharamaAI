from __future__ import annotations

import math
from datetime import date, datetime
from typing import ClassVar, List, Literal

from pydantic import (
    BaseModel,
    Field,
    field_validator,
    model_validator,
)


# ============================================================================
# Constants
# ============================================================================


TREND_INCREASING = "increasing"
TREND_DECREASING = "decreasing"
TREND_STABLE = "stable"

VALID_TRENDS = {
    TREND_INCREASING,
    TREND_DECREASING,
    TREND_STABLE,
}


# ============================================================================
# Helpers
# ============================================================================


def _validate_finite(
    value: float,
    field_name: str,
) -> float:
    """
    Ensure a numeric value is finite.

    NaN and +/- infinity are never allowed in the public forecast schema.
    """

    try:
        numeric_value = float(value)

    except (
        TypeError,
        ValueError,
    ) as exc:

        raise ValueError(
            f"{field_name} must be numeric."
        ) from exc

    if not math.isfinite(
        numeric_value
    ):

        raise ValueError(
            f"{field_name} must be a finite number."
        )

    return numeric_value


def _validate_nonnegative_finite(
    value: float,
    field_name: str,
) -> float:
    """
    Ensure a numeric value is finite and >= 0.
    """

    numeric_value = _validate_finite(
        value,
        field_name,
    )

    if numeric_value < 0:

        raise ValueError(
            f"{field_name} cannot be negative."
        )

    return numeric_value


def _validate_nonnegative_integer(
    value: int,
    field_name: str,
) -> int:
    """
    Ensure an integer value is >= 0.

    Boolean values are explicitly rejected because bool is a subclass
    of int in Python.
    """

    if isinstance(
        value,
        bool,
    ):

        raise ValueError(
            f"{field_name} must be an integer, "
            "not a boolean."
        )

    try:

        integer_value = int(
            value
        )

    except (
        TypeError,
        ValueError,
    ) as exc:

        raise ValueError(
            f"{field_name} must be an integer."
        ) from exc

    if integer_value < 0:

        raise ValueError(
            f"{field_name} cannot be negative."
        )

    return integer_value


def _validate_positive_integer(
    value: int,
    field_name: str,
) -> int:
    """
    Ensure an integer value is > 0.
    """

    integer_value = (
        _validate_nonnegative_integer(
            value,
            field_name,
        )
    )

    if integer_value <= 0:

        raise ValueError(
            f"{field_name} must be positive."
        )

    return integer_value


def _normalize_nonempty_string(
    value: str,
    field_name: str,
) -> str:
    """
    Normalize and validate a required string.
    """

    if value is None:

        raise ValueError(
            f"{field_name} cannot be None."
        )

    normalized = str(
        value
    ).strip()

    if not normalized:

        raise ValueError(
            f"{field_name} cannot be empty."
        )

    return normalized


def _validate_datetime(
    value: datetime,
    field_name: str,
) -> datetime:
    """
    Validate datetime values.

    Naive datetimes are allowed because the existing production
    artifact/API contract may already use them. The schema does not
    silently alter timezone semantics.
    """

    if not isinstance(
        value,
        datetime,
    ):

        raise ValueError(
            f"{field_name} must be a datetime."
        )

    return value


# ============================================================================
# Quantile forecast
# ============================================================================


class QuantileForecast(BaseModel):
    """
    Probabilistic forecast quantiles for one forecast day.

    Required ordering:

        P10 <= P20 <= P30 <= P40 <= P50
            <= P60 <= P70 <= P80 <= P90

    All quantiles must be finite and non-negative.

    Point-only models may use the same value for every quantile.
    """

    p10: float
    p20: float
    p30: float
    p40: float
    p50: float
    p60: float
    p70: float
    p80: float
    p90: float

    @field_validator(
        "p10",
        "p20",
        "p30",
        "p40",
        "p50",
        "p60",
        "p70",
        "p80",
        "p90",
    )
    @classmethod
    def validate_quantile_value(
        cls,
        value: float,
        info,
    ) -> float:

        return _validate_nonnegative_finite(
            value,
            info.field_name,
        )

    @model_validator(
        mode="after"
    )
    def validate_quantile_order(
        self,
    ) -> "QuantileForecast":

        values = [
            self.p10,
            self.p20,
            self.p30,
            self.p40,
            self.p50,
            self.p60,
            self.p70,
            self.p80,
            self.p90,
        ]

        for index in range(
            len(values) - 1
        ):

            if (
                values[index]
                > values[index + 1]
            ):

                raise ValueError(
                    "Forecast quantiles must satisfy "
                    "P10 <= P20 <= P30 <= P40 <= P50 "
                    "<= P60 <= P70 <= P80 <= P90."
                )

        return self


# ============================================================================
# Daily forecast result
# ============================================================================


class ForecastDayResult(BaseModel):
    """
    Forecast for one medicine on one future calendar day.

    predicted_demand represents the operational point forecast.

    Quantiles represent the uncertainty distribution.
    """

    forecast_date: date

    predicted_demand: float

    quantiles: QuantileForecast

    @field_validator(
        "predicted_demand"
    )
    @classmethod
    def validate_predicted_demand(
        cls,
        value: float,
    ) -> float:

        return _validate_nonnegative_finite(
            value,
            "predicted_demand",
        )


# ============================================================================
# Medicine-level forecast result
# ============================================================================


class MedicineForecastResult(BaseModel):
    """
    Complete forecast for one medicine over the prediction horizon.

    Production invariants:

    - medicine_id is non-empty.
    - model_id is non-empty.
    - generated_at is a valid datetime.
    - context_length_used >= 0.
    - prediction_length > 0.
    - days count must equal prediction_length.
    - forecast dates must be unique.
    """

    TREND_THRESHOLD_PCT: ClassVar[
        float
    ] = 5.0

    medicine_id: str

    generated_at: datetime

    context_length_used: int

    prediction_length: int

    model_id: str

    days: List[
        ForecastDayResult
    ] = Field(
        default_factory=list
    )

    @field_validator(
        "medicine_id"
    )
    @classmethod
    def validate_medicine_id(
        cls,
        value: str,
    ) -> str:

        return _normalize_nonempty_string(
            value,
            "medicine_id",
        )

    @field_validator(
        "generated_at"
    )
    @classmethod
    def validate_generated_at(
        cls,
        value: datetime,
    ) -> datetime:

        return _validate_datetime(
            value,
            "generated_at",
        )

    @field_validator(
        "context_length_used"
    )
    @classmethod
    def validate_context_length(
        cls,
        value: int,
    ) -> int:
        """
        Context length represents effective history consumed by
        the selected forecasting route.

        Zero is valid for forecasting routes that legitimately do not
        consume historical context.
        """

        return _validate_nonnegative_integer(
            value,
            "context_length_used",
        )

    @field_validator(
        "prediction_length"
    )
    @classmethod
    def validate_prediction_length(
        cls,
        value: int,
    ) -> int:

        return _validate_positive_integer(
            value,
            "prediction_length",
        )

    @field_validator(
        "model_id"
    )
    @classmethod
    def validate_model_id(
        cls,
        value: str,
    ) -> str:

        return _normalize_nonempty_string(
            value,
            "model_id",
        )

    @model_validator(
        mode="after"
    )
    def validate_days(
        self,
    ) -> "MedicineForecastResult":

        if not self.days:

            raise ValueError(
                "days cannot be empty when "
                "prediction_length is positive."
            )

        if (
            len(self.days)
            != self.prediction_length
        ):

            raise ValueError(
                "prediction_length does not match "
                "the number of forecast days. "
                f"Expected {self.prediction_length}, "
                f"got {len(self.days)}."
            )

        dates = [
            day.forecast_date
            for day in self.days
        ]

        if (
            len(dates)
            != len(set(dates))
        ):

            raise ValueError(
                "Forecast contains duplicate "
                "forecast_date values."
            )

        return self

    # ==================================================================
    # Summary calculation
    # ==================================================================

    def to_summary(
        self,
    ) -> "ForecastSummary":
        """
        Convert the complete forecast into a medicine-level summary.

        Trend methodology
        -----------------

        The forecast horizon is sorted chronologically and divided into
        two contiguous periods.

        For even horizons:

            [first half] [second half]

        For odd horizons:

            The first period receives floor(n / 2) observations and
            the second period receives the remaining observations.

        Example:

            n = 5

            first period  = days 1-2
            second period = days 3-5

        Trend percentage change is:

            (second_avg - first_avg)
            ------------------------ * 100
                    first_avg

        When first_avg is zero:

            zero -> zero = 0%
            zero -> positive = +100%

        The +100% value is intentionally a bounded operational
        representation rather than mathematical infinity. This keeps
        ranking, API serialization, and downstream consumers finite.
        """

        if not self.days:

            raise ValueError(
                "Cannot summarize an empty forecast."
            )

        ordered_days = sorted(
            self.days,
            key=lambda day: (
                day.forecast_date
            ),
        )

        values = [
            _validate_nonnegative_finite(
                day.predicted_demand,
                "predicted_demand",
            )
            for day in ordered_days
        ]

        actual_length = len(
            values
        )

        if actual_length <= 0:

            raise ValueError(
                "Cannot summarize an empty forecast."
            )

        total_predicted_demand = float(
            math.fsum(
                values
            )
        )

        avg_daily_demand = float(
            total_predicted_demand
            / actual_length
        )

        # --------------------------------------------------------------
        # Horizon split.
        # --------------------------------------------------------------

        if actual_length == 1:

            first_half = values

            second_half = values

        else:

            split_index = max(
                1,
                actual_length // 2,
            )

            first_half = values[
                :split_index
            ]

            second_half = values[
                split_index:
            ]

            if not second_half:

                second_half = first_half

        first_half_avg = float(
            math.fsum(
                first_half
            )
            / len(first_half)
        )

        second_half_avg = float(
            math.fsum(
                second_half
            )
            / len(second_half)
        )

        # --------------------------------------------------------------
        # Trend percentage.
        # --------------------------------------------------------------

        if first_half_avg > 0:

            trend_pct_change = float(
                (
                    (
                        second_half_avg
                        - first_half_avg
                    )
                    / first_half_avg
                )
                * 100.0
            )

        elif second_half_avg > 0:

            # A percentage change from zero is mathematically undefined.
            #
            # We deliberately expose a bounded +100% operational signal
            # instead of infinity so all API consumers remain safe.
            trend_pct_change = 100.0

        else:

            trend_pct_change = 0.0

        trend_pct_change = (
            _validate_finite(
                trend_pct_change,
                "trend_pct_change",
            )
        )

        # --------------------------------------------------------------
        # Trend classification.
        # --------------------------------------------------------------

        threshold = (
            self.TREND_THRESHOLD_PCT
        )

        if (
            trend_pct_change
            > threshold
        ):

            trend = TREND_INCREASING

        elif (
            trend_pct_change
            < -threshold
        ):

            trend = TREND_DECREASING

        else:

            trend = TREND_STABLE

        return ForecastSummary(
            medicine_id=self.medicine_id,
            generated_at=self.generated_at,
            model_id=self.model_id,
            prediction_length=actual_length,
            total_predicted_demand=(
                total_predicted_demand
            ),
            avg_daily_demand=(
                avg_daily_demand
            ),
            first_half_avg=(
                first_half_avg
            ),
            second_half_avg=(
                second_half_avg
            ),
            trend=trend,
            trend_pct_change=(
                trend_pct_change
            ),
        )


# ============================================================================
# Medicine-level summary
# ============================================================================


class ForecastSummary(BaseModel):
    """
    Medicine-level forecast summary.

    Consumed by:

        - FastAPI
        - LLM tools
        - ranking/comparison tools
        - dashboards
        - downstream application services

    The summary contains only finite numeric values.
    """

    medicine_id: str

    generated_at: datetime

    model_id: str

    prediction_length: int

    total_predicted_demand: float

    avg_daily_demand: float

    first_half_avg: float

    second_half_avg: float

    trend: Literal[
        "increasing",
        "decreasing",
        "stable",
    ]

    trend_pct_change: float

    @field_validator(
        "medicine_id"
    )
    @classmethod
    def validate_medicine_id(
        cls,
        value: str,
    ) -> str:

        return _normalize_nonempty_string(
            value,
            "medicine_id",
        )

    @field_validator(
        "generated_at"
    )
    @classmethod
    def validate_generated_at(
        cls,
        value: datetime,
    ) -> datetime:

        return _validate_datetime(
            value,
            "generated_at",
        )

    @field_validator(
        "model_id"
    )
    @classmethod
    def validate_model_id(
        cls,
        value: str,
    ) -> str:

        return _normalize_nonempty_string(
            value,
            "model_id",
        )

    @field_validator(
        "prediction_length"
    )
    @classmethod
    def validate_prediction_length(
        cls,
        value: int,
    ) -> int:

        return _validate_positive_integer(
            value,
            "prediction_length",
        )

    @field_validator(
        "total_predicted_demand",
        "avg_daily_demand",
        "first_half_avg",
        "second_half_avg",
    )
    @classmethod
    def validate_nonnegative_summary_numbers(
        cls,
        value: float,
        info,
    ) -> float:

        return _validate_nonnegative_finite(
            value,
            info.field_name,
        )

    @field_validator(
        "trend_pct_change"
    )
    @classmethod
    def validate_trend_pct_change(
        cls,
        value: float,
    ) -> float:

        return _validate_finite(
            value,
            "trend_pct_change",
        )


# ============================================================================
# Batch forecast run result
# ============================================================================


class BatchForecastRunResult(BaseModel):
    """
    Metadata for one production forecast batch.

    Production invariants:

        medicines_succeeded + medicines_failed
            == medicines_requested

        completed_at >= started_at

        failed_medicine_ids:
            - contain no empty values
            - contain no duplicates
            - cannot exceed medicines_failed

        published=True indicates that the artifact was accepted by the
        publication layer. Partial forecast runs may still be published
        when the publication gate explicitly allows them, so this schema
        does not impose medicines_failed == 0.
    """

    run_id: str

    started_at: datetime

    completed_at: datetime

    medicines_requested: int

    medicines_succeeded: int

    medicines_failed: int

    failed_medicine_ids: List[
        str
    ] = Field(
        default_factory=list
    )

    output_path: str

    published: bool

    publish_note: str = ""

    @field_validator(
        "run_id",
        "output_path",
    )
    @classmethod
    def validate_required_strings(
        cls,
        value: str,
        info,
    ) -> str:

        return _normalize_nonempty_string(
            value,
            info.field_name,
        )

    @field_validator(
        "started_at",
        "completed_at",
    )
    @classmethod
    def validate_batch_datetimes(
        cls,
        value: datetime,
        info,
    ) -> datetime:

        return _validate_datetime(
            value,
            info.field_name,
        )

    @field_validator(
        "medicines_requested",
        "medicines_succeeded",
        "medicines_failed",
    )
    @classmethod
    def validate_counts(
        cls,
        value: int,
        info,
    ) -> int:

        return _validate_nonnegative_integer(
            value,
            info.field_name,
        )

    @field_validator(
        "failed_medicine_ids"
    )
    @classmethod
    def validate_failed_medicine_ids(
        cls,
        values: List[str],
    ) -> List[str]:

        normalized = [
            _normalize_nonempty_string(
                value,
                "failed_medicine_id",
            )
            for value in values
        ]

        if (
            len(normalized)
            != len(set(normalized))
        ):

            raise ValueError(
                "failed_medicine_ids "
                "cannot contain duplicates."
            )

        return normalized

    @field_validator(
        "publish_note"
    )
    @classmethod
    def normalize_publish_note(
        cls,
        value: str,
    ) -> str:

        if value is None:

            return ""

        return str(
            value
        ).strip()

    @model_validator(
        mode="after"
    )
    def validate_batch_consistency(
        self,
    ) -> "BatchForecastRunResult":

        if (
            self.medicines_succeeded
            + self.medicines_failed
            != self.medicines_requested
        ):

            raise ValueError(
                "medicines_succeeded + medicines_failed "
                "must equal medicines_requested."
            )

        if (
            len(
                self.failed_medicine_ids
            )
            > self.medicines_failed
        ):

            raise ValueError(
                "failed_medicine_ids contains more IDs "
                "than medicines_failed."
            )

        if (
            self.completed_at
            < self.started_at
        ):

            raise ValueError(
                "completed_at cannot be earlier "
                "than started_at."
            )

        if (
            self.published
            and not self.publish_note
        ):

            raise ValueError(
                "publish_note cannot be empty "
                "when published is True."
            )

        return self