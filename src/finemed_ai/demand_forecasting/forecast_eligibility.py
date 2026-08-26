from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import pandas as pd


# ============================================================================
# FORECAST ELIGIBILITY POLICY
# ============================================================================
#
# ACTIVE:
#     Last observed demand is within 90 days of the production data cutoff.
#
# STALE:
#     Last observed demand is between 91 and 365 days before the cutoff.
#
# DORMANT:
#     Last observed demand is more than 365 days before the cutoff.
#
# Boundary policy:
#
#     gap <= 90       -> ACTIVE
#     90 < gap <= 365 -> STALE
#     gap > 365       -> DORMANT
#
# ============================================================================


ACTIVE_MAX_GAP_DAYS = 90

STALE_MAX_GAP_DAYS = 365


class EligibilityStatus(str, Enum):
    """
    Production forecasting eligibility state.
    """

    ACTIVE = "ACTIVE"

    STALE = "STALE"

    DORMANT = "DORMANT"


@dataclass(frozen=True)
class EligibilityPolicy:
    """
    Immutable policy controlling medicine forecast eligibility.
    """

    active_max_gap_days: int = ACTIVE_MAX_GAP_DAYS

    stale_max_gap_days: int = STALE_MAX_GAP_DAYS

    def __post_init__(self) -> None:

        if not isinstance(
            self.active_max_gap_days,
            int,
        ):
            raise TypeError(
                "active_max_gap_days must be an integer."
            )

        if not isinstance(
            self.stale_max_gap_days,
            int,
        ):
            raise TypeError(
                "stale_max_gap_days must be an integer."
            )

        if self.active_max_gap_days < 0:
            raise ValueError(
                "active_max_gap_days must be >= 0."
            )

        if self.stale_max_gap_days <= (
            self.active_max_gap_days
        ):
            raise ValueError(
                "stale_max_gap_days must be greater than "
                "active_max_gap_days."
            )


DEFAULT_ELIGIBILITY_POLICY = EligibilityPolicy()


def classify_gap_days(
    gap_days: int,
    policy: EligibilityPolicy = DEFAULT_ELIGIBILITY_POLICY,
) -> EligibilityStatus:
    """
    Classify a medicine based on the number of days since
    its last observed demand.
    """

    if not isinstance(
        gap_days,
        int,
    ):
        raise TypeError(
            "gap_days must be an integer."
        )

    if gap_days < 0:
        raise ValueError(
            "gap_days must not be negative."
        )

    if gap_days <= policy.active_max_gap_days:

        return EligibilityStatus.ACTIVE

    if gap_days <= policy.stale_max_gap_days:

        return EligibilityStatus.STALE

    return EligibilityStatus.DORMANT


def build_eligibility_table(
    history_df: pd.DataFrame,
    medicine_id_column: str,
    timestamp_column: str,
    policy: EligibilityPolicy = DEFAULT_ELIGIBILITY_POLICY,
) -> pd.DataFrame:
    """
    Build a deterministic production forecast eligibility table.

    The production data cutoff is defined as the maximum timestamp
    present anywhere in the validated forecasting dataset.

    Returns:

        Medicine_ID
        Last_Historical_Date
        Data_Cutoff_Date
        Gap_Days
        Eligibility_Status
    """

    if not isinstance(
        history_df,
        pd.DataFrame,
    ):
        raise TypeError(
            "history_df must be a pandas DataFrame."
        )

    if history_df.empty:
        raise ValueError(
            "history_df must not be empty."
        )

    required_columns = {
        medicine_id_column,
        timestamp_column,
    }

    missing_columns = (
        required_columns
        - set(history_df.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing required eligibility columns: "
            f"{sorted(missing_columns)}"
        )

    working = history_df[
        [
            medicine_id_column,
            timestamp_column,
        ]
    ].copy()

    if working[
        medicine_id_column
    ].isna().any():

        raise ValueError(
            "Medicine IDs must not contain null values."
        )

    if working[
        timestamp_column
    ].isna().any():

        raise ValueError(
            "Historical timestamps must not contain null values."
        )

    working[
        timestamp_column
    ] = pd.to_datetime(
        working[
            timestamp_column
        ],
        errors="raise",
    )

    cutoff_date = working[
        timestamp_column
    ].max()

    if pd.isna(
        cutoff_date
    ):
        raise ValueError(
            "Unable to determine production data cutoff date."
        )

    eligibility = (
        working
        .groupby(
            medicine_id_column,
            as_index=False,
        )
        .agg(
            Last_Historical_Date=(
                timestamp_column,
                "max",
            ),
        )
    )

    eligibility[
        "Data_Cutoff_Date"
    ] = cutoff_date

    eligibility[
        "Gap_Days"
    ] = (
        eligibility[
            "Data_Cutoff_Date"
        ]
        - eligibility[
            "Last_Historical_Date"
        ]
    ).dt.days

    if (
        eligibility[
            "Gap_Days"
        ]
        < 0
    ).any():

        raise RuntimeError(
            "Eligibility table contains negative gap days."
        )

    eligibility[
        "Eligibility_Status"
    ] = eligibility[
        "Gap_Days"
    ].apply(
        lambda gap: classify_gap_days(
            int(gap),
            policy,
        ).value
    )

    eligibility = eligibility.rename(
        columns={
            medicine_id_column:
                "Medicine_ID",
        }
    )

    eligibility = eligibility[
        [
            "Medicine_ID",
            "Last_Historical_Date",
            "Data_Cutoff_Date",
            "Gap_Days",
            "Eligibility_Status",
        ]
    ]

    eligibility = eligibility.sort_values(
        "Medicine_ID"
    ).reset_index(
        drop=True
    )

    return eligibility