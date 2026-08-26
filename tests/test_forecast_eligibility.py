from __future__ import annotations

import pandas as pd
import pytest

from finemed_ai.demand_forecasting.forecast_eligibility import (
    EligibilityPolicy,
    EligibilityStatus,
    build_eligibility_table,
    classify_gap_days,
)


def test_gap_90_is_active():

    assert (
        classify_gap_days(90)
        == EligibilityStatus.ACTIVE
    )


def test_gap_91_is_stale():

    assert (
        classify_gap_days(91)
        == EligibilityStatus.STALE
    )


def test_gap_365_is_stale():

    assert (
        classify_gap_days(365)
        == EligibilityStatus.STALE
    )


def test_gap_366_is_dormant():

    assert (
        classify_gap_days(366)
        == EligibilityStatus.DORMANT
    )


def test_negative_gap_fails():

    with pytest.raises(
        ValueError,
    ):

        classify_gap_days(-1)


def test_invalid_policy_fails():

    with pytest.raises(
        ValueError,
    ):

        EligibilityPolicy(
            active_max_gap_days=365,
            stale_max_gap_days=90,
        )


def test_build_eligibility_table():

    history = pd.DataFrame(
        {
            "Medicine_ID": [
                "0001",
                "0002",
                "0003",
            ],
            "timestamp": pd.to_datetime(
                [
                    "2026-05-30",
                    "2026-02-24",
                    "2025-05-29",
                ]
            ),
        }
    )

    result = build_eligibility_table(
        history_df=history,
        medicine_id_column="Medicine_ID",
        timestamp_column="timestamp",
    )

    result = result.set_index(
        "Medicine_ID"
    )

    assert (
        result.loc[
            "0001",
            "Gap_Days",
        ]
        == 0
    )

    assert (
        result.loc[
            "0001",
            "Eligibility_Status",
        ]
        == "ACTIVE"
    )

    assert (
        result.loc[
            "0002",
            "Eligibility_Status",
        ]
        == "STALE"
    )

    assert (
        result.loc[
            "0003",
            "Eligibility_Status",
        ]
        == "DORMANT"
    )


def test_missing_column_fails():

    history = pd.DataFrame(
        {
            "Medicine_ID": [
                "0001",
            ],
        }
    )

    with pytest.raises(
        ValueError,
    ):

        build_eligibility_table(
            history_df=history,
            medicine_id_column="Medicine_ID",
            timestamp_column="timestamp",
        )


def test_null_medicine_id_fails():

    history = pd.DataFrame(
        {
            "Medicine_ID": [None],
            "timestamp": [
                "2026-05-30",
            ],
        }
    )

    with pytest.raises(
        ValueError,
    ):

        build_eligibility_table(
            history_df=history,
            medicine_id_column="Medicine_ID",
            timestamp_column="timestamp",
        )


def test_null_timestamp_fails():

    history = pd.DataFrame(
        {
            "Medicine_ID": ["0001"],
            "timestamp": [None],
        }
    )

    with pytest.raises(
        ValueError,
    ):

        build_eligibility_table(
            history_df=history,
            medicine_id_column="Medicine_ID",
            timestamp_column="timestamp",
        )


def test_output_is_deterministic():

    history = pd.DataFrame(
        {
            "Medicine_ID": [
                "0002",
                "0001",
                "0003",
            ],
            "timestamp": [
                "2026-01-01",
                "2026-05-30",
                "2025-01-01",
            ],
        }
    )

    first = build_eligibility_table(
        history_df=history,
        medicine_id_column="Medicine_ID",
        timestamp_column="timestamp",
    )

    second = build_eligibility_table(
        history_df=history,
        medicine_id_column="Medicine_ID",
        timestamp_column="timestamp",
    )

    pd.testing.assert_frame_equal(
        first,
        second,
    )

    assert (
        first["Medicine_ID"]
        .tolist()
        == [
            "0001",
            "0002",
            "0003",
        ]
    )