from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest

from finemed_ai.demand_forecasting.store import ForecastStore
from finemed_ai.llm.tools import ForecastTools


@pytest.fixture
def store(tmp_path):
    """
    Create a valid 30-day probabilistic forecast fixture.

    The forecast has an increasing demand trend and quantiles
    that remain non-negative, finite, and monotonically ordered.
    """

    prediction_length = 30
    rows = []

    generated_at = datetime.now()

    for i in range(prediction_length):

        predicted_demand = 10.0 + (i * 0.5)

        rows.append(
            {
                "Medicine_ID": "42",

                "Forecast_Date": (
                    pd.Timestamp("2025-06-01")
                    + pd.Timedelta(days=i)
                ),

                "Predicted_Demand": predicted_demand,

                "P10": max(
                    0.0,
                    predicted_demand - 5.0,
                ),

                "P20": max(
                    0.0,
                    predicted_demand - 4.0,
                ),

                "P30": max(
                    0.0,
                    predicted_demand - 3.0,
                ),

                "P40": max(
                    0.0,
                    predicted_demand - 2.0,
                ),

                "P50": predicted_demand,

                "P60": (
                    predicted_demand + 2.0
                ),

                "P70": (
                    predicted_demand + 3.0
                ),

                "P80": (
                    predicted_demand + 4.0
                ),

                "P90": (
                    predicted_demand + 5.0
                ),

                "Context_Length_Used": 730,

                "Prediction_Length": (
                    prediction_length
                ),

                "Model_ID": (
                    "amazon/chronos-2"
                ),

                "Selected_Model": (
                    "chronos-2-P50"
                ),

                "Generated_At": (
                    generated_at
                ),
            }
        )

    path = tmp_path / "latest.parquet"

    pd.DataFrame(
        rows
    ).to_parquet(
        path,
        index=False,
    )

    return ForecastStore(path)


def test_get_forecast_tool(store):
    """
    The get_forecast tool should return the complete
    30-day forecast for a known medicine.
    """

    tools = ForecastTools(store)

    result = tools.execute(
        "get_forecast",
        {
            "medicine_id": "42",
        },
    )

    assert (
        result["medicine_id"]
        == "42"
    )

    assert len(
        result["days"]
    ) == 30

    first_day = (
        result["days"][0]
    )

    assert "p10" in first_day
    assert "p50" in first_day
    assert "p90" in first_day


def test_get_trend_tool_detects_increase(
    store,
):
    """
    The fixture contains a steadily increasing forecast,
    so the trend tool should classify it as increasing.
    """

    tools = ForecastTools(store)

    result = tools.execute(
        "get_trend",
        {
            "medicine_id": "42",
        },
    )

    assert (
        result["trend"]
        == "increasing"
    )


def test_get_summary_tool(store):
    """
    The summary tool should return a valid summary
    for a known medicine.
    """

    tools = ForecastTools(store)

    result = tools.execute(
        "get_summary",
        {
            "medicine_id": "42",
        },
    )

    assert (
        result["medicine_id"]
        == "42"
    )

    assert (
        result["avg_daily_demand"]
        > 0
    )

    assert (
        result[
            "total_predicted_demand"
        ]
        > 0
    )

    assert (
        result["prediction_length"]
        == 30
    )


def test_unknown_medicine_returns_error_not_exception(
    store,
):
    """
    Unknown medicines should produce a structured error
    instead of leaking an exception.
    """

    tools = ForecastTools(store)

    result = tools.execute(
        "get_forecast",
        {
            "medicine_id": (
                "does-not-exist"
            ),
        },
    )

    assert "error" in result


def test_unknown_tool_name_returns_error(
    store,
):
    """
    Unknown tool names should produce a structured error.
    """

    tools = ForecastTools(store)

    result = tools.execute(
        "not_a_real_tool",
        {
            "medicine_id": "42",
        },
    )

    assert "error" in result


def test_store_reload_picks_up_new_data(
    store,
):
    """
    The store should detect a changed forecast file and,
    after reload, expose newly added medicines.

    Medicine 99 receives a complete valid forecast rather
    than a single row incorrectly declaring a 30-day horizon.
    """

    assert (
        store.list_medicine_ids()
        == ["42"]
    )

    new_rows = pd.read_parquet(
        store.latest_path,
    )

    # Copy the complete 30-day forecast.
    extra = new_rows.copy()

    # Convert the complete forecast into a second medicine.
    extra["Medicine_ID"] = "99"

    # Prediction_Length remains 30 because medicine 99
    # now also has exactly 30 forecast rows.
    extra["Prediction_Length"] = 30

    pd.concat(
        [
            new_rows,
            extra,
        ],
        ignore_index=True,
    ).to_parquet(
        store.latest_path,
        index=False,
    )

    assert store.is_stale()

    store.reload()

    medicine_ids = (
        store.list_medicine_ids()
    )

    assert "42" in medicine_ids
    assert "99" in medicine_ids