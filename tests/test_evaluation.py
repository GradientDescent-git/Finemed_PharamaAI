from datetime import datetime, timedelta
import pandas as pd
import pytest

from finemed_ai.demand_forecasting.evaluation import (
    ForecastEvaluator,
    compute_metrics,
)


def test_compute_metrics_exact_match():
    actuals = [10.0, 20.0, 30.0]
    preds = [10.0, 20.0, 30.0]
    p10s = [8.0, 15.0, 25.0]
    p90s = [12.0, 25.0, 35.0]

    m = compute_metrics(actuals, preds, p10s, p90s)
    assert m["wape_pct"] == 0.0
    assert m["mae"] == 0.0
    assert m["coverage_pct"] == 100.0


def test_compute_metrics_wape_calculation():
    actuals = [100.0, 200.0]
    preds = [110.0, 180.0]  # abs errors: 10, 20 => sum = 30. total actual = 300 => WAPE = 10%
    m = compute_metrics(actuals, preds)
    assert m["wape_pct"] == 10.0
    assert m["mae"] == 15.0


def test_forecast_evaluator_run(tmp_path):
    dates = pd.date_range("2025-06-01", periods=5, freq="D")
    actuals_df = pd.DataFrame({
        "Medicine_ID": ["1"] * 5,
        "Forecast_Date": dates,
        "Demand_Qty": [10, 20, 30, 40, 50],
    })

    forecast_df = pd.DataFrame({
        "Medicine_ID": ["1"] * 5,
        "Forecast_Date": dates,
        "Predicted_Demand": [12, 18, 33, 38, 52],
        "P10": [5, 10, 20, 30, 40],
        "P90": [15, 25, 35, 45, 55],
    })

    evaluator = ForecastEvaluator(tmp_path)
    result = evaluator.evaluate(actuals_df, forecast_df)

    assert result.total_medicines_evaluated == 1
    assert result.overall_wape_pct > 0
    assert len(result.medicines) == 1
    assert result.medicines[0].medicine_id in ["1", "0001"]

    # verify persistence
    latest = evaluator.load_latest_evaluation()
    assert latest is not None
    assert latest.total_medicines_evaluated == 1
