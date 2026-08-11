from datetime import datetime
import pandas as pd
import pytest

from finemed_ai.automation.alert_engine import AlertEngine, AlertSeverity, AlertType


def test_alert_engine_detects_stockout_risk(tmp_path):
    dates = pd.date_range("2025-06-01", periods=30, freq="D")
    forecast_df = pd.DataFrame({
        "Medicine_ID": ["101"] * 30,
        "Forecast_Date": dates,
        "Predicted_Demand": [10.0] * 30,  # 300 units needed
        "P10": [7.0] * 30,
        "P90": [13.0] * 30,
    })

    inventory_df = pd.DataFrame({
        "Medicine_ID": ["101"],
        "Stock_On_Hand": [50.0],  # only 50 units in stock -> shortage of 250
    })

    engine = AlertEngine(tmp_path)
    store = engine.scan_forecasts(forecast_df, inventory_df=inventory_df)

    assert store.total_alerts >= 1
    stockout_alert = next((a for a in store.alerts if a.alert_type == AlertType.STOCKOUT_RISK), None)
    assert stockout_alert is not None
    assert stockout_alert.severity == AlertSeverity.CRITICAL
    assert stockout_alert.metric_value == 250.0


def test_alert_engine_detects_demand_spike(tmp_path):
    dates = pd.date_range("2025-06-01", periods=30, freq="D")
    forecast_df = pd.DataFrame({
        "Medicine_ID": ["202"] * 30,
        "Forecast_Date": dates,
        "Predicted_Demand": [50.0] * 30,  # 50 units/day forecast
        "P10": [40.0] * 30,
        "P90": [60.0] * 30,
    })

    historical_df = pd.DataFrame({
        "Medicine_ID": ["202"] * 90,
        "Forecast_Date": pd.date_range("2025-03-01", periods=90, freq="D"),
        "Demand_Qty": [10.0] * 90,  # historical avg is 10 units/day -> 400% surge
    })

    engine = AlertEngine(tmp_path)
    store = engine.scan_forecasts(forecast_df, historical_demand_df=historical_df)

    spike_alert = next((a for a in store.alerts if a.alert_type == AlertType.DEMAND_SPIKE), None)
    assert spike_alert is not None
    assert spike_alert.severity == AlertSeverity.WARNING
    assert spike_alert.metric_value == 400.0
