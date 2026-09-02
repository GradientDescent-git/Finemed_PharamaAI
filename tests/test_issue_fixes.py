"""
Unit tests covering P0 and P1 issue fixes:
1. DB connection fast-fail check
2. PostDeploymentEvaluator normalization & diagnostic logging
3. NotificationService stage failure wiring
"""

from __future__ import annotations

import logging
import pytest
import pandas as pd
from unittest.mock import patch, MagicMock

from finemed_ai.database.db_connection import test_connection as check_db_connection
from finemed_ai.demand_forecasting.post_evaluator import PostDeploymentEvaluator, PostEvaluationResult
from finemed_ai.automation.notification import NotificationService


def test_db_test_connection_fast_fail():
    """
    Verify check_db_connection() fails fast with RuntimeError when DB connection fails.
    """
    with patch("finemed_ai.database.db_connection.get_engine") as mock_get_engine:
        mock_engine = MagicMock()
        mock_engine.connect.side_effect = Exception("psycopg2.OperationalError: password authentication failed for user 'postgres'")
        mock_get_engine.return_value = mock_engine
        
        with pytest.raises(RuntimeError) as exc_info:
            check_db_connection()
        assert "Database authentication/connection check failed" in str(exc_info.value)


def test_post_evaluator_normalization_and_match(caplog):
    """
    Verify PostDeploymentEvaluator handles string zfill Medicine_IDs and date dtypes cleanly.
    """
    evaluator = PostDeploymentEvaluator()
    
    preds_df = pd.DataFrame({
        "Medicine_ID": ["1", "2"],
        "Forecast_Date": ["2026-06-01", "2026-06-02"],
        "Predicted_Demand": [100.0, 200.0]
    })
    
    actuals_df = pd.DataFrame({
        "Medicine_ID": ["0001", "0002"],
        "Daily_Demand_Date": ["2026-06-01 00:00:00", "2026-06-02 00:00:00"],
        "Daily_Demand": [95.0, 205.0]
    })
    
    result = evaluator.evaluate(
        run_id="test_run_001",
        predictions_df=preds_df,
        actuals_df=actuals_df
    )
    
    assert isinstance(result, PostEvaluationResult)
    assert result.total_medicines == 2
    assert result.status == "PASSED"
    assert result.wape > 0.0


def test_post_evaluator_diagnostic_logging(caplog):
    """
    Verify PostDeploymentEvaluator emits diagnostic warning when merged dataset is empty.
    """
    evaluator = PostDeploymentEvaluator()
    
    preds_df = pd.DataFrame({
        "Medicine_ID": ["0001"],
        "Forecast_Date": ["2026-06-01"],
        "Predicted_Demand": [100.0]
    })
    
    actuals_df = pd.DataFrame({
        "Medicine_ID": ["0005"],
        "Daily_Demand_Date": ["2026-07-01"],
        "Daily_Demand": [95.0]
    })
    
    with caplog.at_level(logging.WARNING):
        result = evaluator.evaluate(
            run_id="non_matching_run",
            predictions_df=preds_df,
            actuals_df=actuals_df
        )
    
    assert result.total_medicines == 0
    assert result.status == "DEGRADED"
    assert "No matching records found in PostDeploymentEvaluator merge" in caplog.text


def test_notification_service_notify_failure(caplog):
    """
    Verify NotificationService logs failure messages clearly.
    """
    service = NotificationService()
    with caplog.at_level(logging.ERROR):
        service.notify_failure("Pipeline Stage 1 (ETL) failed: DB timeout")
    assert "[FAILED] Pipeline Stage 1 (ETL) failed: DB timeout" in caplog.text
