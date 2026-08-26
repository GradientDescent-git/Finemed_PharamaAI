from __future__ import annotations

import pandas as pd
import pytest

from finemed_ai.forecast_intelligence.conversation_orchestrator import (
    ForecastConversationOrchestrator,
)
from finemed_ai.forecast_intelligence.query_service import ForecastQueryService
from finemed_ai.forecast_intelligence.repository import ForecastRepository


def test_fresh_clone_portability_end_to_end(tmp_path):
    """
    Assert that on a fresh clone (where data/ directory or .env file does not exist),
    the repository, store, query service, and conversational orchestrator load
    cleanly without crashing or throwing FileNotFoundError/RuntimeError.
    """
    empty_forecast_path = tmp_path / "nonexistent" / "latest.parquet"
    empty_routing_path = tmp_path / "nonexistent" / "routing.parquet"
    empty_medicine_path = tmp_path / "nonexistent" / "medicine.parquet"

    repo = ForecastRepository(
        forecast_path=empty_forecast_path,
        routing_path=empty_routing_path,
        medicine_path=empty_medicine_path,
    )

    forecasts, routing, medicines = repo.load_all()

    # 1. Assert fallbacks return non-empty DataFrames
    assert not forecasts.empty, "Forecast fallback DataFrame must not be empty."
    assert not routing.empty, "Routing fallback DataFrame must not be empty."
    assert not medicines.empty, "Medicines fallback DataFrame must not be empty."

    # 2. Assert schema contract includes required column names
    assert "Medicine_ID" in forecasts.columns
    assert "Predicted_Demand" in forecasts.columns
    assert "Selected_Model" in forecasts.columns

    assert "Medicine_ID" in routing.columns
    assert "Selected_Model" in routing.columns

    assert "MDCODE" in medicines.columns
    assert "Product_Display_Name" in medicines.columns

    # 3. Assert QueryService works end-to-end with fallback repository
    query_service = ForecastQueryService(repository=repo)
    resolved = query_service.resolve_medicine("0001")
    assert resolved["resolved"] is True
    assert resolved["medicine_name"] == "OTACARE EAR DROPS 5ML"

    forecast_res = query_service.get_forecast("0001")
    assert forecast_res["found"] is True
    assert forecast_res["total_predicted_demand"] > 0


    top_res = query_service.get_top_demand(5)
    assert isinstance(top_res, list)
    assert len(top_res) > 0


    # 4. Assert Orchestrator functions end-to-end without crashing
    orchestrator = ForecastConversationOrchestrator(
        query_service=query_service,
    )

    response = orchestrator.ask("What is the forecast for medicine 0001?")
    assert response["action"] == "forecast"
    assert response["data"]["found"] is True
    assert "OTACARE EAR DROPS 5ML" in response["answer"]
