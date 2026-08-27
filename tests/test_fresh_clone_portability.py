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
    the repository, store, query service, and conversational orchestrator handle missing
    artifacts honestly and gracefully without throwing FileNotFoundError or crashing.
    """
    empty_forecast_path = tmp_path / "nonexistent" / "latest.parquet"
    empty_routing_path = tmp_path / "nonexistent" / "routing.parquet"
    empty_medicine_path = tmp_path / "nonexistent" / "medicine.parquet"

    repo = ForecastRepository(
        forecast_path=empty_forecast_path,
        routing_path=empty_routing_path,
        medicine_path=empty_medicine_path,
    )

    assert repo.is_available() is False
    forecasts, routing, medicines = repo.load_all()

    # 1. Assert missing artifacts return empty DataFrames without crashing
    assert forecasts.empty
    assert routing.empty
    assert medicines.empty

    # 2. Assert QueryService handles empty repository gracefully
    query_service = ForecastQueryService(repository=repo)
    resolved = query_service.resolve_medicine("0001")
    assert resolved["resolved"] is False

    forecast_res = query_service.get_forecast("0001")
    assert forecast_res["found"] is False

    top_res = query_service.get_top_demand(5)
    assert isinstance(top_res, list)
    assert len(top_res) == 0

    # 3. Assert Orchestrator functions safely without throwing exceptions
    orchestrator = ForecastConversationOrchestrator(
        query_service=query_service,
    )

    response = orchestrator.ask("What is the forecast for medicine 0001?")
    assert "data" in response
    assert response["data"]["found"] is False
