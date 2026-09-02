import pytest
from pathlib import Path
from fastapi.testclient import TestClient
import pandas as pd

from finemed_ai.api.main import app
from finemed_ai.demand_forecasting.store import ForecastStore
from finemed_ai.forecast_intelligence.repository import ForecastRepository
from finemed_ai.demand_forecasting.drift_detector import DemandDriftDetector
from finemed_ai.demand_forecasting.post_evaluator import PostDeploymentEvaluator


@pytest.fixture
def client():
    return TestClient(app)


def test_missing_forecast_artifact_honest_failure(tmp_path):
    """Assert ForecastStore handles non-existent file cleanly without generating fake data."""
    missing_file = tmp_path / "non_existent_latest.parquet"
    store = ForecastStore(missing_file)
    assert store.is_available() is False
    assert store.is_empty() is True


def test_missing_repository_artifacts_honest_failure(tmp_path):
    """Assert ForecastRepository handles non-existent files cleanly."""
    repo = ForecastRepository(
        forecast_path=tmp_path / "missing.parquet",
        routing_path=tmp_path / "missing.parquet",
        medicine_path=tmp_path / "missing.parquet",
    )
    assert repo.is_available() is False
    assert repo.load_forecasts().empty
    assert repo.load_routing().empty
    assert repo.load_medicines().empty


def test_invalid_api_key_returns_401(client):
    """Assert requests with invalid API key return 401 Unauthorized."""
    res = client.get("/forecast/top", headers={"X-API-Key": "WRONG_KEY"})
    assert res.status_code == 401


def test_unconfigured_api_key_returns_503(client, monkeypatch):
    """Assert requests when CLIENT_API_KEY is absent return 503 Service Unavailable."""
    monkeypatch.delenv("CLIENT_API_KEY", raising=False)
    res = client.get("/forecast/top", headers={"X-API-Key": "test-key"})
    assert res.status_code == 503


def test_drift_detector_handles_empty_inputs():
    """Assert DemandDriftDetector gracefully handles empty DataFrames."""
    detector = DemandDriftDetector()
    report = detector.detect_drift(pd.DataFrame(), pd.DataFrame())
    assert report.status == "WARNING"


def test_post_evaluator_handles_empty_inputs():
    """Assert PostDeploymentEvaluator gracefully handles empty inputs."""
    evaluator = PostDeploymentEvaluator()
    result = evaluator.evaluate("run_test", pd.DataFrame(), pd.DataFrame())
    assert result.status == "DEGRADED"


def test_request_id_middleware_propagation(client):
    """Assert X-Request-ID header is propagated in response."""
    custom_id = "test-req-id-12345"
    res = client.get("/health", headers={"X-Request-ID": custom_id})
    assert res.status_code == 200
    assert res.headers.get("X-Request-ID") == custom_id


def test_all_protected_routes_require_auth(client):
    """
    Iterate all OpenAPI paths and verify non-public routes return 401 or 503 without credentials.
    """
    public_paths = {"/", "/health", "/ready", "/version", "/docs", "/openapi.json", "/redoc"}
    openapi_spec = app.openapi()
    paths = openapi_spec.get("paths", {})

    for path, methods in paths.items():
        if path in public_paths or path.startswith("/static"):
            continue

        for method in methods.keys():
            if method.lower() not in ("get", "post", "put", "delete"):
                continue

            test_path = path.replace("{medicine_id}", "0001").replace("{run_id}", "current").replace("{path:path}", "index.html")
            
            if method.lower() == "get":
                res = client.get(test_path)
            elif method.lower() == "post":
                res = client.post(test_path, json={})
            
            assert res.status_code in (401, 403, 422, 503), f"Route {method.upper()} {path} allowed unauthenticated access (status {res.status_code})"


def test_cors_configuration_restricts_unlisted_origins(client):
    """Assert CORS middleware blocks or omits origin headers for unlisted origins."""
    res = client.get("/health", headers={"Origin": "https://malicious-domain.com"})
    assert res.headers.get("access-control-allow-origin") != "https://malicious-domain.com"


def test_chat_rate_limiting_exceeded(client, monkeypatch):
    """Assert /chat returns 429 Too Many Requests when request threshold is exceeded."""
    api_key = "rate-limit-test-key"
    monkeypatch.setenv("CLIENT_API_KEY", api_key)
    headers = {"X-API-Key": api_key}
    responses = []
    for _ in range(65):
        res = client.post("/chat", headers=headers, json={"question": "What is the forecast?"})
        responses.append(res.status_code)
    
    assert 429 in responses
