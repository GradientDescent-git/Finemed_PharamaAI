from __future__ import annotations

import io
import zipfile
from datetime import datetime

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import finemed_ai.api.main as api_main
from finemed_ai.demand_forecasting.store import ForecastStore


@pytest.fixture
def api_client(tmp_path, monkeypatch):
    rows = [
        {
            "Medicine_ID": "0001",
            "Forecast_Date": pd.Timestamp("2025-06-01"),
            "Predicted_Demand": 100.0,
            "P10": 50.0,
            "P50": 100.0,
            "P90": 150.0,
            "Context_Length_Used": 365,
            "Prediction_Length": 1,
            "Selected_Model": "TSB",
            "Generated_At": datetime.now(),
        }
    ]

    output_dir = tmp_path / "forecasts"
    output_dir.mkdir()
    forecast_path = output_dir / "latest.parquet"
    pd.DataFrame(rows).to_parquet(forecast_path, index=False)

    api_main._store = ForecastStore(forecast_path)
    api_main._orchestrator = None

    monkeypatch.setenv("ADMIN_TOKEN", "test-admin-secret")
    api_main.ADMIN_TOKEN = "test-admin-secret"
    monkeypatch.setenv("CLIENT_API_KEY", "test-client-key")
    api_main.CLIENT_API_KEY = "test-client-key"

    return TestClient(
        api_main.app,
        headers={
            "x-api-key": "test-client-key",
            "x-admin-token": "test-admin-secret",
        },
    )


def test_ready_endpoint(api_client):
    resp = api_client.get("/ready")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ready"
    assert data["ready"] is True
    assert data["store_ready"] is True


def test_version_endpoint(api_client):
    resp = api_client.get("/version")
    assert resp.status_code == 200
    data = resp.json()
    assert data["name"] == "Finemed PharmaAI"
    assert data["version"] == "1.0.0"


def test_pipeline_status_endpoints(api_client):
    resp = api_client.get("/admin/pipeline-status")
    assert resp.status_code == 200

    resp_latest = api_client.get("/pipeline/latest")
    assert resp_latest.status_code == 200

    resp_id = api_client.get("/pipeline/status/nonexistent-id")
    assert resp_id.status_code == 200
    assert resp_id.json()["status"] == "not_found"


def test_operations_summary_endpoint(api_client):
    resp = api_client.get("/operations/summary")
    assert resp.status_code == 200
    data = resp.json()
    assert data["api_status"] == "ok"
    assert data["forecast_store_loaded"] is True
    assert data["total_medicines"] == 1
    assert "freshness" in data


def test_pipeline_upload_invalid_zip(api_client):
    invalid_file = io.BytesIO(b"not a zip file content")
    resp = api_client.post(
        "/pipeline/upload?month=2025-07",
        files={"file": ("test.zip", invalid_file, "application/zip")},
    )
    assert resp.status_code == 400
    assert "valid ZIP" in resp.json()["detail"]


def test_pipeline_upload_missing_required_files(api_client):
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zf:
        zf.writestr("INVOICE.DAT", b"sample invoice data")
    zip_buffer.seek(0)

    resp = api_client.post(
        "/pipeline/upload?month=2025-07",
        files={"file": ("test.zip", zip_buffer, "application/zip")},
    )
    assert resp.status_code == 400
    assert "missing required files" in resp.json()["detail"].lower()


def test_admin_auth_required(tmp_path, monkeypatch):
    api_main.ADMIN_TOKEN = "secret-token"
    monkeypatch.setenv("ADMIN_TOKEN", "secret-token")
    client = TestClient(api_main.app)

    resp = client.get("/operations/summary")
    assert resp.status_code == 401
