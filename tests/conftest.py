import os
import pytest

@pytest.fixture(autouse=True)
def setup_test_environment(monkeypatch):
    """Provide safe test-only credentials for tests if not already set."""
    if "CLIENT_API_KEY" not in os.environ:
        monkeypatch.setenv("CLIENT_API_KEY", "test-client-key")
    if "ADMIN_TOKEN" not in os.environ:
        monkeypatch.setenv("ADMIN_TOKEN", "test-admin-token")
