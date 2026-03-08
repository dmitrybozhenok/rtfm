"""Fixtures for end-to-end Playwright tests.

Assumes the RTFM server is already running at http://localhost:8000
with Redis and Ollama available. Start with:
    docker compose up redis -d
    uvicorn rtfm.api.routes:app
"""

import pytest

BASE_URL = "http://localhost:8000"


_server_available = None


def _check_server():
    global _server_available
    if _server_available is None:
        import httpx
        try:
            r = httpx.get(f"{BASE_URL}/health", timeout=5)
            _server_available = r.status_code == 200
        except Exception:
            _server_available = False
    return _server_available


@pytest.fixture(autouse=True)
def skip_if_no_server():
    """Skip e2e tests if the server is not running."""
    if not _check_server():
        pytest.skip(f"RTFM server not running at {BASE_URL}")


@pytest.fixture(scope="session")
def browser_context_args(browser_context_args):
    """Configure browser context with the base URL."""
    return {**browser_context_args, "base_url": BASE_URL}
