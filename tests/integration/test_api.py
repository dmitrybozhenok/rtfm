"""API endpoint contract tests.

These test HTTP endpoints using FastAPI's TestClient.
They require Redis but NOT Ollama (LLM calls are mocked).
"""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from rtfm.api.routes import app

client = TestClient(app)

MOCK_ASK_RESPONSE = {
    "answer": "The capital of France is Paris.",
    "sources": [
        {"file": "test.md", "section": "Geography", "score": 0.12}
    ],
    "cached": False,
    "latency_ms": 42.5,
    "tokens_used": 150,
}


@pytest.mark.integration
def test_ingest_path_success(tmp_path):
    """POST /ingest/path with a valid path returns 200 with files/chunks count."""
    doc = tmp_path / "sample.md"
    doc.write_text("# Hello\n\nThis is a test document for ingestion.", encoding="utf-8")

    response = client.post("/ingest/path", data={"path": str(tmp_path)})
    assert response.status_code == 200
    body = response.json()
    assert "files" in body
    assert "chunks" in body
    assert body["status"] == "ok"
    assert body["files"] >= 1
    assert body["chunks"] >= 1


@pytest.mark.integration
def test_ingest_path_not_found():
    """POST /ingest/path with a non-existent path returns 404."""
    response = client.post(
        "/ingest/path",
        data={"path": "/tmp/rtfm_nonexistent_path_abc123"},
    )
    assert response.status_code == 404
    body = response.json()
    assert "error" in body


@pytest.mark.integration
def test_metrics_endpoint():
    """GET /metrics returns 200 with all expected metric fields."""
    response = client.get("/metrics")
    assert response.status_code == 200
    body = response.json()
    expected_fields = {
        "cache_hits",
        "cache_misses",
        "hit_rate_pct",
        "total_queries",
        "avg_latency_ms",
        "total_tokens",
        "estimated_cost_usd",
        "estimated_savings_usd",
    }
    assert expected_fields.issubset(body.keys()), (
        f"Missing metric fields: {expected_fields - body.keys()}"
    )


@pytest.mark.integration
def test_cache_flush_endpoint():
    """POST /cache/flush returns 200 with success message."""
    response = client.post("/cache/flush")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "message" in body


@pytest.mark.integration
def test_session_clear_endpoint():
    """POST /session/{session_id}/clear returns 200."""
    response = client.post("/session/test-123/clear")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "test-123" in body["message"]


@pytest.mark.integration
def test_ask_endpoint_structure():
    """POST /ask returns response with all expected keys (LLM mocked)."""
    with patch("rtfm.api.routes.ask", return_value=MOCK_ASK_RESPONSE):
        response = client.post(
            "/ask",
            data={"question": "What is the capital of France?"},
        )
    assert response.status_code == 200
    body = response.json()
    expected_keys = {"answer", "sources", "cached", "latency_ms", "tokens_used"}
    assert expected_keys.issubset(body.keys()), (
        f"Missing keys: {expected_keys - body.keys()}"
    )
    assert isinstance(body["answer"], str)
    assert isinstance(body["sources"], list)
    assert isinstance(body["cached"], bool)
    assert isinstance(body["latency_ms"], (int, float))
    assert isinstance(body["tokens_used"], int)
