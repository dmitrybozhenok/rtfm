"""Unit tests for UX improvement features: list-sources, search CLI commands,
and /health, /sources API endpoints.

Uses Typer CliRunner for CLI tests and FastAPI TestClient for API tests.
All external services (Redis, Ollama, Memory Server) are mocked.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from rtfm.api.routes import app as fastapi_app
from rtfm.cli.app import app as cli_app
from rtfm.retrieval.search import SearchResult

runner = CliRunner()
client = TestClient(fastapi_app)


# ── CLI: list-sources ────────────────────────────────────────


class TestListSourcesCLI:
    @patch("rtfm.redis_client.get_redis")
    def test_list_sources_shows_files_and_web(self, mock_get_redis):
        """Sources are grouped by type (file vs web) with chunk counts."""
        mock_redis = MagicMock()
        mock_get_redis.return_value = mock_redis

        # Simulate two scan iterations then cursor=0 to stop
        mock_redis.scan.side_effect = [
            (1, ["doc:abc1", "doc:abc2"]),
            (0, ["doc:abc3"]),
        ]
        mock_redis.hmget.side_effect = [
            ["guide.pdf", "Installation"],
            ["guide.pdf", "Usage"],
            ["https://example.com/docs", "API Reference"],
        ]

        result = runner.invoke(cli_app, ["list-sources"])

        assert result.exit_code == 0
        assert "guide.pdf" in result.output
        assert "2 chunks" in result.output
        assert "https://example.com/docs" in result.output
        assert "1 chunks" in result.output
        assert "Files:" in result.output
        assert "Web:" in result.output

    @patch("rtfm.redis_client.get_redis")
    def test_list_sources_empty(self, mock_get_redis):
        """When no docs are ingested, show a 'No documents' message."""
        mock_redis = MagicMock()
        mock_get_redis.return_value = mock_redis
        mock_redis.scan.return_value = (0, [])

        result = runner.invoke(cli_app, ["list-sources"])

        assert result.exit_code == 0
        assert "No documents ingested" in result.output

    @patch("rtfm.redis_client.get_redis")
    def test_list_sources_sections_displayed(self, mock_get_redis):
        """Sections from chunks are displayed for each source."""
        mock_redis = MagicMock()
        mock_get_redis.return_value = mock_redis

        mock_redis.scan.return_value = (0, ["doc:a1", "doc:a2"])
        mock_redis.hmget.side_effect = [
            ["readme.md", "Getting Started"],
            ["readme.md", "Configuration"],
        ]

        result = runner.invoke(cli_app, ["list-sources"])

        assert result.exit_code == 0
        assert "Getting Started" in result.output
        assert "Configuration" in result.output

    @patch("rtfm.redis_client.get_redis")
    def test_list_sources_total_chunks(self, mock_get_redis):
        """Total chunk count is displayed in the summary line."""
        mock_redis = MagicMock()
        mock_get_redis.return_value = mock_redis

        mock_redis.scan.return_value = (0, ["doc:a", "doc:b", "doc:c"])
        mock_redis.hmget.side_effect = [
            ["file1.md", ""],
            ["file2.md", ""],
            ["file2.md", ""],
        ]

        result = runner.invoke(cli_app, ["list-sources"])

        assert result.exit_code == 0
        assert "3 total chunks" in result.output
        assert "2 sources" in result.output


# ── CLI: search ──────────────────────────────────────────────


class TestSearchCLI:
    @patch("rtfm.retrieval.search.search_documents")
    def test_search_displays_results(self, mock_search):
        """Search results show rank, source, score, and text preview."""
        mock_search.return_value = [
            SearchResult(
                text="Git is a distributed version control system that tracks changes.",
                source_file="progit.pdf",
                section="Introduction",
                score=0.0523,
            ),
            SearchResult(
                text="To install Git, use your package manager.",
                source_file="install.md",
                section="Setup",
                score=0.1234,
            ),
        ]

        result = runner.invoke(cli_app, ["search", "how to install git"])

        assert result.exit_code == 0
        assert "2 hits" in result.output
        assert "progit.pdf" in result.output
        assert "Introduction" in result.output
        assert "0.0523" in result.output
        assert "install.md" in result.output
        assert "Setup" in result.output
        assert "0.1234" in result.output
        assert "distributed version control" in result.output

    @patch("rtfm.retrieval.search.search_documents")
    def test_search_no_results(self, mock_search):
        """Empty results show 'No results found' message."""
        mock_search.return_value = []

        result = runner.invoke(cli_app, ["search", "nonexistent topic"])

        assert result.exit_code == 0
        assert "No results found" in result.output

    @patch("rtfm.retrieval.search.search_documents")
    def test_search_with_top_k_flag(self, mock_search):
        """--top-k flag is passed through to search_documents."""
        mock_search.return_value = []

        runner.invoke(cli_app, ["search", "query", "--top-k", "3"])

        mock_search.assert_called_once()
        assert mock_search.call_args.kwargs["top_k"] == 3

    @patch("rtfm.retrieval.search.search_documents")
    def test_search_with_source_flag(self, mock_search):
        """--source flag is passed through as source_filter."""
        mock_search.return_value = []

        runner.invoke(cli_app, ["search", "query", "--source", "readme.md"])

        mock_search.assert_called_once()
        assert mock_search.call_args.kwargs["source_filter"] == "readme.md"

    @patch("rtfm.retrieval.search.search_documents")
    def test_search_with_no_rerank_flag(self, mock_search):
        """--no-rerank disables hybrid search."""
        mock_search.return_value = []

        runner.invoke(cli_app, ["search", "query", "--no-rerank"])

        mock_search.assert_called_once()
        assert mock_search.call_args.kwargs["hybrid"] is False

    @patch("rtfm.retrieval.search.search_documents")
    def test_search_truncates_long_text(self, mock_search):
        """Text longer than 200 chars is truncated with ellipsis in preview."""
        long_text = "A" * 300
        mock_search.return_value = [
            SearchResult(text=long_text, source_file="doc.md", section="", score=0.1),
        ]

        result = runner.invoke(cli_app, ["search", "query"])

        assert result.exit_code == 0
        assert "..." in result.output
        # Full 300-char text should NOT appear
        assert long_text not in result.output


# ── API: GET /health ─────────────────────────────────────────


class TestHealthEndpoint:
    @patch("httpx.AsyncClient.get")
    @patch("rtfm.redis_client.get_redis")
    def test_health_all_healthy(self, mock_get_redis, mock_httpx_get):
        """All services up => status 'healthy'."""
        mock_redis = MagicMock()
        mock_redis.ping.return_value = True
        mock_get_redis.return_value = mock_redis

        # Both Ollama and Memory Server return 200
        mock_resp = AsyncMock()
        mock_resp.status_code = 200
        mock_httpx_get.return_value = mock_resp

        resp = client.get("/health")

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["redis"] is True
        assert data["ollama"] is True
        assert data["memory_server"] is True

    @patch("httpx.AsyncClient.get")
    @patch("rtfm.redis_client.get_redis")
    def test_health_degraded_ollama_down(self, mock_get_redis, mock_httpx_get):
        """Redis up but Ollama down => status 'degraded'."""
        mock_redis = MagicMock()
        mock_redis.ping.return_value = True
        mock_get_redis.return_value = mock_redis

        # Ollama fails, memory server might also fail
        mock_httpx_get.side_effect = Exception("Connection refused")

        resp = client.get("/health")

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "degraded"
        assert data["redis"] is True
        assert data["ollama"] is False

    @patch("httpx.AsyncClient.get")
    @patch("rtfm.redis_client.get_redis")
    def test_health_unhealthy_redis_down(self, mock_get_redis, mock_httpx_get):
        """Redis down => status 'unhealthy' regardless of other services."""
        mock_get_redis.side_effect = Exception("Redis connection failed")

        resp = client.get("/health")

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "unhealthy"
        assert data["redis"] is False

    @patch("httpx.AsyncClient.get")
    @patch("rtfm.redis_client.get_redis")
    def test_health_redis_up_memory_down(self, mock_get_redis, mock_httpx_get):
        """Redis and Ollama up, Memory Server down => still 'healthy'."""
        mock_redis = MagicMock()
        mock_redis.ping.return_value = True
        mock_get_redis.return_value = mock_redis

        # First call is Ollama (success), second is Memory Server (fail)
        mock_resp_ok = AsyncMock()
        mock_resp_ok.status_code = 200
        mock_resp_fail = AsyncMock()
        mock_resp_fail.status_code = 500

        mock_httpx_get.side_effect = [mock_resp_ok, mock_resp_fail]

        resp = client.get("/health")

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["redis"] is True
        assert data["ollama"] is True
        assert data["memory_server"] is False


# ── API: GET /sources ────────────────────────────────────────


class TestSourcesEndpoint:
    @patch("rtfm.redis_client.get_redis")
    def test_sources_returns_json_list(self, mock_get_redis):
        """Sources endpoint returns structured JSON with type, chunks, sections."""
        mock_redis = MagicMock()
        mock_get_redis.return_value = mock_redis

        mock_redis.scan.side_effect = [
            (1, ["doc:a1", "doc:a2"]),
            (0, ["doc:a3"]),
        ]
        mock_redis.hmget.side_effect = [
            ["guide.pdf", "Installation"],
            ["guide.pdf", "Usage"],
            ["https://example.com", "API"],
        ]

        resp = client.get("/sources")

        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2

        # Results are sorted by source name
        file_src = next(s for s in data if s["source"] == "guide.pdf")
        assert file_src["type"] == "file"
        assert file_src["chunks"] == 2
        assert "Installation" in file_src["sections"]
        assert "Usage" in file_src["sections"]

        web_src = next(s for s in data if s["source"] == "https://example.com")
        assert web_src["type"] == "web"
        assert web_src["chunks"] == 1
        assert "API" in web_src["sections"]

    @patch("rtfm.redis_client.get_redis")
    def test_sources_empty(self, mock_get_redis):
        """Empty Redis returns empty array."""
        mock_redis = MagicMock()
        mock_get_redis.return_value = mock_redis
        mock_redis.scan.return_value = (0, [])

        resp = client.get("/sources")

        assert resp.status_code == 200
        data = resp.json()
        assert data == []

    @patch("rtfm.redis_client.get_redis")
    def test_sources_missing_fields_default(self, mock_get_redis):
        """Missing source_file defaults to 'unknown', missing section is excluded."""
        mock_redis = MagicMock()
        mock_get_redis.return_value = mock_redis

        mock_redis.scan.return_value = (0, ["doc:x"])
        mock_redis.hmget.return_value = [None, None]

        resp = client.get("/sources")

        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["source"] == "unknown"
        assert data[0]["sections"] == []
        assert data[0]["type"] == "file"
