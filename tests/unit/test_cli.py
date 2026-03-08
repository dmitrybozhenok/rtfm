"""Unit tests for rtfm.cli.app module.

Uses Typer's CliRunner and mocks underlying services.
CLI commands use lazy imports, so we patch at the source module.
"""

from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from rtfm.cli.app import app

runner = CliRunner()

MOCK_ASK_RESULT = {
    "answer": "Paris is the capital of France.",
    "sources": [{"file": "geo.md", "section": "Europe", "score": 0.1}],
    "cached": False,
    "latency_ms": 123.4,
    "tokens_used": 100,
}


class TestIngestCommand:
    @patch("rtfm.ingest.pipeline.ingest_path")
    def test_ingest_success(self, mock_ingest, tmp_path):
        doc = tmp_path / "test.md"
        doc.write_text("# Test", encoding="utf-8")

        mock_ingest.return_value = {"files": 1, "chunks": 3}

        result = runner.invoke(app, ["ingest", str(tmp_path)])

        assert result.exit_code == 0
        assert "1 files" in result.output
        assert "3 chunks" in result.output

    def test_ingest_nonexistent_path(self):
        result = runner.invoke(app, ["ingest", "/nonexistent/path/xyz"])

        assert result.exit_code == 1
        assert "not found" in result.output.lower()

    @patch("rtfm.ingest.pipeline.ingest_path")
    def test_ingest_single_file(self, mock_ingest, tmp_path):
        f = tmp_path / "single.md"
        f.write_text("content", encoding="utf-8")
        mock_ingest.return_value = {"files": 1, "chunks": 1}

        result = runner.invoke(app, ["ingest", str(f)])

        assert result.exit_code == 0
        mock_ingest.assert_called_once()


class TestAskCommand:
    @patch("rtfm.retrieval.rag.ask")
    def test_ask_displays_answer(self, mock_ask):
        mock_ask.return_value = MOCK_ASK_RESULT

        result = runner.invoke(app, ["ask", "What is the capital of France?"])

        assert result.exit_code == 0
        assert "Paris" in result.output

    @patch("rtfm.retrieval.rag.ask")
    def test_ask_shows_metadata(self, mock_ask):
        mock_ask.return_value = {
            **MOCK_ASK_RESULT,
            "sources": [],
            "latency_ms": 50.0,
            "tokens_used": 75,
        }

        result = runner.invoke(app, ["ask", "test question"])

        assert result.exit_code == 0
        assert "50.0" in result.output
        assert "75" in result.output

    @patch("rtfm.retrieval.rag.ask")
    def test_ask_shows_cached(self, mock_ask):
        mock_ask.return_value = {
            **MOCK_ASK_RESULT,
            "cached": True,
            "tokens_used": 0,
            "latency_ms": 5.0,
        }

        result = runner.invoke(app, ["ask", "cached Q"])

        assert result.exit_code == 0
        assert "cache" in result.output.lower()

    @patch("rtfm.retrieval.rag.ask")
    def test_ask_shows_sources(self, mock_ask):
        mock_ask.return_value = {
            **MOCK_ASK_RESULT,
            "sources": [
                {"file": "doc.md", "section": "Setup", "score": 0.05},
                {"file": "readme.md", "section": "", "score": 0.12},
            ],
        }

        result = runner.invoke(app, ["ask", "how to setup?"])

        assert result.exit_code == 0
        assert "doc.md" in result.output
        assert "Setup" in result.output
        assert "readme.md" in result.output

    @patch("rtfm.retrieval.rag.ask")
    def test_ask_with_source_filter(self, mock_ask):
        mock_ask.return_value = MOCK_ASK_RESULT

        runner.invoke(app, ["ask", "Q?", "--source", "readme.md"])

        mock_ask.assert_called_once_with(
            "Q?", source_filter="readme.md", section_filter=None
        )

    @patch("rtfm.retrieval.rag.ask")
    def test_ask_with_section_filter(self, mock_ask):
        mock_ask.return_value = MOCK_ASK_RESULT

        runner.invoke(app, ["ask", "Q?", "--section", "Install"])

        mock_ask.assert_called_once_with(
            "Q?", source_filter=None, section_filter="Install"
        )


class TestClearCacheCommand:
    @patch("rtfm.cache.semantic_cache.flush_cache")
    def test_clear_cache(self, mock_flush):
        result = runner.invoke(app, ["clear-cache"])

        assert result.exit_code == 0
        assert "flushed" in result.output.lower()
        mock_flush.assert_called_once()
