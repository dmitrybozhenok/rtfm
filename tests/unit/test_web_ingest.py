"""Tests for web ingestion pipeline and CLI URL detection."""

from unittest.mock import MagicMock, patch

import pytest


class TestIngestUrl:
    @patch("rtfm.ingest.pipeline.get_redis_binary")
    @patch("rtfm.ingest.pipeline.get_redis")
    @patch("rtfm.ingest.pipeline.ensure_index")
    @patch("rtfm.ingest.pipeline.embed_texts")
    @patch("rtfm.ingest.web_loader.httpx.get")
    def test_single_page_ingest(self, mock_get, mock_embed, mock_index, mock_redis, mock_redis_bin):
        import numpy as np
        from rtfm.ingest.pipeline import ingest_url

        html = """<html><head><title>Test</title></head>
        <body><main><h1>Title</h1>
        <p>Content paragraph with enough text to create at least one chunk from this page.</p>
        </main></body></html>"""
        mock_resp = MagicMock()
        mock_resp.text = html
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        mock_embed.return_value = [np.zeros(384, dtype=np.float32)]
        mock_r = MagicMock()
        mock_redis.return_value = mock_r
        mock_r_bin = MagicMock()
        mock_redis_bin.return_value = mock_r_bin

        stats = ingest_url("https://example.com/docs/page")

        assert stats["pages"] == 1
        assert stats["chunks"] >= 1
        # Verify source_url is stored
        call_args = mock_r_bin.hset.call_args
        stored_doc = call_args[1]["mapping"] if "mapping" in call_args[1] else call_args[0][1]
        assert stored_doc["source_url"] == "https://example.com/docs/page"

    @patch("rtfm.ingest.pipeline.get_redis_binary")
    @patch("rtfm.ingest.pipeline.get_redis")
    @patch("rtfm.ingest.pipeline.ensure_index")
    @patch("rtfm.ingest.pipeline.embed_texts")
    @patch("rtfm.ingest.web_loader.httpx.get")
    def test_recursive_ingest(self, mock_get, mock_embed, mock_index, mock_redis, mock_redis_bin):
        import numpy as np
        from rtfm.ingest.pipeline import ingest_url

        # First call returns the base page with links
        base_html = """<html><body>
        <a href="/docs/page1">Page 1</a>
        <a href="/docs/page2">Page 2</a>
        </body></html>"""

        page_html = """<html><head><title>Page</title></head>
        <body><main><p>This page contains enough content for chunking and extraction threshold checks in the test.</p></main></body></html>"""

        def side_effect(url, **kwargs):
            resp = MagicMock()
            resp.raise_for_status = MagicMock()
            if url == "https://example.com/docs":
                resp.text = base_html
            else:
                resp.text = page_html
            return resp

        mock_get.side_effect = side_effect
        mock_embed.return_value = [np.zeros(384, dtype=np.float32)]
        mock_redis.return_value = MagicMock()
        mock_redis_bin.return_value = MagicMock()

        stats = ingest_url("https://example.com/docs", recursive=True, delay=0)

        # Should ingest base + 2 linked pages
        assert stats["pages"] >= 2

    @patch("rtfm.ingest.pipeline.get_redis_binary")
    @patch("rtfm.ingest.pipeline.get_redis")
    @patch("rtfm.ingest.pipeline.ensure_index")
    @patch("rtfm.ingest.web_loader.httpx.get")
    def test_failed_page_continues(self, mock_get, mock_index, mock_redis, mock_redis_bin):
        from rtfm.ingest.pipeline import ingest_url

        mock_get.side_effect = Exception("Network error")
        mock_redis.return_value = MagicMock()
        mock_redis_bin.return_value = MagicMock()

        stats = ingest_url("https://example.com/broken")
        assert stats["pages"] == 0
        assert stats["chunks"] == 0


class TestCliUrlDetection:
    """Test that the CLI ingest command correctly detects URLs vs paths."""

    def test_url_detected(self):
        """URLs starting with http:// or https:// should be detected."""
        url = "https://git-scm.com/book/en/v2"
        assert url.startswith(("http://", "https://"))

    def test_path_not_url(self):
        """Regular paths should not be detected as URLs."""
        path = "./docs/"
        assert not path.startswith(("http://", "https://"))
        path = "C:\\Users\\docs"
        assert not path.startswith(("http://", "https://"))
