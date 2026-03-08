"""Unit tests for rtfm.memory.longterm module.

Mocks httpx to avoid requiring the Redis Agent Memory Server.
"""

from unittest.mock import MagicMock, patch

import pytest

from rtfm.memory.longterm import format_memories, search_memory, store_memory


class TestStoreMemory:
    @patch("rtfm.memory.longterm.httpx.post")
    def test_stores_messages_successfully(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"status": "ok"}
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        result = store_memory("session-1", [{"role": "user", "content": "hello"}])

        assert result == {"status": "ok"}
        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args
        assert "session-1" in call_kwargs.args[0]
        assert call_kwargs.kwargs["json"]["messages"] == [
            {"role": "user", "content": "hello"}
        ]
        assert call_kwargs.kwargs["json"]["namespace"] == "rtfm"

    @patch("rtfm.memory.longterm.httpx.post")
    def test_custom_namespace(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"status": "ok"}
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        store_memory("s1", [{"role": "user", "content": "hi"}], namespace="custom")

        payload = mock_post.call_args.kwargs["json"]
        assert payload["namespace"] == "custom"

    @patch("rtfm.memory.longterm.httpx.post")
    def test_returns_none_on_network_error(self, mock_post):
        mock_post.side_effect = ConnectionError("server down")

        result = store_memory("s1", [{"role": "user", "content": "hi"}])

        assert result is None

    @patch("rtfm.memory.longterm.httpx.post")
    def test_returns_none_on_http_error(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = Exception("500 Internal Server Error")
        mock_post.return_value = mock_resp

        result = store_memory("s1", [{"role": "user", "content": "hi"}])

        assert result is None

    @patch("rtfm.memory.longterm.httpx.post")
    def test_uses_correct_url(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {}
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        store_memory("my-session", [])

        url = mock_post.call_args.args[0]
        assert url.endswith("/sessions/my-session/memory")

    @patch("rtfm.memory.longterm.httpx.post")
    def test_timeout_set(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {}
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        store_memory("s1", [])

        assert mock_post.call_args.kwargs["timeout"] == 10.0


class TestSearchMemory:
    @patch("rtfm.memory.longterm.httpx.post")
    def test_returns_memories(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "memories": [
                {"text": "Python is great", "topics": ["python"]},
                {"text": "Redis is fast", "topics": ["redis"]},
            ]
        }
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        result = search_memory("programming")

        assert len(result) == 2
        assert result[0]["text"] == "Python is great"

    @patch("rtfm.memory.longterm.httpx.post")
    def test_sends_correct_payload(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"memories": []}
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        search_memory("test query", namespace="ns", limit=3)

        payload = mock_post.call_args.kwargs["json"]
        assert payload == {"query": "test query", "namespace": "ns", "limit": 3}

    @patch("rtfm.memory.longterm.httpx.post")
    def test_returns_empty_on_error(self, mock_post):
        mock_post.side_effect = ConnectionError("server down")

        result = search_memory("anything")

        assert result == []

    @patch("rtfm.memory.longterm.httpx.post")
    def test_returns_empty_on_http_error(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = Exception("404")
        mock_post.return_value = mock_resp

        result = search_memory("anything")

        assert result == []

    @patch("rtfm.memory.longterm.httpx.post")
    def test_uses_correct_url(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"memories": []}
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        search_memory("test")

        url = mock_post.call_args.args[0]
        assert url.endswith("/long_term_memory/search")

    @patch("rtfm.memory.longterm.httpx.post")
    def test_default_params(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"memories": []}
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        search_memory("test")

        payload = mock_post.call_args.kwargs["json"]
        assert payload["namespace"] == "rtfm"
        assert payload["limit"] == 5


class TestFormatMemories:
    def test_empty_list_returns_empty_string(self):
        assert format_memories([]) == ""

    def test_single_memory_with_text(self):
        memories = [{"text": "Python was created in 1991", "topics": ["python"]}]
        result = format_memories(memories)

        assert "Remembered context:" in result
        assert "Python was created in 1991" in result
        assert "(topics: python)" in result

    def test_multiple_memories(self):
        memories = [
            {"text": "fact one", "topics": ["a"]},
            {"text": "fact two", "topics": ["b", "c"]},
        ]
        result = format_memories(memories)

        assert "- fact one (topics: a)" in result
        assert "- fact two (topics: b, c)" in result

    def test_memory_with_no_topics(self):
        memories = [{"text": "some fact", "topics": []}]
        result = format_memories(memories)

        assert "- some fact" in result
        assert "(topics:" not in result

    def test_memory_with_content_fallback(self):
        """When 'text' key is missing, falls back to 'content'."""
        memories = [{"content": "fallback content", "topics": []}]
        result = format_memories(memories)

        assert "fallback content" in result

    def test_memory_with_no_text_or_content(self):
        memories = [{"topics": ["x"]}]
        result = format_memories(memories)

        assert "- " in result  # Empty text, but still formatted

    def test_format_starts_with_header(self):
        memories = [{"text": "test", "topics": []}]
        result = format_memories(memories)
        assert result.startswith("Remembered context:\n")
