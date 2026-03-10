"""Tests for the streaming chat endpoint and streaming error paths.

Uses FastAPI TestClient with mocked services.
"""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from rtfm.api.routes import app

client = TestClient(app)


def _chat_patches(**overrides):
    """Return context managers for all chat dependencies."""
    defaults = {
        "rtfm.api.routes.store_memory": MagicMock(),
        "rtfm.api.routes.summarize_if_needed": MagicMock(),
        "rtfm.api.routes.add_message": MagicMock(),
        "rtfm.api.routes.format_memories": "",
        "rtfm.api.routes.search_memory": [],
        "rtfm.api.routes.get_history_truncated": [],
    }
    defaults.update(overrides)
    return defaults


def _stream_chunks(texts):
    """Generator that yields dicts like ask_stream."""
    for t in texts:
        yield {"token": t}
    yield {"sources": [{"file": "test.md", "section": "", "score": 0.9}]}


class TestStreamingChat:
    @patch("rtfm.api.routes.store_memory")
    @patch("rtfm.api.routes.summarize_if_needed")
    @patch("rtfm.api.routes.add_message")
    @patch("rtfm.api.routes.format_memories", return_value="")
    @patch("rtfm.api.routes.search_memory", return_value=[])
    @patch("rtfm.api.routes.get_history_truncated", return_value=[])
    @patch("rtfm.api.routes.ask_stream")
    def test_streaming_returns_sse_events(
        self, mock_stream, mock_hist, mock_search_mem, mock_fmt,
        mock_add, mock_summarize, mock_store
    ):
        mock_stream.return_value = _stream_chunks(["Hello", " world", "!"])

        with client.stream(
            "POST", "/chat",
            data={"question": "test", "session_id": "s1", "stream": "true"},
        ) as response:
            assert response.status_code == 200
            body = response.read().decode()

        assert "Hello" in body
        assert "world" in body

    @patch("rtfm.api.routes.store_memory")
    @patch("rtfm.api.routes.summarize_if_needed")
    @patch("rtfm.api.routes.add_message")
    @patch("rtfm.api.routes.format_memories", return_value="")
    @patch("rtfm.api.routes.search_memory", return_value=[])
    @patch("rtfm.api.routes.get_history_truncated", return_value=[])
    @patch("rtfm.api.routes.ask_stream")
    def test_streaming_saves_session_messages(
        self, mock_stream, mock_hist, mock_search_mem, mock_fmt,
        mock_add, mock_summarize, mock_store
    ):
        mock_stream.return_value = _stream_chunks(["The answer"])

        with client.stream(
            "POST", "/chat",
            data={"question": "Q?", "session_id": "s1", "stream": "true"},
        ) as response:
            response.read()

        assert mock_add.call_count == 2
        mock_add.assert_any_call("s1", "user", "Q?")
        mock_add.assert_any_call("s1", "assistant", "The answer")

    @patch("rtfm.api.routes.store_memory")
    @patch("rtfm.api.routes.summarize_if_needed")
    @patch("rtfm.api.routes.add_message")
    @patch("rtfm.api.routes.format_memories", return_value="")
    @patch("rtfm.api.routes.search_memory", return_value=[])
    @patch("rtfm.api.routes.get_history_truncated", return_value=[])
    @patch("rtfm.api.routes.ask_stream")
    def test_streaming_stores_longterm_memory(
        self, mock_stream, mock_hist, mock_search_mem, mock_fmt,
        mock_add, mock_summarize, mock_store
    ):
        mock_stream.return_value = _stream_chunks(["answer text"])

        with client.stream(
            "POST", "/chat",
            data={"question": "Q?", "session_id": "s1", "stream": "true"},
        ) as response:
            response.read()

        mock_store.assert_called_once()
        args = mock_store.call_args.args
        assert args[0] == "s1"
        assert args[1][0]["role"] == "user"
        assert args[1][1]["role"] == "assistant"
        assert args[1][1]["content"] == "answer text"

    @patch("rtfm.api.routes.store_memory")
    @patch("rtfm.api.routes.summarize_if_needed")
    @patch("rtfm.api.routes.add_message")
    @patch("rtfm.api.routes.format_memories", return_value="")
    @patch("rtfm.api.routes.search_memory", return_value=[])
    @patch("rtfm.api.routes.get_history_truncated", return_value=[])
    @patch("rtfm.api.routes.ask_stream")
    def test_streaming_calls_summarize(
        self, mock_stream, mock_hist, mock_search_mem, mock_fmt,
        mock_add, mock_summarize, mock_store
    ):
        mock_stream.return_value = _stream_chunks(["answer"])

        with client.stream(
            "POST", "/chat",
            data={"question": "Q?", "session_id": "s1", "stream": "true"},
        ) as response:
            response.read()

        mock_summarize.assert_called_once_with("s1")

    @patch("rtfm.api.routes.store_memory")
    @patch("rtfm.api.routes.summarize_if_needed")
    @patch("rtfm.api.routes.add_message")
    @patch("rtfm.api.routes.format_memories", return_value="mem context")
    @patch("rtfm.api.routes.search_memory", return_value=[{"text": "old fact"}])
    @patch("rtfm.api.routes.get_history_truncated", return_value=[
        {"role": "user", "content": "prev"},
    ])
    @patch("rtfm.api.routes.ask_stream")
    def test_streaming_passes_history_and_memories(
        self, mock_stream, mock_hist, mock_search_mem, mock_fmt,
        mock_add, mock_summarize, mock_store
    ):
        mock_stream.return_value = _stream_chunks(["ans"])

        with client.stream(
            "POST", "/chat",
            data={"question": "follow up", "session_id": "s1", "stream": "true"},
        ) as response:
            response.read()

        call_kwargs = mock_stream.call_args.kwargs
        assert call_kwargs["session_history"] == [{"role": "user", "content": "prev"}]
        assert call_kwargs["long_term_memories"] == "mem context"

    @patch("rtfm.api.routes.store_memory")
    @patch("rtfm.api.routes.summarize_if_needed")
    @patch("rtfm.api.routes.add_message")
    @patch("rtfm.api.routes.format_memories", return_value="")
    @patch("rtfm.api.routes.search_memory", return_value=[])
    @patch("rtfm.api.routes.get_history_truncated", return_value=[])
    @patch("rtfm.api.routes.ask_stream")
    def test_streaming_concatenates_chunks_for_session(
        self, mock_stream, mock_hist, mock_search_mem, mock_fmt,
        mock_add, mock_summarize, mock_store
    ):
        """Full answer stored in session is the concatenation of all chunks."""
        mock_stream.return_value = _stream_chunks(["part1", " part2", " part3"])

        with client.stream(
            "POST", "/chat",
            data={"question": "Q?", "session_id": "s1", "stream": "true"},
        ) as response:
            response.read()

        # The assistant message should be the full concatenated answer
        assistant_call = [
            c for c in mock_add.call_args_list if c.args[1] == "assistant"
        ]
        assert len(assistant_call) == 1
        assert assistant_call[0].args[2] == "part1 part2 part3"

    @patch("rtfm.api.routes.store_memory")
    @patch("rtfm.api.routes.summarize_if_needed")
    @patch("rtfm.api.routes.add_message")
    @patch("rtfm.api.routes.format_memories", return_value="")
    @patch("rtfm.api.routes.search_memory", return_value=[])
    @patch("rtfm.api.routes.get_history_truncated", return_value=[])
    @patch("rtfm.api.routes.ask_stream")
    def test_streaming_with_filters(
        self, mock_stream, mock_hist, mock_search_mem, mock_fmt,
        mock_add, mock_summarize, mock_store
    ):
        mock_stream.return_value = _stream_chunks(["ans"])

        with client.stream(
            "POST", "/chat",
            data={
                "question": "Q?",
                "session_id": "s1",
                "stream": "true",
                "source_filter": "readme.md",
                "section_filter": "Setup",
            },
        ) as response:
            response.read()

        call_kwargs = mock_stream.call_args.kwargs
        assert call_kwargs["source_filter"] == "readme.md"
        assert call_kwargs["section_filter"] == "Setup"

    @patch("rtfm.api.routes.store_memory")
    @patch("rtfm.api.routes.summarize_if_needed")
    @patch("rtfm.api.routes.add_message")
    @patch("rtfm.api.routes.format_memories", return_value="")
    @patch("rtfm.api.routes.search_memory", return_value=[])
    @patch("rtfm.api.routes.get_history_truncated", return_value=[])
    @patch("rtfm.api.routes.ask_stream")
    def test_streaming_generates_session_id_when_missing(
        self, mock_stream, mock_hist, mock_search_mem, mock_fmt,
        mock_add, mock_summarize, mock_store
    ):
        mock_stream.return_value = _stream_chunks(["ans"])

        with client.stream(
            "POST", "/chat",
            data={"question": "Q?", "stream": "true"},
        ) as response:
            response.read()

        # add_message should have been called with a UUID session_id
        session_id = mock_add.call_args_list[0].args[0]
        assert len(session_id) == 36  # UUID format
