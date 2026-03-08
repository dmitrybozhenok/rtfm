"""Phase 4 API tests — extended coverage for chat, streaming, sessions, and error paths.

Uses FastAPI TestClient with mocked underlying services.
"""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from rtfm.api.routes import app

client = TestClient(app)

MOCK_ASK_RESULT = {
    "answer": "Test answer",
    "sources": [{"file": "doc.md", "section": "Intro", "score": 0.1}],
    "cached": False,
    "latency_ms": 100.0,
    "tokens_used": 150,
}


class TestChatEndpoint:
    @patch("rtfm.api.routes.store_memory")
    @patch("rtfm.api.routes.summarize_if_needed")
    @patch("rtfm.api.routes.add_message")
    @patch("rtfm.api.routes.format_memories", return_value="")
    @patch("rtfm.api.routes.search_memory", return_value=[])
    @patch("rtfm.api.routes.get_history_truncated", return_value=[])
    @patch("rtfm.api.routes.ask", return_value=MOCK_ASK_RESULT)
    def test_chat_non_streaming(
        self, mock_ask, mock_hist, mock_search_mem, mock_fmt,
        mock_add, mock_summarize, mock_store
    ):
        response = client.post("/chat", data={
            "question": "What is Python?",
            "session_id": "test-session",
        })

        assert response.status_code == 200
        body = response.json()
        assert body["answer"] == "Test answer"
        assert body["session_id"] == "test-session"
        assert "sources" in body

    @patch("rtfm.api.routes.store_memory")
    @patch("rtfm.api.routes.summarize_if_needed")
    @patch("rtfm.api.routes.add_message")
    @patch("rtfm.api.routes.format_memories", return_value="")
    @patch("rtfm.api.routes.search_memory", return_value=[])
    @patch("rtfm.api.routes.get_history_truncated", return_value=[])
    @patch("rtfm.api.routes.ask", return_value=MOCK_ASK_RESULT)
    def test_chat_generates_session_id(
        self, mock_ask, mock_hist, mock_search_mem, mock_fmt,
        mock_add, mock_summarize, mock_store
    ):
        response = client.post("/chat", data={"question": "test"})

        assert response.status_code == 200
        body = response.json()
        assert "session_id" in body
        assert len(body["session_id"]) == 36  # UUID format

    @patch("rtfm.api.routes.store_memory")
    @patch("rtfm.api.routes.summarize_if_needed")
    @patch("rtfm.api.routes.add_message")
    @patch("rtfm.api.routes.format_memories", return_value="")
    @patch("rtfm.api.routes.search_memory", return_value=[])
    @patch("rtfm.api.routes.get_history_truncated", return_value=[])
    @patch("rtfm.api.routes.ask", return_value=MOCK_ASK_RESULT)
    def test_chat_saves_to_session(
        self, mock_ask, mock_hist, mock_search_mem, mock_fmt,
        mock_add, mock_summarize, mock_store
    ):
        client.post("/chat", data={
            "question": "hello",
            "session_id": "s1",
        })

        # Should save user question and assistant answer
        assert mock_add.call_count == 2
        mock_add.assert_any_call("s1", "user", "hello")
        mock_add.assert_any_call("s1", "assistant", "Test answer")
        mock_summarize.assert_called_once_with("s1")

    @patch("rtfm.api.routes.store_memory")
    @patch("rtfm.api.routes.summarize_if_needed")
    @patch("rtfm.api.routes.add_message")
    @patch("rtfm.api.routes.format_memories", return_value="")
    @patch("rtfm.api.routes.search_memory", return_value=[])
    @patch("rtfm.api.routes.get_history_truncated", return_value=[])
    @patch("rtfm.api.routes.ask", return_value=MOCK_ASK_RESULT)
    def test_chat_stores_in_longterm_memory(
        self, mock_ask, mock_hist, mock_search_mem, mock_fmt,
        mock_add, mock_summarize, mock_store
    ):
        client.post("/chat", data={
            "question": "hello",
            "session_id": "s1",
        })

        mock_store.assert_called_once()
        call_args = mock_store.call_args
        assert call_args.args[0] == "s1"
        messages = call_args.args[1]
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"

    @patch("rtfm.api.routes.store_memory")
    @patch("rtfm.api.routes.summarize_if_needed")
    @patch("rtfm.api.routes.add_message")
    @patch("rtfm.api.routes.format_memories", return_value="remembered stuff")
    @patch("rtfm.api.routes.search_memory", return_value=[{"text": "old"}])
    @patch("rtfm.api.routes.get_history_truncated", return_value=[
        {"role": "user", "content": "prev Q"},
    ])
    @patch("rtfm.api.routes.ask", return_value=MOCK_ASK_RESULT)
    def test_chat_passes_history_and_memories_to_ask(
        self, mock_ask, mock_hist, mock_search_mem, mock_fmt,
        mock_add, mock_summarize, mock_store
    ):
        client.post("/chat", data={
            "question": "follow up",
            "session_id": "s1",
        })

        mock_ask.assert_called_once()
        call_kwargs = mock_ask.call_args.kwargs
        assert call_kwargs["session_history"] == [
            {"role": "user", "content": "prev Q"}
        ]
        assert call_kwargs["long_term_memories"] == "remembered stuff"

    @patch("rtfm.api.routes.store_memory")
    @patch("rtfm.api.routes.summarize_if_needed")
    @patch("rtfm.api.routes.add_message")
    @patch("rtfm.api.routes.format_memories", return_value="")
    @patch("rtfm.api.routes.search_memory", return_value=[])
    @patch("rtfm.api.routes.get_history_truncated", return_value=[])
    @patch("rtfm.api.routes.ask", return_value=MOCK_ASK_RESULT)
    def test_chat_with_filters(
        self, mock_ask, mock_hist, mock_search_mem, mock_fmt,
        mock_add, mock_summarize, mock_store
    ):
        client.post("/chat", data={
            "question": "Q?",
            "source_filter": "readme.md",
            "section_filter": "Setup",
        })

        call_kwargs = mock_ask.call_args.kwargs
        assert call_kwargs["source_filter"] == "readme.md"
        assert call_kwargs["section_filter"] == "Setup"


class TestAskEndpointExtended:
    @patch("rtfm.api.routes.ask")
    def test_ask_with_filters(self, mock_ask):
        mock_ask.return_value = MOCK_ASK_RESULT

        response = client.post("/ask", data={
            "question": "Q?",
            "source_filter": "file.md",
            "section_filter": "Intro",
        })

        assert response.status_code == 200
        mock_ask.assert_called_once_with(
            "Q?", source_filter="file.md", section_filter="Intro"
        )

    @patch("rtfm.api.routes.ask")
    def test_ask_no_filters(self, mock_ask):
        mock_ask.return_value = MOCK_ASK_RESULT

        client.post("/ask", data={"question": "Q?"})

        mock_ask.assert_called_once_with(
            "Q?", source_filter=None, section_filter=None
        )


class TestSessionEndpoint:
    @patch("rtfm.api.routes.clear_session")
    def test_clear_session_calls_module(self, mock_clear):
        response = client.post("/session/my-session/clear")

        assert response.status_code == 200
        mock_clear.assert_called_once_with("my-session")

    @patch("rtfm.api.routes.clear_session")
    def test_clear_session_response_format(self, mock_clear):
        response = client.post("/session/sess-42/clear")

        body = response.json()
        assert body["status"] == "ok"
        assert "sess-42" in body["message"]


class TestIngestEndpointExtended:
    @pytest.mark.skipif(
        __import__("sys").platform == "win32",
        reason="Upload endpoint uses hardcoded /tmp path; skipped on Windows",
    )
    @patch("rtfm.api.routes.ingest_path")
    def test_ingest_upload(self, mock_ingest):
        mock_ingest.return_value = {"files": 1, "chunks": 5}

        response = client.post(
            "/ingest",
            files={"file": ("test.md", b"# Hello\n\nContent", "text/markdown")},
        )

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["files"] == 1
        assert body["chunks"] == 5
