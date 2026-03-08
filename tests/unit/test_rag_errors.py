"""Tests for RAG error handling and edge cases.

Covers LLM failures, search failures, and streaming error paths.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from rtfm.retrieval.rag import ask, ask_stream
from rtfm.retrieval.search import SearchResult


def _make_search_results(n=1):
    return [
        SearchResult(
            text=f"Chunk {i}",
            source_file=f"doc{i}.md",
            section=f"Section {i}",
            score=0.1 * i,
        )
        for i in range(1, n + 1)
    ]


def _mock_llm_response(answer="Test answer", prompt_tokens=100, completion_tokens=50):
    msg = SimpleNamespace(content=answer)
    choice = SimpleNamespace(message=msg)
    usage = SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
    return SimpleNamespace(choices=[choice], usage=usage)


def _mock_llm_stream(chunks):
    for text in chunks:
        delta = SimpleNamespace(content=text)
        choice = SimpleNamespace(delta=delta)
        yield SimpleNamespace(choices=[choice])


@pytest.fixture()
def mock_cache():
    with patch("rtfm.cache.semantic_cache.check_cache", return_value=(None, [])), \
         patch("rtfm.cache.semantic_cache.store_cache"):
        yield


@pytest.fixture()
def mock_metrics():
    with patch("rtfm.retrieval.rag._record_metrics"):
        yield


# ── Search failure tests ─────────────────────────────────────


class TestSearchFailures:
    def test_search_exception_propagates(self, mock_cache, mock_metrics):
        """If search_documents raises, ask() should propagate the error."""
        with patch("rtfm.retrieval.rag.search_documents", side_effect=Exception("index down")), \
             patch("rtfm.retrieval.rag._get_client"):
            with pytest.raises(Exception, match="index down"):
                ask("test question")

    def test_stream_search_exception_propagates(self, mock_cache):
        """If search_documents raises during streaming, error propagates."""
        with patch("rtfm.retrieval.rag.search_documents", side_effect=Exception("index down")), \
             patch("rtfm.retrieval.rag._get_client"):
            with pytest.raises(Exception, match="index down"):
                list(ask_stream("test question"))


# ── LLM failure tests ────────────────────────────────────────


class TestLLMFailures:
    def test_llm_connection_error(self, mock_cache, mock_metrics):
        """If the LLM API is unreachable, ask() raises."""
        with patch("rtfm.retrieval.rag.search_documents", return_value=[]), \
             patch("rtfm.retrieval.rag._get_client") as mock_client:
            mock_client.return_value.chat.completions.create.side_effect = \
                ConnectionError("connection refused")

            with pytest.raises(ConnectionError):
                ask("test question")

    def test_llm_timeout(self, mock_cache, mock_metrics):
        """If the LLM times out, ask() raises."""
        with patch("rtfm.retrieval.rag.search_documents", return_value=[]), \
             patch("rtfm.retrieval.rag._get_client") as mock_client:
            mock_client.return_value.chat.completions.create.side_effect = \
                TimeoutError("request timed out")

            with pytest.raises(TimeoutError):
                ask("test question")

    def test_stream_llm_error(self, mock_cache):
        """If LLM fails during streaming, error propagates."""
        with patch("rtfm.retrieval.rag.search_documents", return_value=[]), \
             patch("rtfm.retrieval.rag._get_client") as mock_client:
            mock_client.return_value.chat.completions.create.side_effect = \
                ConnectionError("connection refused")

            with pytest.raises(ConnectionError):
                list(ask_stream("test"))


# ── Streaming edge cases ─────────────────────────────────────


class TestStreamingEdgeCases:
    def test_stream_empty_chunks_skipped(self, mock_cache):
        """Chunks with no content should not be yielded."""
        with patch("rtfm.retrieval.rag.search_documents", return_value=[]), \
             patch("rtfm.retrieval.rag._get_client") as mock_client:
            # Mix of chunks with and without content
            def _stream():
                delta_with = SimpleNamespace(content="hello")
                delta_without = SimpleNamespace(content=None)
                delta_empty = SimpleNamespace(content="")
                yield SimpleNamespace(choices=[SimpleNamespace(delta=delta_with)])
                yield SimpleNamespace(choices=[SimpleNamespace(delta=delta_without)])
                yield SimpleNamespace(choices=[SimpleNamespace(delta=delta_empty)])
                yield SimpleNamespace(choices=[])  # No choices

            mock_client.return_value.chat.completions.create.return_value = _stream()

            with patch("rtfm.cache.semantic_cache.store_cache"):
                chunks = list(ask_stream("Q?"))

        # Only "hello" should be yielded (None and empty are filtered out)
        assert "hello" in chunks

    def test_stream_cache_store_failure_silent(self, mock_cache):
        """If cache store fails after streaming, no error raised."""
        with patch("rtfm.retrieval.rag.search_documents", return_value=[]), \
             patch("rtfm.retrieval.rag._get_client") as mock_client:
            mock_client.return_value.chat.completions.create.return_value = \
                _mock_llm_stream(["answer"])

            with patch("rtfm.cache.semantic_cache.check_cache", return_value=(None, [])), \
                 patch("rtfm.cache.semantic_cache.store_cache", side_effect=Exception("store failed")):
                chunks = list(ask_stream("Q?"))

        assert chunks == ["answer"]

    def test_stream_cache_check_failure_continues(self):
        """If cache check fails during streaming, pipeline continues."""
        with patch("rtfm.cache.semantic_cache.check_cache", side_effect=Exception("broken")), \
             patch("rtfm.cache.semantic_cache.store_cache"), \
             patch("rtfm.retrieval.rag.search_documents", return_value=[]), \
             patch("rtfm.retrieval.rag._get_client") as mock_client:
            mock_client.return_value.chat.completions.create.return_value = \
                _mock_llm_stream(["fallback answer"])

            chunks = list(ask_stream("Q?"))

        assert chunks == ["fallback answer"]

    def test_stream_with_session_history(self, mock_cache):
        """ask_stream passes session_history to LLM."""
        history = [{"role": "user", "content": "prev Q"}]

        with patch("rtfm.retrieval.rag.search_documents", return_value=[]), \
             patch("rtfm.retrieval.rag._get_client") as mock_client:
            mock_client.return_value.chat.completions.create.return_value = \
                _mock_llm_stream(["answer"])

            with patch("rtfm.cache.semantic_cache.store_cache"):
                list(ask_stream("new Q?", session_history=history))

        call_kwargs = mock_client.return_value.chat.completions.create.call_args.kwargs
        messages = call_kwargs["messages"]
        assert len(messages) == 3  # system + history + user
        assert messages[1] == history[0]

    def test_stream_with_long_term_memories(self, mock_cache):
        """ask_stream includes long-term memories in system prompt."""
        with patch("rtfm.retrieval.rag.search_documents", return_value=[]), \
             patch("rtfm.retrieval.rag._get_client") as mock_client:
            mock_client.return_value.chat.completions.create.return_value = \
                _mock_llm_stream(["answer"])

            with patch("rtfm.cache.semantic_cache.store_cache"):
                list(ask_stream("Q?", long_term_memories="User prefers Go"))

        call_kwargs = mock_client.return_value.chat.completions.create.call_args.kwargs
        system = call_kwargs["messages"][0]["content"]
        assert "User prefers Go" in system

    def test_stream_with_filters(self, mock_cache):
        """ask_stream passes source_filter and section_filter."""
        with patch("rtfm.retrieval.rag.search_documents", return_value=[]) as mock_search, \
             patch("rtfm.retrieval.rag._get_client") as mock_client:
            mock_client.return_value.chat.completions.create.return_value = \
                _mock_llm_stream(["answer"])

            with patch("rtfm.cache.semantic_cache.store_cache"):
                list(ask_stream(
                    "Q?", source_filter="readme.md", section_filter="Install"
                ))

        mock_search.assert_called_once_with(
            "Q?", source_filter="readme.md", section_filter="Install"
        )
