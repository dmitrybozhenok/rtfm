"""Unit tests for rtfm.retrieval.rag module.

Mocks LLM, search, and cache to test orchestration logic.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from rtfm.retrieval.rag import (
    SYSTEM_PROMPT,
    _build_messages,
    _format_context,
    ask,
    ask_stream,
)
from rtfm.retrieval.search import SearchResult


# ── Helpers ──────────────────────────────────────────────────


def _make_search_results(n=2):
    return [
        SearchResult(
            text=f"Document chunk {i} content.",
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
    """Return an iterable that mimics OpenAI streaming."""
    for text in chunks:
        delta = SimpleNamespace(content=text)
        choice = SimpleNamespace(delta=delta)
        yield SimpleNamespace(choices=[choice])


@pytest.fixture()
def mock_cache():
    """Mock check_cache and store_cache at the import source."""
    with patch("rtfm.cache.semantic_cache.check_cache", return_value=None) as m_check, \
         patch("rtfm.cache.semantic_cache.store_cache") as m_store:
        yield m_check, m_store


@pytest.fixture()
def mock_llm():
    """Mock the OpenAI client."""
    with patch("rtfm.retrieval.rag._get_client") as m:
        client = MagicMock()
        m.return_value = client
        yield client


@pytest.fixture()
def mock_search():
    """Mock search_documents."""
    with patch("rtfm.retrieval.rag.search_documents") as m:
        yield m


@pytest.fixture()
def mock_metrics():
    """Mock _record_metrics to avoid Redis."""
    with patch("rtfm.retrieval.rag._record_metrics"):
        yield


# ── _format_context tests ────────────────────────────────────


class TestFormatContext:
    def test_empty_results(self):
        assert _format_context([]) == "No relevant documentation found."

    def test_single_result_with_section(self):
        results = [SearchResult("hello", "file.md", "Intro", 0.1)]
        ctx = _format_context(results)
        assert "[Source 1: file.md (section: Intro)]" in ctx
        assert "hello" in ctx

    def test_single_result_no_section(self):
        results = [SearchResult("hello", "file.md", "", 0.1)]
        ctx = _format_context(results)
        assert "[Source 1: file.md]" in ctx
        assert "(section:" not in ctx

    def test_multiple_results_separated(self):
        results = _make_search_results(3)
        ctx = _format_context(results)
        assert ctx.count("---") == 2  # 3 results, 2 separators

    def test_preserves_document_text(self):
        results = [SearchResult("The quick brown fox", "a.md", "", 0.1)]
        ctx = _format_context(results)
        assert "The quick brown fox" in ctx


# ── _build_messages tests ────────────────────────────────────


class TestBuildMessages:
    def test_basic_structure(self):
        msgs = _build_messages("What is X?", "context here", "system prompt")
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[0]["content"] == "system prompt"
        assert msgs[1]["role"] == "user"
        assert "What is X?" in msgs[1]["content"]
        assert "context here" in msgs[1]["content"]

    def test_with_session_history(self):
        history = [
            {"role": "user", "content": "previous question"},
            {"role": "assistant", "content": "previous answer"},
        ]
        msgs = _build_messages("new Q?", "ctx", "sys", session_history=history)
        assert len(msgs) == 4
        assert msgs[0]["role"] == "system"
        assert msgs[1] == history[0]
        assert msgs[2] == history[1]
        assert msgs[3]["role"] == "user"
        assert "new Q?" in msgs[3]["content"]

    def test_without_session_history(self):
        msgs = _build_messages("Q?", "ctx", "sys", session_history=None)
        assert len(msgs) == 2

    def test_empty_session_history(self):
        msgs = _build_messages("Q?", "ctx", "sys", session_history=[])
        assert len(msgs) == 2


# ── ask() tests ──────────────────────────────────────────────


class TestAsk:
    def test_cache_miss_full_pipeline(self, mock_cache, mock_search, mock_llm, mock_metrics):
        mock_search.return_value = _make_search_results(2)
        mock_llm.chat.completions.create.return_value = _mock_llm_response(
            "Paris is the capital."
        )

        result = ask("What is the capital of France?")

        assert result["answer"] == "Paris is the capital."
        assert result["cached"] is False
        assert result["tokens_used"] == 150
        assert len(result["sources"]) == 2
        assert result["sources"][0]["file"] == "doc1.md"
        assert isinstance(result["latency_ms"], float)

    def test_cache_hit_returns_early(self, mock_search, mock_metrics):
        with patch("rtfm.cache.semantic_cache.check_cache", return_value="Cached answer"), \
             patch("rtfm.cache.semantic_cache.store_cache"):
            result = ask("cached question")

        assert result["answer"] == "Cached answer"
        assert result["cached"] is True
        assert result["tokens_used"] == 0
        assert result["sources"] == []

    def test_cache_exception_degrades_gracefully(self, mock_search, mock_llm, mock_metrics):
        """If cache check fails, pipeline continues without caching."""
        mock_search.return_value = _make_search_results(1)
        mock_llm.chat.completions.create.return_value = _mock_llm_response("answer")

        with patch("rtfm.cache.semantic_cache.check_cache", side_effect=Exception("broken")), \
             patch("rtfm.cache.semantic_cache.store_cache", side_effect=Exception("broken")):
            result = ask("question")

        assert result["answer"] == "answer"
        assert result["cached"] is False

    def test_passes_filters_to_search(self, mock_cache, mock_search, mock_llm, mock_metrics):
        mock_search.return_value = []
        mock_llm.chat.completions.create.return_value = _mock_llm_response("no info")

        ask("Q?", source_filter="readme.md", section_filter="Install")

        mock_search.assert_called_once_with(
            "Q?", source_filter="readme.md", section_filter="Install"
        )

    def test_session_history_passed_to_llm(self, mock_cache, mock_search, mock_llm, mock_metrics):
        mock_search.return_value = []
        mock_llm.chat.completions.create.return_value = _mock_llm_response("answer")

        history = [{"role": "user", "content": "previous Q"}]
        ask("new Q?", session_history=history)

        call_args = mock_llm.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]
        assert len(messages) == 3
        assert messages[1] == history[0]

    def test_long_term_memories_in_system_prompt(self, mock_cache, mock_search, mock_llm, mock_metrics):
        mock_search.return_value = []
        mock_llm.chat.completions.create.return_value = _mock_llm_response("answer")

        ask("Q?", long_term_memories="User prefers Python 3.12")

        call_args = mock_llm.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]
        system_content = messages[0]["content"]
        assert "User prefers Python 3.12" in system_content

    def test_empty_search_results_still_calls_llm(self, mock_cache, mock_search, mock_llm, mock_metrics):
        mock_search.return_value = []
        mock_llm.chat.completions.create.return_value = _mock_llm_response(
            "I don't have enough information"
        )

        result = ask("unknown topic")

        assert "don't have enough information" in result["answer"]
        assert result["sources"] == []

    def test_no_usage_stats_defaults_to_zero(self, mock_cache, mock_search, mock_llm, mock_metrics):
        mock_search.return_value = []
        response = _mock_llm_response("answer")
        response.usage = None
        mock_llm.chat.completions.create.return_value = response

        result = ask("Q?")

        assert result["tokens_used"] == 0

    def test_response_structure_complete(self, mock_cache, mock_search, mock_llm, mock_metrics):
        mock_search.return_value = _make_search_results(1)
        mock_llm.chat.completions.create.return_value = _mock_llm_response("ans")

        result = ask("Q?")

        expected_keys = {"answer", "sources", "cached", "latency_ms", "tokens_used"}
        assert set(result.keys()) == expected_keys
        assert isinstance(result["answer"], str)
        assert isinstance(result["sources"], list)
        assert isinstance(result["cached"], bool)
        assert isinstance(result["latency_ms"], float)
        assert isinstance(result["tokens_used"], int)


# ── ask_stream() tests ───────────────────────────────────────


class TestAskStream:
    def test_yields_chunks(self, mock_cache, mock_search, mock_llm):
        mock_search.return_value = _make_search_results(1)
        mock_llm.chat.completions.create.return_value = _mock_llm_stream(
            ["Hello", " world", "!"]
        )

        chunks = list(ask_stream("Q?"))

        assert chunks == ["Hello", " world", "!"]

    def test_cache_hit_yields_cached_answer(self):
        with patch("rtfm.cache.semantic_cache.check_cache", return_value="Cached!"), \
             patch("rtfm.cache.semantic_cache.store_cache"):
            chunks = list(ask_stream("cached Q"))

        assert chunks == ["Cached!"]

    def test_caches_full_answer_after_streaming(self, mock_search, mock_llm):
        mock_search.return_value = []
        mock_llm.chat.completions.create.return_value = _mock_llm_stream(
            ["part1", "part2"]
        )

        with patch("rtfm.cache.semantic_cache.check_cache", return_value=None), \
             patch("rtfm.cache.semantic_cache.store_cache") as mock_store:
            list(ask_stream("Q?"))  # Consume generator

        mock_store.assert_called_once_with("Q?", "part1part2")


# ── _record_metrics tests ────────────────────────────────────


class TestRecordMetrics:
    def test_records_cache_hit(self):
        from rtfm.retrieval.rag import _record_metrics

        r = MagicMock()
        with patch("rtfm.redis_client.get_redis", return_value=r):
            _record_metrics(100.0, cached=True, tokens=0)

        r.incr.assert_any_call("rtfm:metrics:cache_hits")
        r.incr.assert_any_call("rtfm:metrics:total_queries")

    def test_records_cache_miss(self):
        from rtfm.retrieval.rag import _record_metrics

        r = MagicMock()
        with patch("rtfm.redis_client.get_redis", return_value=r):
            _record_metrics(200.0, cached=False, tokens=500)

        r.incr.assert_any_call("rtfm:metrics:cache_misses")
        r.incrby.assert_called_with("rtfm:metrics:total_tokens", 500)

    def test_metrics_failure_silenced(self):
        from rtfm.retrieval.rag import _record_metrics

        with patch("rtfm.redis_client.get_redis", side_effect=Exception("redis down")):
            _record_metrics(100.0, cached=True, tokens=0)  # Should not raise
