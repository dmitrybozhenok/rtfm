"""Unit tests for the CLI chat command.

Mocks all external dependencies (RAG, session, memory, console I/O).
"""

from unittest.mock import MagicMock, call, patch

import pytest
from typer.testing import CliRunner

from rtfm.cli.app import app

runner = CliRunner()

MOCK_ASK_RESULT = {
    "answer": "Git is a version control system.",
    "sources": [{"file": "progit.md", "section": "Basics", "score": 0.05}],
    "cached": False,
    "latency_ms": 200.0,
    "tokens_used": 120,
}


def _patches():
    """Return a dict of all patches needed for the chat command."""
    return {
        "rag_ask": patch("rtfm.retrieval.rag.ask", return_value=MOCK_ASK_RESULT),
        "get_history_truncated": patch(
            "rtfm.memory.session.get_history_truncated", return_value=[]
        ),
        "get_history": patch(
            "rtfm.memory.session.get_history", return_value=[]
        ),
        "add_message": patch("rtfm.memory.session.add_message"),
        "clear_session": patch("rtfm.memory.session.clear_session"),
        "summarize": patch("rtfm.memory.session.summarize_if_needed"),
        "search_memory": patch(
            "rtfm.memory.longterm.search_memory", return_value=[]
        ),
        "format_memories": patch(
            "rtfm.memory.longterm.format_memories", return_value=""
        ),
        "store_memory": patch("rtfm.memory.longterm.store_memory"),
    }


class TestChatQuit:
    def test_quit_exits_cleanly(self):
        with _patches()["rag_ask"], _patches()["get_history"], \
             _patches()["search_memory"], _patches()["format_memories"], \
             _patches()["store_memory"]:
            result = runner.invoke(app, ["chat"], input="/quit\n")

        assert result.exit_code == 0
        assert "Goodbye" in result.output

    def test_quit_stores_longterm_memory_when_history_exists(self):
        history = [
            {"role": "user", "content": "What is git?"},
            {"role": "assistant", "content": "A VCS."},
        ]
        with _patches()["rag_ask"], \
             patch("rtfm.memory.session.get_history", return_value=history), \
             _patches()["search_memory"], _patches()["format_memories"], \
             patch("rtfm.memory.longterm.store_memory") as mock_store:
            runner.invoke(app, ["chat"], input="/quit\n")

        mock_store.assert_called_once()
        args = mock_store.call_args.args
        assert args[1] == history  # passes history to store_memory

    def test_quit_skips_store_when_no_history(self):
        with _patches()["rag_ask"], \
             patch("rtfm.memory.session.get_history", return_value=[]), \
             _patches()["search_memory"], _patches()["format_memories"], \
             patch("rtfm.memory.longterm.store_memory") as mock_store:
            runner.invoke(app, ["chat"], input="/quit\n")

        mock_store.assert_not_called()


class TestChatClear:
    def test_clear_resets_session(self):
        with _patches()["rag_ask"], _patches()["get_history"], \
             patch("rtfm.memory.session.clear_session") as mock_clear, \
             _patches()["search_memory"], _patches()["format_memories"], \
             _patches()["store_memory"]:
            result = runner.invoke(app, ["chat"], input="/clear\n/quit\n")

        assert result.exit_code == 0
        assert "cleared" in result.output.lower()
        mock_clear.assert_called_once()

    def test_clear_resets_sources(self):
        """After /clear, /sources should show 'No sources available'."""
        with _patches()["rag_ask"], _patches()["get_history"], \
             _patches()["clear_session"], \
             _patches()["search_memory"], _patches()["format_memories"], \
             _patches()["store_memory"]:
            result = runner.invoke(
                app, ["chat"], input="/clear\n/sources\n/quit\n"
            )

        assert "No sources" in result.output


class TestChatSources:
    def test_sources_displays_last_answer_sources(self):
        with patch("rtfm.retrieval.rag.ask", return_value=MOCK_ASK_RESULT), \
             _patches()["get_history_truncated"], _patches()["get_history"], \
             _patches()["add_message"], _patches()["summarize"], \
             _patches()["search_memory"], _patches()["format_memories"], \
             _patches()["store_memory"]:
            result = runner.invoke(
                app, ["chat"], input="What is git?\n/sources\n/quit\n"
            )

        assert "progit.md" in result.output
        assert "Basics" in result.output

    def test_sources_without_prior_answer(self):
        with _patches()["rag_ask"], _patches()["get_history"], \
             _patches()["search_memory"], _patches()["format_memories"], \
             _patches()["store_memory"]:
            result = runner.invoke(
                app, ["chat"], input="/sources\n/quit\n"
            )

        assert "No sources" in result.output


class TestChatConversation:
    def test_ask_question_displays_answer(self):
        with patch("rtfm.retrieval.rag.ask", return_value=MOCK_ASK_RESULT), \
             _patches()["get_history_truncated"], _patches()["get_history"], \
             _patches()["add_message"], _patches()["summarize"], \
             _patches()["search_memory"], _patches()["format_memories"], \
             _patches()["store_memory"]:
            result = runner.invoke(
                app, ["chat"], input="What is git?\n/quit\n"
            )

        assert "Git is a version control system" in result.output

    def test_question_saves_to_session(self):
        with patch("rtfm.retrieval.rag.ask", return_value=MOCK_ASK_RESULT), \
             _patches()["get_history_truncated"], \
             patch("rtfm.memory.session.get_history", return_value=[]), \
             patch("rtfm.memory.session.add_message") as mock_add, \
             _patches()["summarize"], \
             _patches()["search_memory"], _patches()["format_memories"], \
             _patches()["store_memory"]:
            runner.invoke(app, ["chat"], input="What is git?\n/quit\n")

        assert mock_add.call_count == 2
        # First call: user message
        assert mock_add.call_args_list[0].args[1] == "user"
        assert mock_add.call_args_list[0].args[2] == "What is git?"
        # Second call: assistant message
        assert mock_add.call_args_list[1].args[1] == "assistant"

    def test_question_triggers_summarize(self):
        with patch("rtfm.retrieval.rag.ask", return_value=MOCK_ASK_RESULT), \
             _patches()["get_history_truncated"], \
             patch("rtfm.memory.session.get_history", return_value=[]), \
             _patches()["add_message"], \
             patch("rtfm.memory.session.summarize_if_needed") as mock_sum, \
             _patches()["search_memory"], _patches()["format_memories"], \
             _patches()["store_memory"]:
            runner.invoke(app, ["chat"], input="What is git?\n/quit\n")

        mock_sum.assert_called_once()

    def test_passes_session_history_and_memories(self):
        history = [{"role": "user", "content": "prev Q"}]
        memories = [{"text": "user likes Go", "topics": ["go"]}]

        with patch("rtfm.retrieval.rag.ask", return_value=MOCK_ASK_RESULT) as mock_ask, \
             patch("rtfm.memory.session.get_history_truncated", return_value=history), \
             patch("rtfm.memory.session.get_history", return_value=[]), \
             _patches()["add_message"], _patches()["summarize"], \
             patch("rtfm.memory.longterm.search_memory", return_value=memories), \
             patch("rtfm.memory.longterm.format_memories", return_value="Remembered context:\n- user likes Go"), \
             _patches()["store_memory"]:
            runner.invoke(app, ["chat"], input="What is git?\n/quit\n")

        call_kwargs = mock_ask.call_args.kwargs
        assert call_kwargs["session_history"] == history
        assert "user likes Go" in call_kwargs["long_term_memories"]

    def test_cached_answer_shows_cache_notice(self):
        cached_result = {**MOCK_ASK_RESULT, "cached": True, "tokens_used": 0}
        with patch("rtfm.retrieval.rag.ask", return_value=cached_result), \
             _patches()["get_history_truncated"], \
             patch("rtfm.memory.session.get_history", return_value=[]), \
             _patches()["add_message"], _patches()["summarize"], \
             _patches()["search_memory"], _patches()["format_memories"], \
             _patches()["store_memory"]:
            result = runner.invoke(app, ["chat"], input="cached Q\n/quit\n")

        assert "cache" in result.output.lower()

    def test_empty_input_ignored(self):
        """Empty lines should not trigger a search."""
        with patch("rtfm.retrieval.rag.ask", return_value=MOCK_ASK_RESULT) as mock_ask, \
             _patches()["get_history_truncated"], \
             patch("rtfm.memory.session.get_history", return_value=[]), \
             _patches()["add_message"], _patches()["summarize"], \
             _patches()["search_memory"], _patches()["format_memories"], \
             _patches()["store_memory"]:
            runner.invoke(app, ["chat"], input="\n\n/quit\n")

        mock_ask.assert_not_called()


class TestChatFilters:
    def test_source_filter_passed_to_rag(self):
        with patch("rtfm.retrieval.rag.ask", return_value=MOCK_ASK_RESULT) as mock_ask, \
             _patches()["get_history_truncated"], \
             patch("rtfm.memory.session.get_history", return_value=[]), \
             _patches()["add_message"], _patches()["summarize"], \
             _patches()["search_memory"], _patches()["format_memories"], \
             _patches()["store_memory"]:
            runner.invoke(
                app, ["chat", "--source", "readme.md"],
                input="Q?\n/quit\n",
            )

        assert mock_ask.call_args.kwargs["source_filter"] == "readme.md"

    def test_section_filter_passed_to_rag(self):
        with patch("rtfm.retrieval.rag.ask", return_value=MOCK_ASK_RESULT) as mock_ask, \
             _patches()["get_history_truncated"], \
             patch("rtfm.memory.session.get_history", return_value=[]), \
             _patches()["add_message"], _patches()["summarize"], \
             _patches()["search_memory"], _patches()["format_memories"], \
             _patches()["store_memory"]:
            runner.invoke(
                app, ["chat", "--section", "Install"],
                input="Q?\n/quit\n",
            )

        assert mock_ask.call_args.kwargs["section_filter"] == "Install"


class TestChatEOF:
    def test_eof_exits_cleanly(self):
        """Ctrl+D (EOF) should exit without error."""
        with _patches()["rag_ask"], _patches()["get_history"], \
             _patches()["search_memory"], _patches()["format_memories"], \
             _patches()["store_memory"]:
            # Empty input simulates EOF
            result = runner.invoke(app, ["chat"], input="")

        assert result.exit_code == 0
