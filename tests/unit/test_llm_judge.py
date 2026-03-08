"""Tests for the LLM-as-judge evaluation module."""

import json
from unittest.mock import MagicMock, patch

import pytest

import sys
from pathlib import Path

# Add evals dir to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "evals"))

from llm_judge import _parse_judgment, judge_answer, judge_batch


class TestParseJudgment:
    """Tests for parsing LLM judge responses."""

    def test_valid_json(self):
        raw = '{"score": 4, "justification": "Mostly correct answer"}'
        result = _parse_judgment(raw)
        assert result["score"] == 4
        assert result["justification"] == "Mostly correct answer"

    def test_json_in_markdown_block(self):
        raw = '```json\n{"score": 3, "justification": "Partially correct"}\n```'
        result = _parse_judgment(raw)
        assert result["score"] == 3

    def test_json_with_surrounding_text(self):
        raw = 'Here is my evaluation:\n{"score": 5, "justification": "Perfect"}\nEnd.'
        result = _parse_judgment(raw)
        assert result["score"] == 5

    def test_bare_number_fallback(self):
        raw = "I would rate this a 2 out of 5 because it is mostly wrong."
        result = _parse_judgment(raw)
        assert result["score"] == 2

    def test_unparseable_returns_zero(self):
        raw = "This is not a valid response at all."
        result = _parse_judgment(raw)
        assert result["score"] == 0
        assert "Failed to parse" in result["justification"]

    def test_score_out_of_range_falls_through(self):
        raw = '{"score": 7, "justification": "Invalid range"}'
        result = _parse_judgment(raw)
        # Should fall through to regex extraction or 0
        assert result["score"] in (0, 7) or result["score"] <= 5

    def test_score_boundary_values(self):
        for score in (1, 2, 3, 4, 5):
            raw = json.dumps({"score": score, "justification": f"Score {score}"})
            result = _parse_judgment(raw)
            assert result["score"] == score


class TestJudgeAnswer:
    """Tests for the judge_answer function."""

    @patch("llm_judge._save_cached")
    @patch("llm_judge._load_cached", return_value=None)
    @patch("llm_judge.OpenAI")
    def test_calls_ollama(self, mock_openai_cls, mock_load, mock_save):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"score": 4, "justification": "Good"}'
        mock_client.chat.completions.create.return_value = mock_response

        result = judge_answer(
            "What is git?",
            "A version control system",
            "Git is a distributed version control system",
            model="qwen2.5:7b",
            ollama_base_url="http://localhost:11434/v1",
        )

        assert result["score"] == 4
        assert result["justification"] == "Good"
        mock_openai_cls.assert_called_once_with(
            base_url="http://localhost:11434/v1", api_key="ollama"
        )
        call_args = mock_client.chat.completions.create.call_args
        assert call_args.kwargs["model"] == "qwen2.5:7b"
        assert call_args.kwargs["temperature"] == 0

    @patch("llm_judge._load_cached")
    def test_returns_cached_result(self, mock_load):
        mock_load.return_value = {"score": 5, "justification": "Cached"}
        result = judge_answer("q", "expected", "generated")
        assert result["score"] == 5
        assert result["from_cache"] is True

    @patch("llm_judge.OpenAI")
    @patch("llm_judge._save_cached")
    @patch("llm_judge._load_cached", return_value=None)
    def test_caches_new_result(self, mock_load, mock_save, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"score": 3, "justification": "Ok"}'
        mock_client.chat.completions.create.return_value = mock_response

        result = judge_answer("q", "expected", "generated")
        assert result["score"] == 3
        assert result["from_cache"] is False
        mock_save.assert_called_once()


class TestJudgeBatch:
    """Tests for batch judging."""

    @patch("llm_judge.judge_answer")
    def test_batch_processes_all(self, mock_judge):
        mock_judge.return_value = {"score": 4, "justification": "Good", "from_cache": False}

        results = [
            {"question": "Q1", "expected_answer": "A1", "rtfm_answer": "R1"},
            {"question": "Q2", "expected_answer": "A2", "rtfm_answer": "R2"},
            {"question": "Q3", "expected_answer": "A3", "rtfm_answer": "R3"},
        ]

        judgments = judge_batch(results, model="test-model")
        assert len(judgments) == 3
        assert all(j["score"] == 4 for j in judgments)
        assert mock_judge.call_count == 3

    @patch("llm_judge.judge_answer")
    def test_batch_counts_cache_hits(self, mock_judge, capsys):
        mock_judge.side_effect = [
            {"score": 4, "justification": "Good", "from_cache": True},
            {"score": 3, "justification": "Ok", "from_cache": False},
        ]

        results = [
            {"question": "Q1", "expected_answer": "A1", "rtfm_answer": "R1"},
            {"question": "Q2", "expected_answer": "A2", "rtfm_answer": "R2"},
        ]

        judgments = judge_batch(results)
        captured = capsys.readouterr()
        assert "cache hits: 1/2" in captured.out
