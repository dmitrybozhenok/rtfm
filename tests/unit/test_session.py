"""Unit tests for rtfm.memory.session module.

Uses fakeredis to avoid requiring a running Redis instance.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from rtfm.memory.session import (
    MAX_HISTORY_TOKENS,
    SESSION_PREFIX,
    _session_key,
    add_message,
    clear_session,
    get_history,
    get_history_truncated,
    summarize_if_needed,
)


@pytest.fixture()
def fake_redis():
    """Provide a dict-backed fake Redis for session tests."""

    class FakeRedis:
        def __init__(self):
            self.data: dict[str, list[str]] = {}
            self.ttls: dict[str, int] = {}

        def rpush(self, key, value):
            self.data.setdefault(key, []).append(value)

        def lrange(self, key, start, stop):
            lst = self.data.get(key, [])
            if stop == -1:
                return lst[start:]
            return lst[start : stop + 1]

        def expire(self, key, ttl):
            self.ttls[key] = ttl

        def delete(self, key):
            self.data.pop(key, None)
            self.ttls.pop(key, None)

    r = FakeRedis()
    with patch("rtfm.memory.session.get_redis", return_value=r):
        yield r


class TestSessionKey:
    def test_session_key_format(self):
        assert _session_key("abc") == f"{SESSION_PREFIX}abc"

    def test_session_key_with_uuid(self):
        sid = "550e8400-e29b-41d4-a716-446655440000"
        assert _session_key(sid) == f"{SESSION_PREFIX}{sid}"


class TestAddMessage:
    def test_adds_message_to_redis(self, fake_redis):
        add_message("s1", "user", "hello")

        key = _session_key("s1")
        assert len(fake_redis.data[key]) == 1
        msg = json.loads(fake_redis.data[key][0])
        assert msg == {"role": "user", "content": "hello"}

    def test_appends_multiple_messages(self, fake_redis):
        add_message("s1", "user", "hello")
        add_message("s1", "assistant", "hi there")
        add_message("s1", "user", "how are you?")

        key = _session_key("s1")
        assert len(fake_redis.data[key]) == 3

    def test_sets_ttl(self, fake_redis):
        add_message("s1", "user", "hello")

        key = _session_key("s1")
        from rtfm.config import settings

        assert fake_redis.ttls[key] == settings.session_ttl

    def test_separate_sessions_isolated(self, fake_redis):
        add_message("s1", "user", "hello from s1")
        add_message("s2", "user", "hello from s2")

        key1 = _session_key("s1")
        key2 = _session_key("s2")
        assert len(fake_redis.data[key1]) == 1
        assert len(fake_redis.data[key2]) == 1
        assert "s1" in json.loads(fake_redis.data[key1][0])["content"]
        assert "s2" in json.loads(fake_redis.data[key2][0])["content"]


class TestGetHistory:
    def test_returns_empty_for_nonexistent_session(self, fake_redis):
        assert get_history("nonexistent") == []

    def test_returns_messages_in_order(self, fake_redis):
        add_message("s1", "user", "first")
        add_message("s1", "assistant", "second")
        add_message("s1", "user", "third")

        history = get_history("s1")
        assert len(history) == 3
        assert history[0] == {"role": "user", "content": "first"}
        assert history[1] == {"role": "assistant", "content": "second"}
        assert history[2] == {"role": "user", "content": "third"}

    def test_returns_dicts_with_role_and_content(self, fake_redis):
        add_message("s1", "user", "test")
        history = get_history("s1")
        assert set(history[0].keys()) == {"role", "content"}


class TestGetHistoryTruncated:
    def test_returns_all_when_under_budget(self, fake_redis):
        add_message("s1", "user", "short")
        add_message("s1", "assistant", "reply")

        result = get_history_truncated("s1")
        assert len(result) == 2

    def test_returns_empty_for_nonexistent(self, fake_redis):
        assert get_history_truncated("nonexistent") == []

    def test_truncates_oldest_messages(self, fake_redis):
        # max_chars=40 means only ~40 chars of content fit
        add_message("s1", "user", "A" * 30)  # old, should be dropped
        add_message("s1", "assistant", "B" * 30)  # old, should be dropped
        add_message("s1", "user", "C" * 15)  # recent, fits
        add_message("s1", "assistant", "D" * 15)  # recent, fits

        result = get_history_truncated("s1", max_chars=40)
        assert len(result) == 2
        assert result[0]["content"] == "C" * 15
        assert result[1]["content"] == "D" * 15

    def test_always_keeps_last_message(self, fake_redis):
        # Even if the last message alone exceeds max_chars
        add_message("s1", "user", "X" * 100)

        result = get_history_truncated("s1", max_chars=10)
        assert len(result) == 1
        assert result[0]["content"] == "X" * 100

    def test_uses_default_max_chars(self, fake_redis):
        add_message("s1", "user", "hello")
        result = get_history_truncated("s1")
        # Under budget, should return all
        assert len(result) == 1


class TestClearSession:
    def test_clears_existing_session(self, fake_redis):
        add_message("s1", "user", "hello")
        assert get_history("s1") != []

        clear_session("s1")
        assert get_history("s1") == []

    def test_clearing_nonexistent_session_is_safe(self, fake_redis):
        clear_session("nonexistent")  # Should not raise


class TestSummarizeIfNeeded:
    def test_no_op_when_under_threshold(self, fake_redis):
        add_message("s1", "user", "short question")
        add_message("s1", "assistant", "short answer")

        summarize_if_needed("s1")
        history = get_history("s1")
        assert len(history) == 2  # Unchanged

    def test_summarizes_when_over_threshold(self, fake_redis):
        # Exceed MAX_HISTORY_TOKENS * 4 chars total
        big_content = "X" * 3000
        for i in range(8):
            add_message("s1", "user", f"Q{i}: {big_content}")
            add_message("s1", "assistant", f"A{i}: {big_content}")

        original_count = len(get_history("s1"))
        assert original_count == 16

        summarize_if_needed("s1")
        history = get_history("s1")

        # Should be fewer messages now: summary + ack + last 4
        assert len(history) < original_count
        # Last 4 original messages preserved
        assert len(history) >= 6  # 2 summary + 4 kept

    def test_summary_preserves_recent_messages(self, fake_redis):
        big_content = "Y" * 3000
        for i in range(8):
            add_message("s1", "user", f"Q{i}: {big_content}")
            add_message("s1", "assistant", f"A{i}: {big_content}")

        summarize_if_needed("s1")
        history = get_history("s1")

        # The last 4 messages from original should be preserved
        assert history[-1]["content"].startswith("A7:")
        assert history[-2]["content"].startswith("Q7:")

    def test_summary_message_content(self, fake_redis):
        big_content = "Z" * 3000
        for i in range(8):
            add_message("s1", "user", f"Q{i}: {big_content}")
            add_message("s1", "assistant", f"A{i}: {big_content}")

        summarize_if_needed("s1")
        history = get_history("s1")

        # First message should be the summary
        assert "Previous conversation summary:" in history[0]["content"]
        assert history[0]["role"] == "user"
        # Second message should be acknowledgment
        assert "context" in history[1]["content"].lower()
        assert history[1]["role"] == "assistant"
