"""Session conversation history stored in Redis lists with TTL."""

import json

from rtfm.config import settings
from rtfm.redis_client import get_redis

SESSION_PREFIX = "rtfm:session:"
MAX_HISTORY_TOKENS = 4000  # Approximate token budget for session history


def _session_key(session_id: str) -> str:
    return f"{SESSION_PREFIX}{session_id}"


def add_message(session_id: str, role: str, content: str) -> None:
    """Add a message to the session history."""
    r = get_redis()
    key = _session_key(session_id)
    msg = json.dumps({"role": role, "content": content})
    r.rpush(key, msg)
    r.expire(key, settings.session_ttl)


def get_history(session_id: str) -> list[dict]:
    """Get the full conversation history for a session."""
    r = get_redis()
    key = _session_key(session_id)
    raw = r.lrange(key, 0, -1)
    return [json.loads(m) for m in raw]


def get_history_truncated(session_id: str, max_chars: int | None = None) -> list[dict]:
    """Get conversation history, truncating oldest messages if too long.

    Keeps the most recent messages that fit within the token budget.
    """
    max_chars = max_chars or MAX_HISTORY_TOKENS * 4  # ~4 chars per token
    history = get_history(session_id)

    if not history:
        return []

    # Check if summarization is needed
    total_chars = sum(len(m["content"]) for m in history)
    if total_chars <= max_chars:
        return history

    # Keep most recent messages that fit
    kept = []
    used = 0
    for msg in reversed(history):
        msg_len = len(msg["content"])
        if used + msg_len > max_chars:
            break
        kept.insert(0, msg)
        used += msg_len

    # Ensure we always have at least the last message
    if not kept and history:
        kept = [history[-1]]

    return kept


def clear_session(session_id: str) -> None:
    """Clear all messages for a session."""
    r = get_redis()
    r.delete(_session_key(session_id))


def summarize_if_needed(session_id: str) -> None:
    """Summarize conversation history if it exceeds the token threshold.

    Replaces history with a summary message + recent messages.
    """
    history = get_history(session_id)
    total_chars = sum(len(m["content"]) for m in history)

    if total_chars <= MAX_HISTORY_TOKENS * 4:
        return

    # Keep last 4 messages, summarize the rest
    to_summarize = history[:-4]
    to_keep = history[-4:]

    if not to_summarize:
        return

    # Build a simple extractive summary
    summary_parts = []
    for msg in to_summarize:
        role = msg["role"]
        content = msg["content"][:200]
        summary_parts.append(f"{role}: {content}")

    summary = "Previous conversation summary:\n" + "\n".join(summary_parts)

    # Replace history
    r = get_redis()
    key = _session_key(session_id)
    r.delete(key)

    add_message(session_id, "user", summary)
    add_message(session_id, "assistant", "Understood, I have the context from our earlier conversation.")
    for msg in to_keep:
        add_message(session_id, msg["role"], msg["content"])
