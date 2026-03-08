"""Long-term memory via Redis Agent Memory Server."""

import httpx

from rtfm.config import settings


def _base_url() -> str:
    return settings.memory_server_url.rstrip("/")


def store_memory(
    session_id: str,
    messages: list[dict],
    namespace: str = "rtfm",
) -> dict | None:
    """Store conversation messages in long-term memory.

    The Memory Server automatically extracts topics and entities.
    """
    try:
        resp = httpx.post(
            f"{_base_url()}/sessions/{session_id}/memory",
            json={"messages": messages, "namespace": namespace},
            timeout=10.0,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None


def search_memory(
    query: str,
    namespace: str = "rtfm",
    limit: int = 5,
) -> list[dict]:
    """Search long-term memories relevant to a query."""
    try:
        resp = httpx.post(
            f"{_base_url()}/long_term_memory/search",
            json={"query": query, "namespace": namespace, "limit": limit},
            timeout=10.0,
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("memories", [])
    except Exception:
        return []


def format_memories(memories: list[dict]) -> str:
    """Format memories into a context string for the LLM prompt."""
    if not memories:
        return ""

    parts = []
    for m in memories:
        text = m.get("text", m.get("content", ""))
        topics = m.get("topics", [])
        topic_str = f" (topics: {', '.join(topics)})" if topics else ""
        parts.append(f"- {text}{topic_str}")

    return "Remembered context:\n" + "\n".join(parts)
