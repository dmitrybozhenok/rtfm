"""Input validation guardrails for query safety."""

import re

import numpy as np

# Regex patterns for common prompt injection attempts
_INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(all\s+)?(previous|prior|above)\s+(instructions|prompts|rules)", re.IGNORECASE),
    re.compile(r"disregard\s+(all\s+)?(previous|prior|above)\s+(instructions|prompts|rules)", re.IGNORECASE),
    re.compile(r"forget\s+(all\s+)?(previous|prior|above)\s+(instructions|prompts|rules)", re.IGNORECASE),
    re.compile(r"(reveal|show|print|output|display)\s+(me\s+)?(your\s+)?(system\s+prompt|instructions|rules)", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+", re.IGNORECASE),
    re.compile(r"act\s+as\s+(a\s+|an\s+)?(?!.*(documentation|doc|search))", re.IGNORECASE),
    re.compile(r"pretend\s+(you\s+are|to\s+be)\s+", re.IGNORECASE),
    re.compile(r"new\s+(system\s+)?instructions?:", re.IGNORECASE),
    re.compile(r"\bsystem\s*:\s*", re.IGNORECASE),
    re.compile(r"<\s*/?system\s*>", re.IGNORECASE),
    re.compile(r"\]\s*\]\s*\]\s*", re.IGNORECASE),  # bracket injection
    re.compile(r"jailbreak", re.IGNORECASE),
    re.compile(r"DAN\s+mode", re.IGNORECASE),
    re.compile(r"developer\s+mode\s+(enabled|on|activate)", re.IGNORECASE),
]


def _check_prompt_injection(query: str) -> str | None:
    """Return a reason string if prompt injection is detected, else None."""
    for pattern in _INJECTION_PATTERNS:
        if pattern.search(query):
            return "Query blocked: potential prompt injection detected."
    return None


def _check_query_length(query: str, max_length: int) -> str | None:
    """Return a reason string if query exceeds max length, else None."""
    if len(query) > max_length:
        return f"Query blocked: exceeds maximum length of {max_length} characters ({len(query)} provided)."
    return None


def _check_off_topic(query: str, threshold: float) -> str | None:
    """Return a reason string if query is off-topic (low similarity to all docs).

    Uses a quick vector search against the document index; if the best match
    score is below *threshold*, the query is considered off-topic.
    """
    try:
        from rtfm.retrieval.search import search_documents

        results = search_documents(query)
        if not results:
            return None  # No documents ingested — skip this check
        best_score = max(r.score for r in results)
        if best_score < threshold:
            return (
                f"Query blocked: your question does not appear to be related to "
                f"any ingested documentation (best relevance: {best_score:.2f})."
            )
    except Exception:
        pass  # Graceful degradation — don't block if check fails
    return None


def validate_query(query: str, settings) -> tuple[bool, str]:
    """Validate a user query against guardrail rules.

    Returns:
        (is_valid, reason) — *is_valid* is True when the query should proceed.
        When False, *reason* explains why the query was blocked.
    """
    # 1. Length check
    reason = _check_query_length(query, settings.max_query_length)
    if reason:
        return False, reason

    # 2. Prompt injection check
    reason = _check_prompt_injection(query)
    if reason:
        return False, reason

    # 3. Off-topic check
    reason = _check_off_topic(query, settings.offtopic_threshold)
    if reason:
        return False, reason

    return True, ""
