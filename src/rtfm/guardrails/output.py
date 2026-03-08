"""Output filtering guardrails for response safety."""

import re

# PII patterns
_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_PHONE_RE = re.compile(
    r"(?<!\d)"                         # not preceded by a digit
    r"(?:\+?1[-.\s]?)?"                # optional country code
    r"(?:\(?\d{3}\)?[-.\s]?)"          # area code
    r"\d{3}[-.\s]?\d{4}"              # subscriber number
    r"(?!\d)"                          # not followed by a digit
)
_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_API_KEY_RE = re.compile(
    r"(?<![A-Za-z0-9_-])"
    r"(?:"
    r"(?:sk|pk|api|key|token|secret|bearer)[_-][A-Za-z0-9_-]{16,}"
    r"|[A-Za-z0-9]{40,64}"
    r")"
    r"(?![A-Za-z0-9_-])"
)

_PII_RULES: list[tuple[re.Pattern, str]] = [
    (_SSN_RE, "[SSN REDACTED]"),
    (_EMAIL_RE, "[EMAIL REDACTED]"),
    (_PHONE_RE, "[PHONE REDACTED]"),
    (_API_KEY_RE, "[API KEY REDACTED]"),
]


def _redact_pii(text: str) -> str:
    """Replace detected PII patterns with redaction labels."""
    for pattern, replacement in _PII_RULES:
        text = pattern.sub(replacement, text)
    return text


def _truncate(text: str, max_length: int) -> str:
    """Truncate text to *max_length* characters, appending a marker."""
    if len(text) <= max_length:
        return text
    return text[:max_length] + "... [truncated]"


def filter_output(answer: str, settings) -> str:
    """Apply output guardrails to the LLM response.

    - PII redaction (if enabled)
    - Response length cap
    """
    if settings.pii_redaction:
        answer = _redact_pii(answer)

    answer = _truncate(answer, settings.max_response_length)

    return answer
