"""Structured JSON logging for RTFM."""

import logging
import json
import sys
import time
from typing import Any


class JSONFormatter(logging.Formatter):
    """Format log records as single-line JSON."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry: dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add extra fields passed via `extra={}`
        for key in ("question", "latency_ms", "cached", "chunks_retrieved",
                     "tokens_used", "source_files", "session_id", "model",
                     "llm_model", "embedding_model",
                     "cache_hit", "files_processed", "chunks_created",
                     "guardrail_action", "guardrail_reason", "error",
                     "url", "method", "status_code", "path",
                     "search_type", "reranked", "span_id", "trace_id"):
            val = getattr(record, key, None)
            if val is not None:
                log_entry[key] = val

        if record.exc_info and record.exc_info[0] is not None:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry, default=str)


class TextFormatter(logging.Formatter):
    """Human-readable format for development."""

    def __init__(self):
        super().__init__("%(asctime)s %(levelname)-8s %(name)s — %(message)s")


_configured = False


def setup_logging(log_level: str = "info", log_format: str = "json") -> None:
    """Configure root logging for the application.

    Args:
        log_level: Level name (debug, info, warning, error).
        log_format: "json" for structured output, "text" for human-readable.
    """
    global _configured
    if _configured:
        return
    _configured = True

    level = getattr(logging, log_level.upper(), logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    if log_format == "json":
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(TextFormatter())

    # Configure rtfm loggers
    root_logger = logging.getLogger("rtfm")
    root_logger.setLevel(level)
    root_logger.addHandler(handler)
    root_logger.propagate = False

    # Suppress noisy library loggers
    for name in ("httpx", "httpcore", "uvicorn.access", "openai", "sentence_transformers"):
        logging.getLogger(name).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get a logger under the rtfm namespace."""
    return logging.getLogger(f"rtfm.{name}")
