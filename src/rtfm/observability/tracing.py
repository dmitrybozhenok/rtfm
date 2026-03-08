"""OpenTelemetry tracing for RTFM RAG pipeline."""

from __future__ import annotations

import contextlib
import time
from contextlib import contextmanager
from typing import Any, Generator

from rtfm.config import settings


def _init_tracer():
    """Initialize OpenTelemetry tracer. Returns a no-op tracer if OTel is unavailable."""
    if not getattr(settings, "tracing_enabled", False):
        return None

    try:
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.resources import Resource

        resource = Resource.create({"service.name": "rtfm", "service.version": "0.1.0"})
        provider = TracerProvider(resource=resource)

        # Add exporter if configured
        exporter_url = getattr(settings, "tracing_exporter_url", "")
        if exporter_url:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
            from opentelemetry.sdk.trace.export import BatchSpanProcessor

            exporter = OTLPSpanExporter(endpoint=exporter_url)
            provider.add_span_processor(BatchSpanProcessor(exporter))

        trace.set_tracer_provider(provider)
        return trace.get_tracer("rtfm")
    except ImportError:
        return None


_tracer = None


def get_tracer():
    """Get the global tracer, initializing on first call."""
    global _tracer
    if _tracer is None:
        _tracer = _init_tracer()
    return _tracer


class SpanWrapper:
    """Lightweight span wrapper that works with or without OpenTelemetry."""

    def __init__(self, name: str, attributes: dict[str, Any] | None = None):
        self.name = name
        self.start_time = time.time()
        self._span = None
        self._attributes = attributes or {}

        tracer = get_tracer()
        if tracer is not None:
            self._span = tracer.start_span(name, attributes=self._attributes)

    def set_attribute(self, key: str, value: Any) -> None:
        if self._span is not None:
            self._span.set_attribute(key, value)

    def set_error(self, error: Exception) -> None:
        if self._span is not None:
            self._span.set_attribute("error", True)
            self._span.set_attribute("error.message", str(error))
            self._span.set_attribute("error.type", type(error).__name__)

    def end(self) -> float:
        """End the span and return duration in milliseconds."""
        duration_ms = (time.time() - self.start_time) * 1000
        if self._span is not None:
            self._span.set_attribute("duration_ms", duration_ms)
            self._span.end()
        return duration_ms

    @property
    def duration_ms(self) -> float:
        return (time.time() - self.start_time) * 1000


@contextmanager
def trace_span(name: str, attributes: dict[str, Any] | None = None) -> Generator[SpanWrapper, None, None]:
    """Context manager for tracing a code block.

    Works as a no-op timer when OpenTelemetry is not installed.

    Usage:
        with trace_span("search", {"query": question}) as span:
            results = search(question)
            span.set_attribute("results_count", len(results))
    """
    span = SpanWrapper(name, attributes)
    try:
        yield span
    except Exception as e:
        span.set_error(e)
        raise
    finally:
        span.end()
