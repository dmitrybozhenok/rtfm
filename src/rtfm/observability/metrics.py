"""Prometheus-compatible metrics for RTFM."""

from __future__ import annotations

import time
import threading
from typing import Any


class Histogram:
    """Thread-safe histogram for tracking distributions."""

    def __init__(self, name: str, description: str, buckets: tuple[float, ...] | None = None):
        self.name = name
        self.description = description
        self.buckets = buckets or (5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000, float("inf"))
        self._counts = [0] * len(self.buckets)
        self._sum = 0.0
        self._count = 0
        self._lock = threading.Lock()

    def observe(self, value: float) -> None:
        with self._lock:
            self._sum += value
            self._count += 1
            for i, bound in enumerate(self.buckets):
                if value <= bound:
                    self._counts[i] += 1

    def to_prometheus(self) -> str:
        lines = [f"# HELP {self.name} {self.description}", f"# TYPE {self.name} histogram"]
        with self._lock:
            cumulative = 0
            for i, bound in enumerate(self.buckets):
                cumulative += self._counts[i]
                le = "+Inf" if bound == float("inf") else str(bound)
                lines.append(f'{self.name}_bucket{{le="{le}"}} {cumulative}')
            lines.append(f"{self.name}_sum {self._sum}")
            lines.append(f"{self.name}_count {self._count}")
        return "\n".join(lines)


class Counter:
    """Thread-safe counter."""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self._value = 0
        self._lock = threading.Lock()

    def inc(self, amount: int = 1) -> None:
        with self._lock:
            self._value += amount

    @property
    def value(self) -> int:
        return self._value

    def to_prometheus(self) -> str:
        return "\n".join([
            f"# HELP {self.name} {self.description}",
            f"# TYPE {self.name} counter",
            f"{self.name}_total {self._value}",
        ])


class Gauge:
    """Thread-safe gauge."""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self._value: float = 0
        self._lock = threading.Lock()

    def set(self, value: float) -> None:
        with self._lock:
            self._value = value

    def inc(self, amount: float = 1) -> None:
        with self._lock:
            self._value += amount

    def dec(self, amount: float = 1) -> None:
        with self._lock:
            self._value -= amount

    @property
    def value(self) -> float:
        return self._value

    def to_prometheus(self) -> str:
        return "\n".join([
            f"# HELP {self.name} {self.description}",
            f"# TYPE {self.name} gauge",
            f"{self.name} {self._value}",
        ])


class MetricsRegistry:
    """Central registry for all application metrics."""

    def __init__(self):
        # Counters
        self.queries_total = Counter("rtfm_queries_total", "Total number of queries processed")
        self.cache_hits = Counter("rtfm_cache_hits_total", "Total cache hits")
        self.cache_misses = Counter("rtfm_cache_misses_total", "Total cache misses")
        self.guardrail_blocks = Counter("rtfm_guardrail_blocks_total", "Total queries blocked by guardrails")
        self.errors = Counter("rtfm_errors_total", "Total errors")
        self.ingestion_files = Counter("rtfm_ingestion_files_total", "Total files ingested")
        self.ingestion_chunks = Counter("rtfm_ingestion_chunks_total", "Total chunks created")

        # Histograms
        self.query_latency = Histogram(
            "rtfm_query_latency_ms", "Query latency in milliseconds",
            buckets=(10, 50, 100, 250, 500, 1000, 2500, 5000, 10000, 30000, float("inf")),
        )
        self.search_latency = Histogram(
            "rtfm_search_latency_ms", "Search latency in milliseconds",
            buckets=(5, 10, 25, 50, 100, 250, 500, 1000, float("inf")),
        )
        self.llm_latency = Histogram(
            "rtfm_llm_latency_ms", "LLM call latency in milliseconds",
            buckets=(100, 500, 1000, 2500, 5000, 10000, 30000, float("inf")),
        )
        self.tokens_per_query = Histogram(
            "rtfm_tokens_per_query", "Tokens used per query",
            buckets=(50, 100, 250, 500, 1000, 2000, 4000, float("inf")),
        )
        self.chunks_retrieved = Histogram(
            "rtfm_chunks_retrieved", "Number of chunks retrieved per query",
            buckets=(1, 2, 3, 5, 10, 15, 20, float("inf")),
        )

        # Gauges
        self.active_sessions = Gauge("rtfm_active_sessions", "Number of active sessions")
        self.index_size = Gauge("rtfm_index_size_chunks", "Total chunks in the index")

        self._all = [
            self.queries_total, self.cache_hits, self.cache_misses,
            self.guardrail_blocks, self.errors, self.ingestion_files,
            self.ingestion_chunks,
            self.query_latency, self.search_latency, self.llm_latency,
            self.tokens_per_query, self.chunks_retrieved,
            self.active_sessions, self.index_size,
        ]

    def to_prometheus(self) -> str:
        """Render all metrics in Prometheus exposition format."""
        return "\n\n".join(m.to_prometheus() for m in self._all) + "\n"


# Global metrics instance
metrics = MetricsRegistry()
