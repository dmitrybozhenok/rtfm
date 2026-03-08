"""Cache behavior tests — require a running Redis instance."""

import pytest

from redisvl.extensions.cache.llm import SemanticCache

from rtfm.config import settings
from rtfm.redis_client import get_redis

# Use a unique cache name so tests don't collide with the app cache.
TEST_CACHE_NAME = "rtfm-test-cache"


@pytest.fixture(autouse=True)
def cache():
    """Create a fresh SemanticCache for each test and tear it down after."""
    c = SemanticCache(
        name=TEST_CACHE_NAME,
        redis_url=settings.redis_url,
        distance_threshold=settings.cache_distance_threshold,
        ttl=settings.cache_ttl,
    )
    c.clear()
    yield c
    c.clear()


def _reset_metrics():
    """Zero out the Redis metric counters used by get_cache_metrics."""
    r = get_redis()
    for key in (
        "rtfm:metrics:cache_hits",
        "rtfm:metrics:cache_misses",
        "rtfm:metrics:total_latency_ms",
        "rtfm:metrics:total_queries",
        "rtfm:metrics:total_tokens",
    ):
        r.delete(key)


# ---------- tests ----------


@pytest.mark.integration
def test_cache_store_and_retrieve(cache: SemanticCache):
    """Store a Q&A pair and retrieve it by the same question."""
    cache.store(prompt="What is Python?", response="A programming language.")
    results = cache.check(prompt="What is Python?")
    assert results is not None
    assert len(results) > 0
    assert results[0]["response"] == "A programming language."


@pytest.mark.integration
def test_cache_miss_unrelated(cache: SemanticCache):
    """An unrelated query should not match a cached entry."""
    cache.store(prompt="What is Python?", response="A programming language.")
    results = cache.check(prompt="How do black holes form?")
    # Either no results or empty list counts as a miss.
    assert not results


@pytest.mark.integration
def test_cache_flush(cache: SemanticCache):
    """After flush, previously cached entries should return None."""
    cache.store(prompt="What is Python?", response="A programming language.")
    cache.clear()
    results = cache.check(prompt="What is Python?")
    assert not results


@pytest.mark.integration
def test_cache_metrics_structure():
    """get_cache_metrics() returns all expected keys."""
    from rtfm.cache.semantic_cache import get_cache_metrics

    metrics = get_cache_metrics()
    expected_keys = {
        "cache_hits",
        "cache_misses",
        "hit_rate_pct",
        "total_queries",
        "avg_latency_ms",
        "total_tokens",
        "estimated_cost_usd",
        "estimated_savings_usd",
    }
    assert set(metrics.keys()) == expected_keys


@pytest.mark.integration
def test_cache_metrics_after_queries():
    """After incrementing metric counters, get_cache_metrics reflects them."""
    from rtfm.cache.semantic_cache import get_cache_metrics

    _reset_metrics()

    r = get_redis()
    # Simulate 3 hits and 2 misses
    r.set("rtfm:metrics:cache_hits", 3)
    r.set("rtfm:metrics:cache_misses", 2)
    r.set("rtfm:metrics:total_queries", 5)
    r.set("rtfm:metrics:total_latency_ms", 250)
    r.set("rtfm:metrics:total_tokens", 1000)

    metrics = get_cache_metrics()
    assert metrics["cache_hits"] == 3
    assert metrics["cache_misses"] == 2
    assert metrics["hit_rate_pct"] == 60.0
    assert metrics["total_queries"] == 5
    assert metrics["avg_latency_ms"] == 50.0
    assert metrics["total_tokens"] == 1000
    assert metrics["estimated_cost_usd"] == 0.01
    assert metrics["estimated_savings_usd"] == 0.03

    # Clean up
    _reset_metrics()
