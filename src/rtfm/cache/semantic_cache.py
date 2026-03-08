"""Semantic cache using RedisVL SemanticCache."""

import json

from redisvl.extensions.cache.llm import SemanticCache

from rtfm.config import settings

_cache: SemanticCache | None = None


def get_cache() -> SemanticCache:
    """Return a shared SemanticCache instance."""
    global _cache
    if _cache is None:
        _cache = SemanticCache(
            name="rtfm-cache",
            redis_url=settings.redis_url,
            distance_threshold=settings.cache_distance_threshold,
            ttl=settings.cache_ttl,
        )
    return _cache


def check_cache(question: str) -> tuple[str | None, list[dict]]:
    """Check if a similar question has a cached answer.

    Returns (answer, sources) tuple. If no hit, returns (None, []).
    """
    cache = get_cache()
    results = cache.check(prompt=question)
    if results:
        answer = results[0]["response"]
        sources: list[dict] = []
        metadata = results[0].get("metadata", {})
        if metadata and "sources" in metadata:
            try:
                sources = json.loads(metadata["sources"])
            except (json.JSONDecodeError, TypeError):
                sources = []
        return answer, sources
    return None, []


def store_cache(question: str, answer: str, sources: list[dict] | None = None) -> None:
    """Store a question-answer pair in the semantic cache."""
    cache = get_cache()
    cache.store(
        prompt=question,
        response=answer,
        metadata={"sources": json.dumps(sources or [])},
    )


def flush_cache() -> None:
    """Flush all cached entries."""
    cache = get_cache()
    cache.clear()


def get_cache_metrics() -> dict:
    """Get cache performance metrics from Redis counters."""
    from rtfm.redis_client import get_redis

    r = get_redis()
    hits = int(r.get("rtfm:metrics:cache_hits") or 0)
    misses = int(r.get("rtfm:metrics:cache_misses") or 0)
    total = hits + misses
    total_latency = float(r.get("rtfm:metrics:total_latency_ms") or 0)
    total_queries = int(r.get("rtfm:metrics:total_queries") or 0)
    total_tokens = int(r.get("rtfm:metrics:total_tokens") or 0)

    hit_rate = (hits / total * 100) if total > 0 else 0
    avg_latency = (total_latency / total_queries) if total_queries > 0 else 0

    # Rough cost estimate: ~$3 per 1M input tokens, ~$15 per 1M output tokens
    # Simplified: ~$0.01 per 1K tokens average
    estimated_cost = total_tokens / 1000 * 0.01
    estimated_savings = hits * 0.01  # Each cache hit saves ~$0.01

    return {
        "cache_hits": hits,
        "cache_misses": misses,
        "hit_rate_pct": round(hit_rate, 1),
        "total_queries": total_queries,
        "avg_latency_ms": round(avg_latency, 1),
        "total_tokens": total_tokens,
        "estimated_cost_usd": round(estimated_cost, 4),
        "estimated_savings_usd": round(estimated_savings, 4),
    }
