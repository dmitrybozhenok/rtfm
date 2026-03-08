"""Shared Redis connection factory."""

import redis

from rtfm.config import settings

_client: redis.Redis | None = None


def get_redis() -> redis.Redis:
    """Return a shared Redis client instance."""
    global _client
    if _client is None:
        _client = redis.from_url(settings.redis_url, decode_responses=True)
    return _client


def get_redis_binary() -> redis.Redis:
    """Return a Redis client that does NOT decode responses (for binary data like vectors)."""
    return redis.from_url(settings.redis_url, decode_responses=False)
