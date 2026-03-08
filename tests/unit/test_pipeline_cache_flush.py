"""Tests for cache flush on re-ingestion (Phase 3 gap).

Mocks Redis and cache to verify flush behavior during ingestion.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture()
def mock_ingest_deps():
    """Mock all ingestion dependencies: loader, chunker, embedder, redis, cache."""
    from rtfm.ingest.chunker import Chunk

    chunks = [Chunk(text="test chunk", section="Intro", chunk_index=0, metadata={"source_file": "doc.md"})]

    with patch("rtfm.ingest.pipeline.discover_files") as mock_discover, \
         patch("rtfm.ingest.pipeline.load_file") as mock_load, \
         patch("rtfm.ingest.pipeline.chunk_text") as mock_chunk, \
         patch("rtfm.ingest.pipeline.embed_texts") as mock_embed, \
         patch("rtfm.ingest.pipeline.ensure_index") as mock_index, \
         patch("rtfm.ingest.pipeline.get_redis") as mock_redis, \
         patch("rtfm.ingest.pipeline.get_redis_binary") as mock_redis_bin:

        mock_discover.return_value = [Path("doc.md")]
        mock_load.return_value = ("text content", {"source_file": "doc.md"})
        mock_chunk.return_value = chunks
        mock_embed.return_value = [MagicMock(tobytes=lambda: b"\x00" * 384)]
        mock_index.return_value = MagicMock()
        mock_redis.return_value = MagicMock()
        mock_redis_bin.return_value = MagicMock()

        yield {
            "discover": mock_discover,
            "load": mock_load,
            "chunk": mock_chunk,
            "embed": mock_embed,
            "redis": mock_redis,
            "redis_bin": mock_redis_bin,
        }


class TestCacheFlushOnIngestion:
    def test_flush_called_after_ingestion(self, mock_ingest_deps):
        """Cache should be flushed after documents are ingested."""
        from rtfm.ingest.pipeline import ingest_path

        mock_cache = MagicMock()
        with patch("rtfm.cache.semantic_cache.get_cache", return_value=mock_cache):
            ingest_path(Path("docs/"))

        mock_cache.clear.assert_called_once()

    def test_ingestion_succeeds_when_cache_flush_fails(self, mock_ingest_deps):
        """If cache flush raises, ingestion should still return stats."""
        from rtfm.ingest.pipeline import ingest_path

        with patch(
            "rtfm.ingest.pipeline._flush_cache_on_ingest",
            side_effect=Exception("cache not initialized"),
        ):
            stats = ingest_path(Path("docs/"))

        assert stats["files"] == 1
        assert stats["chunks"] == 1

    def test_flush_not_called_when_no_files(self):
        """If no files found, cache flush should not be attempted."""
        from rtfm.ingest.pipeline import ingest_path

        with patch("rtfm.ingest.pipeline.discover_files", return_value=[]), \
             patch("rtfm.ingest.pipeline.ensure_index"), \
             patch("rtfm.ingest.pipeline.get_redis"), \
             patch("rtfm.ingest.pipeline.get_redis_binary"), \
             patch("rtfm.ingest.pipeline._flush_cache_on_ingest") as mock_flush:
            stats = ingest_path(Path("empty/"))

        assert stats["files"] == 0
        mock_flush.assert_not_called()

    def test_flush_cache_on_ingest_calls_cache_clear(self):
        """_flush_cache_on_ingest should call get_cache().clear()."""
        from rtfm.ingest.pipeline import _flush_cache_on_ingest

        mock_cache = MagicMock()
        mock_redis = MagicMock()

        with patch("rtfm.cache.semantic_cache.get_cache", return_value=mock_cache):
            _flush_cache_on_ingest(mock_redis)

        mock_cache.clear.assert_called_once()
