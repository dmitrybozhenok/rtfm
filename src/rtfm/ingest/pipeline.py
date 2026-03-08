"""Ingestion pipeline: load → chunk → embed → store in Redis."""

import hashlib
from pathlib import Path

import numpy as np
from redisvl.index import SearchIndex

from rtfm.config import settings
from rtfm.embeddings import embed_texts
from rtfm.ingest.chunker import chunk_text
from rtfm.ingest.loader import discover_files, load_file
from rtfm.observability.logging import get_logger
from rtfm.redis_client import get_redis, get_redis_binary

logger = get_logger("ingest")


SCHEMA_PATH = Path(__file__).parent.parent.parent.parent / "schemas" / "documents.yaml"


def _chunk_id(source_file: str, chunk_index: int) -> str:
    """Generate a deterministic chunk ID for idempotent re-ingestion."""
    raw = f"{source_file}:{chunk_index}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def ensure_index() -> SearchIndex:
    """Create or connect to the Redis vector index.

    If the existing index has a different vector dimension than the schema
    (e.g. after an embedding model change), the index is dropped and recreated.
    """
    index = SearchIndex.from_yaml(str(SCHEMA_PATH))
    index.connect(settings.redis_url)
    try:
        index.create(overwrite=False)
    except Exception:
        # Index may exist with incompatible schema — recreate it
        try:
            index.delete(drop=True)
        except Exception:
            pass
        index.create(overwrite=True)
    return index


def _store_chunks(chunks, metadata, r_bin) -> int:
    """Embed and store chunks in Redis. Returns count stored."""
    if not chunks:
        return 0

    texts = [c.text for c in chunks]
    embeddings = embed_texts(texts)

    for chunk, embedding in zip(chunks, embeddings):
        key = f"doc:{_chunk_id(metadata['source_file'], chunk.chunk_index)}"
        doc = {
            "text": chunk.text,
            "source_file": metadata["source_file"],
            "section": chunk.section,
            "chunk_index": chunk.chunk_index,
            "embedding": embedding.tobytes(),
        }
        r_bin.hset(key, mapping=doc)

    return len(chunks)


def ingest_path(path: Path) -> dict:
    """Ingest all supported files under a path. Returns stats."""
    files = discover_files(path)
    if not files:
        return {"files": 0, "chunks": 0}

    ensure_index()
    r = get_redis()
    r_bin = get_redis_binary()
    total_chunks = 0

    for file_path in files:
        text, metadata = load_file(file_path)
        chunks = chunk_text(text, metadata)
        stored = _store_chunks(chunks, metadata, r_bin)
        total_chunks += stored
        logger.debug("File ingested", extra={"path": str(file_path), "chunks_created": stored})

    # Flush semantic cache on re-ingestion to avoid stale answers
    try:
        _flush_cache_on_ingest(r)
    except Exception:
        pass  # Cache may not exist yet

    logger.info("Path ingestion complete",
                extra={"files_processed": len(files), "chunks_created": total_chunks})
    return {"files": len(files), "chunks": total_chunks}


def _flush_cache_on_ingest(r) -> None:
    """Flush semantic cache when documents are re-ingested."""
    from rtfm.cache.semantic_cache import get_cache

    cache = get_cache()
    cache.clear()
