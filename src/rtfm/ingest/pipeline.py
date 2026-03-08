"""Ingestion pipeline: load → chunk → embed → store in Redis."""

import hashlib
from pathlib import Path

import numpy as np
from redisvl.index import SearchIndex

from rtfm.config import settings
from rtfm.embeddings import embed_texts
from rtfm.ingest.chunker import chunk_text
from rtfm.ingest.loader import discover_files, load_file
from rtfm.redis_client import get_redis, get_redis_binary


SCHEMA_PATH = Path(__file__).parent.parent.parent.parent / "schemas" / "documents.yaml"


def _chunk_id(source_file: str, chunk_index: int) -> str:
    """Generate a deterministic chunk ID for idempotent re-ingestion."""
    raw = f"{source_file}:{chunk_index}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def ensure_index() -> SearchIndex:
    """Create or connect to the Redis vector index."""
    index = SearchIndex.from_yaml(str(SCHEMA_PATH))
    index.connect(settings.redis_url)
    index.create(overwrite=False)
    return index


def ingest_path(path: Path) -> dict:
    """Ingest all supported files under a path. Returns stats."""
    files = discover_files(path)
    if not files:
        return {"files": 0, "chunks": 0}

    index = ensure_index()
    r = get_redis()
    r_bin = get_redis_binary()
    total_chunks = 0

    for file_path in files:
        text, metadata = load_file(file_path)
        chunks = chunk_text(text, metadata)

        if not chunks:
            continue

        # Embed all chunks for this file at once
        texts = [c.text for c in chunks]
        embeddings = embed_texts(texts)

        # Store each chunk in Redis
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

        total_chunks += len(chunks)

    # Flush semantic cache on re-ingestion to avoid stale answers
    try:
        _flush_cache_on_ingest(r)
    except Exception:
        pass  # Cache may not exist yet

    return {"files": len(files), "chunks": total_chunks}


def _flush_cache_on_ingest(r) -> None:
    """Flush semantic cache when documents are re-ingested."""
    from rtfm.cache.semantic_cache import get_cache

    cache = get_cache()
    cache.clear()
