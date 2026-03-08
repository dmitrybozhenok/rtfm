"""Ingestion pipeline: load → chunk → embed → store in Redis."""

import hashlib
import time
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


def _store_chunks(chunks, metadata, r_bin) -> int:
    """Embed and store chunks in Redis. Returns count stored."""
    if not chunks:
        return 0

    texts = [c.text for c in chunks]
    embeddings = embed_texts(texts)

    source_url = metadata.get("source_url", "")
    source_type = "web" if source_url else "file"

    for chunk, embedding in zip(chunks, embeddings):
        key = f"doc:{_chunk_id(metadata['source_file'], chunk.chunk_index)}"
        doc = {
            "text": chunk.text,
            "source_file": metadata["source_file"],
            "source_url": source_url,
            "source_type": source_type,
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
        total_chunks += _store_chunks(chunks, metadata, r_bin)

    # Flush semantic cache on re-ingestion to avoid stale answers
    try:
        _flush_cache_on_ingest(r)
    except Exception:
        pass  # Cache may not exist yet

    return {"files": len(files), "chunks": total_chunks}


def ingest_url(url: str, recursive: bool = False, delay: float = 1.0) -> dict:
    """Ingest content from a URL. Returns stats.

    If recursive=True, crawls all linked pages under the same base URL.
    """
    from rtfm.ingest.web_loader import discover_urls, load_url

    if recursive:
        urls = discover_urls(url, delay=delay)
        # Include the base URL itself if it has content
        urls = [url] + urls
    else:
        urls = [url]

    ensure_index()
    r = get_redis()
    r_bin = get_redis_binary()
    total_chunks = 0
    pages_ingested = 0

    for i, page_url in enumerate(urls):
        try:
            text, metadata = load_url(page_url)
            if not text.strip():
                continue

            chunks = chunk_text(text, metadata)
            total_chunks += _store_chunks(chunks, metadata, r_bin)
            pages_ingested += 1

            # Rate limit between requests (skip delay for last page)
            if recursive and i < len(urls) - 1:
                time.sleep(delay)

        except Exception as e:
            # Log but continue with other pages
            import sys
            print(f"Warning: Failed to ingest {page_url}: {e}", file=sys.stderr)
            continue

    # Flush semantic cache on re-ingestion
    try:
        _flush_cache_on_ingest(r)
    except Exception:
        pass

    return {"pages": pages_ingested, "chunks": total_chunks}


def _flush_cache_on_ingest(r) -> None:
    """Flush semantic cache when documents are re-ingested."""
    from rtfm.cache.semantic_cache import get_cache

    cache = get_cache()
    cache.clear()
