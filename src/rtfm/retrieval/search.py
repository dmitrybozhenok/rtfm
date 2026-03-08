"""Redis vector search for document retrieval."""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from redisvl.index import SearchIndex
from redisvl.query import VectorQuery
from redisvl.query.filter import Tag

from rtfm.config import settings
from rtfm.embeddings import embed_query
from rtfm.ingest.pipeline import ensure_index


@dataclass
class SearchResult:
    text: str
    source_file: str
    section: str
    score: float
    source_url: str = ""


# Characters that are special in RediSearch query syntax
_REDISEARCH_SPECIAL = re.compile(r"([{}()\[\]@!\"~*\\<>/,.:;+\-=|&^$?#])")


def _escape_redisearch(text: str) -> str:
    """Escape special characters for RediSearch full-text queries."""
    return _REDISEARCH_SPECIAL.sub(r"\\\1", text)


def _bm25_search(query: str, num_results: int = 20) -> list[SearchResult]:
    """Run BM25 full-text search using Redis FT.SEARCH directly."""
    from rtfm.redis_client import get_redis

    r = get_redis()

    # Build query: escape special chars, join with spaces (OR matching)
    words = query.strip().split()
    escaped_words = [_escape_redisearch(w) for w in words if w.strip()]
    if not escaped_words:
        return []

    query_string = " ".join(escaped_words)

    try:
        raw = r.execute_command(
            "FT.SEARCH",
            "rtfm-docs",
            query_string,
            "LIMIT", "0", str(num_results),
            "RETURN", "4", "text", "source_file", "source_url", "section",
        )
    except Exception:
        return []

    # Parse FT.SEARCH response: [total_count, key1, [field, val, ...], key2, ...]
    if not raw or raw[0] == 0:
        return []

    results = []
    i = 1
    while i < len(raw):
        doc_key = raw[i]
        fields = raw[i + 1] if i + 1 < len(raw) else []
        i += 2

        field_dict = {}
        for j in range(0, len(fields), 2):
            field_dict[fields[j]] = fields[j + 1]

        results.append(
            SearchResult(
                text=field_dict.get("text", ""),
                source_file=field_dict.get("source_file", ""),
                section=field_dict.get("section", ""),
                score=0.0,  # BM25 score not used directly; RRF rank matters
                source_url=field_dict.get("source_url", ""),
            )
        )

    return results


def _reciprocal_rank_fusion(
    vector_results: list[SearchResult],
    bm25_results: list[SearchResult],
    k: int = 60,
) -> list[SearchResult]:
    """Combine two ranked lists using Reciprocal Rank Fusion (RRF)."""
    vector_weight = 1.0 - settings.bm25_weight
    bm25_weight = settings.bm25_weight

    # Use (source_file, text[:100]) as a unique key for deduplication
    def _doc_key(r: SearchResult) -> str:
        return f"{r.source_file}::{r.text[:100]}"

    scores: dict[str, float] = defaultdict(float)
    doc_map: dict[str, SearchResult] = {}

    for rank, r in enumerate(vector_results):
        key = _doc_key(r)
        scores[key] += vector_weight * (1.0 / (k + rank + 1))
        doc_map[key] = r

    for rank, r in enumerate(bm25_results):
        key = _doc_key(r)
        scores[key] += bm25_weight * (1.0 / (k + rank + 1))
        if key not in doc_map:
            doc_map[key] = r

    # Sort by RRF score descending
    sorted_keys = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

    return [
        SearchResult(
            text=doc_map[key].text,
            source_file=doc_map[key].source_file,
            section=doc_map[key].section,
            score=scores[key],
            source_url=doc_map[key].source_url,
        )
        for key in sorted_keys
    ]


def search_documents(
    query: str,
    top_k: int | None = None,
    source_filter: str | None = None,
    section_filter: str | None = None,
    source_url_filter: str | None = None,
    source_type_filter: str | None = None,
    hybrid: bool = True,
) -> list[SearchResult]:
    """Search for relevant document chunks.

    When hybrid=True, combines vector search + BM25 via RRF, then reranks.
    When hybrid=False, uses vector-only search (original behavior).

    Filters:
        source_filter: Exact match on source_file tag
        source_url_filter: Exact match on source_url tag
        source_type_filter: "file" or "web" — match on source_type tag
        section_filter: Exact match on section tag
    """
    top_k = top_k or settings.top_k
    index = ensure_index()

    # Build filter expression
    filter_expression = None
    if source_filter:
        filter_expression = Tag("source_file") == source_filter
    if source_url_filter:
        f = Tag("source_url") == source_url_filter
        filter_expression = f if filter_expression is None else filter_expression & f
    if source_type_filter:
        f = Tag("source_type") == source_type_filter
        filter_expression = f if filter_expression is None else filter_expression & f
    if section_filter:
        f = Tag("section") == section_filter
        filter_expression = f if filter_expression is None else filter_expression & f

    if not hybrid:
        # Vector-only search (original behavior)
        query_embedding = embed_query(query)
        vq = VectorQuery(
            vector=query_embedding.tobytes(),
            vector_field_name="embedding",
            return_fields=["text", "source_file", "source_url", "section"],
            num_results=top_k,
            filter_expression=filter_expression,
        )
        raw_results = index.query(vq)

        return [
            SearchResult(
                text=doc.get("text", ""),
                source_file=doc.get("source_file", ""),
                section=doc.get("section", ""),
                score=float(doc.get("vector_distance", 1.0)),
                source_url=doc.get("source_url", ""),
            )
            for doc in raw_results
        ]

    # --- Hybrid search ---
    # 1. Vector search for top-20 candidates
    query_embedding = embed_query(query)
    vq = VectorQuery(
        vector=query_embedding.tobytes(),
        vector_field_name="embedding",
        return_fields=["text", "source_file", "source_url", "section"],
        num_results=20,
        filter_expression=filter_expression,
    )
    vector_results = [
        SearchResult(
            text=doc.get("text", ""),
            source_file=doc.get("source_file", ""),
            section=doc.get("section", ""),
            score=float(doc.get("vector_distance", 1.0)),
            source_url=doc.get("source_url", ""),
        )
        for doc in index.query(vq)
    ]

    # 2. BM25 full-text search for top-20
    bm25_results = _bm25_search(query, num_results=20)

    # 3. Reciprocal Rank Fusion
    fused = _reciprocal_rank_fusion(vector_results, bm25_results)

    # 4. Rerank top-20 fused results
    from rtfm.retrieval.reranker import rerank

    top_fused = fused[:20]
    reranked = rerank(query, top_fused, top_k=top_k)

    return reranked
