"""Redis vector search for document retrieval."""

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


def search_documents(
    query: str,
    top_k: int | None = None,
    source_filter: str | None = None,
    section_filter: str | None = None,
) -> list[SearchResult]:
    """Search for relevant document chunks using vector similarity.

    Optionally filter by source_file or section tags.
    """
    top_k = top_k or settings.top_k
    query_embedding = embed_query(query)

    index = ensure_index()

    # Build filter expression
    filter_expression = None
    if source_filter:
        filter_expression = Tag("source_file") == source_filter
    if section_filter:
        f = Tag("section") == section_filter
        filter_expression = f if filter_expression is None else filter_expression & f

    vq = VectorQuery(
        vector=query_embedding.tobytes(),
        vector_field_name="embedding",
        return_fields=["text", "source_file", "section"],
        num_results=top_k,
        filter_expression=filter_expression,
    )

    raw_results = index.query(vq)

    results = []
    for doc in raw_results:
        results.append(
            SearchResult(
                text=doc.get("text", ""),
                source_file=doc.get("source_file", ""),
                section=doc.get("section", ""),
                score=float(doc.get("vector_distance", 1.0)),
            )
        )

    return results
