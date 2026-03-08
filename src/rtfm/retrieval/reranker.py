"""Cross-encoder reranker for search results."""

from __future__ import annotations

from sentence_transformers import CrossEncoder

from rtfm.config import settings
from rtfm.retrieval.search import SearchResult

_model: CrossEncoder | None = None


def _get_model() -> CrossEncoder:
    """Lazy-load the cross-encoder model."""
    global _model
    if _model is None:
        _model = CrossEncoder(settings.reranker_model)
    return _model


def rerank(
    query: str,
    results: list[SearchResult],
    top_k: int | None = None,
) -> list[SearchResult]:
    """Rerank search results using a cross-encoder model.

    Scores each (query, result.text) pair and returns the top_k highest-scored results.
    """
    if not results:
        return results

    top_k = top_k or settings.reranker_top_k
    model = _get_model()

    pairs = [[query, r.text] for r in results]
    scores = model.predict(pairs)

    # Attach scores and sort descending
    scored = sorted(zip(results, scores), key=lambda x: x[1], reverse=True)

    return [
        SearchResult(
            text=r.text,
            source_file=r.source_file,
            section=r.section,
            score=float(s),
        )
        for r, s in scored[:top_k]
    ]
