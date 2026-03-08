"""Unit tests for rtfm.retrieval.search module.

Mocks Redis index, embeddings, and pipeline to test search logic directly.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from rtfm.retrieval.search import SearchResult, search_documents


@pytest.fixture(autouse=True)
def mock_embed():
    """Mock embed_query to return a fixed embedding."""
    with patch("rtfm.retrieval.search.embed_query") as m:
        m.return_value = np.zeros(384, dtype=np.float32)
        yield m


@pytest.fixture(autouse=True)
def mock_index():
    """Mock ensure_index to return a fake SearchIndex."""
    idx = MagicMock()
    idx.query.return_value = []
    with patch("rtfm.retrieval.search.ensure_index", return_value=idx) as m:
        yield idx


def _raw_doc(text="chunk text", source_file="doc.md", section="Intro", distance="0.15"):
    return {
        "text": text,
        "source_file": source_file,
        "section": section,
        "vector_distance": distance,
    }


# ── Basic search behavior ────────────────────────────────────


class TestSearchDocuments:
    def test_returns_empty_list_for_no_results(self, mock_index):
        mock_index.query.return_value = []

        results = search_documents("test query", hybrid=False)

        assert results == []

    def test_returns_search_results(self, mock_index):
        mock_index.query.return_value = [
            _raw_doc("chunk 1", "a.md", "Setup", "0.10"),
            _raw_doc("chunk 2", "b.md", "Install", "0.20"),
        ]

        results = search_documents("how to install", hybrid=False)

        assert len(results) == 2
        assert isinstance(results[0], SearchResult)
        assert results[0].text == "chunk 1"
        assert results[0].source_file == "a.md"
        assert results[0].section == "Setup"
        assert results[0].score == 0.10
        assert results[1].score == 0.20

    def test_uses_default_top_k(self, mock_index):
        from rtfm.config import settings

        search_documents("query", hybrid=False)

        vq = mock_index.query.call_args.args[0]
        assert vq._num_results == settings.top_k

    def test_custom_top_k(self, mock_index):
        search_documents("query", top_k=3, hybrid=False)

        vq = mock_index.query.call_args.args[0]
        assert vq._num_results == 3

    def test_embeds_the_query(self, mock_embed, mock_index):
        search_documents("how to configure redis", hybrid=False)

        mock_embed.assert_called_once_with("how to configure redis")

    def test_missing_fields_default_gracefully(self, mock_index):
        mock_index.query.return_value = [{}]

        results = search_documents("query", hybrid=False)

        assert len(results) == 1
        assert results[0].text == ""
        assert results[0].source_file == ""
        assert results[0].section == ""
        assert results[0].score == 1.0  # Default distance


# ── Filter tests ─────────────────────────────────────────────


class TestSearchFilters:
    def test_no_filters_by_default(self, mock_index):
        search_documents("query", hybrid=False)

        vq = mock_index.query.call_args.args[0]
        assert vq.params.get("filter_expression") is None or vq.filter is None

    def test_source_filter_applied(self, mock_index):
        search_documents("query", source_filter="readme.md", hybrid=False)

        vq = mock_index.query.call_args.args[0]
        # VectorQuery stores filter_expression; verify it was set
        assert vq.filter is not None

    def test_section_filter_applied(self, mock_index):
        search_documents("query", section_filter="Install", hybrid=False)

        vq = mock_index.query.call_args.args[0]
        assert vq.filter is not None

    def test_both_filters_combined(self, mock_index):
        search_documents("query", source_filter="doc.md", section_filter="Setup", hybrid=False)

        vq = mock_index.query.call_args.args[0]
        assert vq.filter is not None

    def test_none_source_filter_uses_wildcard(self, mock_index):
        """None source_filter results in wildcard (match-all) filter."""
        search_documents("query", source_filter=None, hybrid=False)

        vq = mock_index.query.call_args.args[0]
        assert str(vq._filter_expression) == "*"

    def test_none_section_filter_uses_wildcard(self, mock_index):
        """None section_filter results in wildcard (match-all) filter."""
        search_documents("query", section_filter=None, hybrid=False)

        vq = mock_index.query.call_args.args[0]
        assert str(vq._filter_expression) == "*"
