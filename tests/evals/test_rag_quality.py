"""RAG quality evaluations.

These tests validate the end-to-end RAG pipeline output quality.
They require Redis (with ingested docs) and Ollama running.
"""

import tempfile
from pathlib import Path

import pytest

from rtfm.ingest.pipeline import ingest_path
from rtfm.retrieval.rag import ask

TEST_DOCUMENT = """\
# Test Knowledge Base

The capital of France is Paris. It is known for the Eiffel Tower.

Python was created by Guido van Rossum and first released in 1991.

The speed of light in a vacuum is approximately 299,792,458 meters per second.
"""


@pytest.fixture(scope="module")
def ingested_test_doc(tmp_path_factory):
    """Create and ingest a small test document with known content."""
    tmp_dir = tmp_path_factory.mktemp("rag_eval_docs")
    doc_path = tmp_dir / "test_knowledge.md"
    doc_path.write_text(TEST_DOCUMENT, encoding="utf-8")

    stats = ingest_path(doc_path)
    assert stats["files"] > 0, "Test document was not ingested"
    assert stats["chunks"] > 0, "No chunks were created from test document"

    return doc_path


@pytest.mark.slow
@pytest.mark.integration
def test_rag_answers_from_context(ingested_test_doc):
    """Question about content in docs should yield an answer with the correct info."""
    result = ask("What is the capital of France?")
    answer = result["answer"].lower()
    assert "paris" in answer, (
        f"Expected 'paris' in answer but got: {result['answer']}"
    )


@pytest.mark.slow
@pytest.mark.integration
def test_rag_cites_sources(ingested_test_doc):
    """Answer should mention the source file name."""
    result = ask("Who created Python?", source_filter="test_knowledge.md")
    answer = result["answer"].lower()
    assert "guido" in answer, (
        f"Expected 'guido' in answer but got: {result['answer']}"
    )
    # Check source appears in sources list
    source_files = [s["file"] for s in result["sources"]]
    assert "test_knowledge.md" in source_files, (
        f"Expected test_knowledge.md in sources but got: {source_files}"
    )


@pytest.mark.slow
@pytest.mark.integration
def test_rag_refuses_unknown(ingested_test_doc):
    """Question NOT in docs should produce a refusal with the expected phrasing."""
    result = ask("What is the population of Neptune's third moon?")
    answer = result["answer"].lower()
    assert "don't have enough information" in answer, (
        f"Expected refusal phrasing in answer but got: {result['answer']}"
    )


@pytest.mark.slow
@pytest.mark.integration
def test_rag_returns_sources(ingested_test_doc):
    """Response dict should have a non-empty sources list for an in-context question."""
    result = ask("What is the speed of light?")
    assert isinstance(result["sources"], list)
    assert len(result["sources"]) > 0, "Expected at least one source"
    first_source = result["sources"][0]
    assert "file" in first_source
    assert "score" in first_source


@pytest.mark.slow
@pytest.mark.integration
def test_rag_response_structure(ingested_test_doc):
    """Response dict should have all expected keys."""
    result = ask("What is the capital of France?")
    expected_keys = {"answer", "sources", "cached", "latency_ms", "tokens_used"}
    assert expected_keys.issubset(result.keys()), (
        f"Missing keys: {expected_keys - result.keys()}"
    )
    assert isinstance(result["answer"], str)
    assert isinstance(result["sources"], list)
    assert isinstance(result["cached"], bool)
    assert isinstance(result["latency_ms"], (int, float))
    assert isinstance(result["tokens_used"], int)
