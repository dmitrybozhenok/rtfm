"""Embedding sanity evals — these load the real model."""

import numpy as np
import pytest
from scipy.spatial.distance import cosine

from rtfm.embeddings import embed_query, embed_texts


@pytest.mark.slow
def test_embed_returns_correct_dims():
    """Verify embeddings are 384-dimensional (all-MiniLM-L6-v2)."""
    vec = embed_query("hello world")
    assert vec.shape == (384,)


@pytest.mark.slow
def test_embed_dtype_float32():
    """Verify embedding dtype is float32."""
    vec = embed_query("hello world")
    assert vec.dtype == np.float32


@pytest.mark.slow
def test_embed_semantic_similarity():
    """'car' and 'automobile' should have cosine similarity > 0.5."""
    v1 = embed_query("car")
    v2 = embed_query("automobile")
    similarity = 1 - cosine(v1, v2)
    assert similarity > 0.5, f"Expected similarity > 0.5, got {similarity}"


@pytest.mark.slow
def test_embed_semantic_difference():
    """'car' and 'quantum physics' should have cosine similarity < 0.3."""
    v1 = embed_query("car")
    v2 = embed_query("quantum physics")
    similarity = 1 - cosine(v1, v2)
    assert similarity < 0.3, f"Expected similarity < 0.3, got {similarity}"


@pytest.mark.slow
def test_embed_deterministic():
    """Same text embedded twice gives identical vectors."""
    v1 = embed_query("deterministic test")
    v2 = embed_query("deterministic test")
    np.testing.assert_array_equal(v1, v2)


@pytest.mark.slow
def test_embed_batch_consistency():
    """embed_texts([a, b]) matches [embed_query(a), embed_query(b)]."""
    a = "the quick brown fox"
    b = "jumps over the lazy dog"

    batch = embed_texts([a, b])
    single_a = embed_query(a)
    single_b = embed_query(b)

    np.testing.assert_allclose(batch[0], single_a, atol=1e-6)
    np.testing.assert_allclose(batch[1], single_b, atol=1e-6)
