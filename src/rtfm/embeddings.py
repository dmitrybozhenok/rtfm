"""Sentence-transformers embedding wrapper."""

import logging
import warnings

import numpy as np

# Suppress noisy position_ids warnings from transformers
warnings.filterwarnings("ignore", message=".*position_ids.*")
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

from sentence_transformers import SentenceTransformer

from rtfm.config import settings

_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    """Return a cached SentenceTransformer model."""
    global _model
    if _model is None:
        _model = SentenceTransformer(settings.embedding_model)
    return _model


def embed_texts(texts: list[str]) -> list[np.ndarray]:
    """Embed a list of texts, returning a list of numpy arrays."""
    model = get_model()
    embeddings = model.encode(texts, show_progress_bar=False)
    return [e.astype(np.float32) for e in embeddings]


def embed_query(text: str) -> np.ndarray:
    """Embed a single query text."""
    return embed_texts([text])[0]
