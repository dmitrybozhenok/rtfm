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

# Models that require trust_remote_code=True
_TRUST_REMOTE_CODE_MODELS = {"nomic-ai/nomic-embed-text-v1.5"}

# Models that use task-type prefixes for asymmetric retrieval
_PREFIXED_MODELS = {"nomic-ai/nomic-embed-text-v1.5"}


def _needs_prefix() -> bool:
    """Check if the current model uses task-type prefixes."""
    return settings.embedding_model in _PREFIXED_MODELS


def get_model() -> SentenceTransformer:
    """Return a cached SentenceTransformer model."""
    global _model
    if _model is None:
        kwargs = {}
        if settings.embedding_model in _TRUST_REMOTE_CODE_MODELS:
            kwargs["trust_remote_code"] = True
        _model = SentenceTransformer(settings.embedding_model, **kwargs)
    return _model


def embed_texts(texts: list[str]) -> list[np.ndarray]:
    """Embed a list of document texts, returning a list of numpy arrays.

    For models that support task-type prefixes (e.g. nomic-embed-text),
    document texts are prefixed with 'search_document: '.
    """
    model = get_model()
    if _needs_prefix():
        texts = [f"search_document: {t}" for t in texts]
    embeddings = model.encode(texts, show_progress_bar=False)
    return [e.astype(np.float32) for e in embeddings]


def embed_query(text: str) -> np.ndarray:
    """Embed a single query text.

    For models that support task-type prefixes (e.g. nomic-embed-text),
    query text is prefixed with 'search_query: '.
    """
    model = get_model()
    if _needs_prefix():
        text = f"search_query: {text}"
    embeddings = model.encode([text], show_progress_bar=False)
    return embeddings[0].astype(np.float32)
