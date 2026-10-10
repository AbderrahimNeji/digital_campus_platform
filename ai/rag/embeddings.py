"""Embedding generation. Every vector is 768 dimensions, matching ``vector(768)``.

``EMBEDDING_PROVIDER`` chooses the implementation. It is never inferred from
the presence of an API key.

* ``local`` (default) — ``sentence-transformers/paraphrase-multilingual-mpnet-base-v2``.
  PyTorch is imported only when this provider is used, so the hosted API does
  not need it.
* ``gemini`` — Gemini ``gemini-embedding-001`` truncated to 768 dimensions and
  L2-normalized. Google only normalizes the full 3072-dimension output, and
  pgvector cosine search expects unit vectors.
"""

from __future__ import annotations

import logging
import math
import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
GEMINI_EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIM = 768
LOCAL_PROVIDER = "local"
GEMINI_PROVIDER = "gemini"
# Stay under the free-tier tokens-per-minute cap when many chunks are embedded.
GEMINI_BATCH_SIZE = 16

_AI_ENV = Path(__file__).resolve().parents[1] / ".env"
_BACKEND_ENV = Path(__file__).resolve().parents[2] / "backend" / ".env"
_PLACEHOLDERS = {"your_gemini_api_key_here", "your_mistral_api_key_here", ""}


def embedding_provider() -> str:
    """Return ``local`` or ``gemini``. Process environment wins over env files."""
    load_dotenv(_BACKEND_ENV, override=False)
    load_dotenv(_AI_ENV, override=False)
    provider = os.getenv("EMBEDDING_PROVIDER", LOCAL_PROVIDER).strip().lower()
    if provider in ("", LOCAL_PROVIDER):
        return LOCAL_PROVIDER
    if provider == GEMINI_PROVIDER:
        return GEMINI_PROVIDER
    raise ValueError(
        f"Unknown EMBEDDING_PROVIDER '{provider}'. Use 'local' or 'gemini'."
    )


def embed_texts(texts: list[str], batch_size: int = 32, show_progress: bool = False) -> list[list[float]]:
    """Embed document texts. Returns one unit-length 768-d vector per text."""
    if not texts:
        return []
    provider = embedding_provider()
    if provider == GEMINI_PROVIDER:
        return _embed_gemini(texts, "RETRIEVAL_DOCUMENT")
    return _embed_local(texts, batch_size=batch_size, show_progress=show_progress)


@lru_cache(maxsize=512)
def embed_query(query: str) -> list[float]:
    """Embed one query. Cached so the warmup and retries skip repeat work.

    The returned list is shared; callers treat it as read-only.
    """
    if embedding_provider() == GEMINI_PROVIDER:
        return _embed_gemini([query], "RETRIEVAL_QUERY")[0]
    return _embed_local([query], batch_size=1, show_progress=False)[0]


def _embed_local(texts: list[str], batch_size: int, show_progress: bool) -> list[list[float]]:
    model = _get_model()
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return vectors.tolist()


@lru_cache(maxsize=1)
def _get_model():
    # Imported here so production, which never selects "local", does not need torch.
    from sentence_transformers import SentenceTransformer

    logger.info("Loading embedding model: %s", MODEL_NAME)
    model = SentenceTransformer(MODEL_NAME)
    logger.info("Embedding model ready (dim=%d)", EMBEDDING_DIM)
    return model


def _gemini_api_key() -> str:
    key = os.getenv("GEMINI_API_KEY", "")
    if not key or key in _PLACEHOLDERS:
        raise ValueError(
            "GEMINI_API_KEY is not configured. Embeddings stay on the local model "
            "unless EMBEDDING_PROVIDER=gemini and a real key are both set."
        )
    return key


@lru_cache(maxsize=1)
def _gemini_client():
    from google import genai

    return genai.Client(api_key=_gemini_api_key())


def _embed_gemini(texts: list[str], task_type: str) -> list[list[float]]:
    _gemini_api_key()
    from google.genai import types

    model = os.getenv("GEMINI_EMBEDDING_MODEL", GEMINI_EMBEDDING_MODEL)
    client = _gemini_client()
    vectors: list[list[float]] = []
    for start in range(0, len(texts), GEMINI_BATCH_SIZE):
        batch = texts[start:start + GEMINI_BATCH_SIZE]
        result = client.models.embed_content(
            model=model,
            contents=batch,
            config=types.EmbedContentConfig(
                output_dimensionality=EMBEDDING_DIM,
                task_type=task_type,
            ),
        )
        embeddings = list(result.embeddings or [])
        if len(embeddings) != len(batch):
            raise RuntimeError(
                f"Gemini returned {len(embeddings)} embeddings for {len(batch)} texts."
            )
        for embedding in embeddings:
            values = [float(value) for value in (embedding.values or [])]
            if len(values) != EMBEDDING_DIM:
                raise RuntimeError(
                    f"Expected {EMBEDDING_DIM} dimensions to match vector(768), got {len(values)}."
                )
            vectors.append(_l2_normalize(values))
    return vectors


def _l2_normalize(values: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in values))
    if norm == 0:
        raise RuntimeError("Gemini returned a zero embedding.")
    return [value / norm for value in values]
