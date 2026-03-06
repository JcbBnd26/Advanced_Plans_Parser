"""Text embedding extraction via sentence-transformers.

Uses a lightweight sentence transformer (default ``all-MiniLM-L6-v2``,
80 MB, fast CPU inference) to produce 384-d dense embeddings from
block / region text.  These embeddings capture semantic similarity —
e.g. "GENERAL NOTES" ≈ "CONSTRUCTION NOTES" — far better than the
binary ``kw_*`` keyword features they supplement.

The module is **optional** — if ``sentence-transformers`` is not
installed it gracefully degrades: :func:`embed` returns a zero vector
so the downstream pipeline still works.

Public API
----------
TextEmbedder        – Lazy-loading sentence-transformer wrapper
embed               – convenience: text → numpy vector
embed_batch         – convenience: list[text] → list[vector]
is_embeddings_available – returns True if sentence-transformers installed
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence

import numpy as np

log = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────

_DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"
TEXT_EMBEDDING_DIM = 384  # Output dim for all-MiniLM-L6-v2

# ── Availability check ─────────────────────────────────────────────────

_SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    import sentence_transformers  # noqa: F401

    _SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    pass


def is_embeddings_available() -> bool:
    """Return True if sentence-transformers is installed."""
    return _SENTENCE_TRANSFORMERS_AVAILABLE


# ── TextEmbedder ───────────────────────────────────────────────────────


class TextEmbedder:
    """Sentence-transformer text embedder with lazy model loading.

    Parameters
    ----------
    model_name : str
        HuggingFace model name (default ``"all-MiniLM-L6-v2"``).
    device : str or None
        PyTorch device.  ``None`` → auto-detect (CPU by default for
        this lightweight model).
    """

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL_NAME,
        device: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self._model = None
        self._device = device
        self._embedding_dim: int = TEXT_EMBEDDING_DIM

    # ── lazy loading ──────────────────────────────────────────────

    def _ensure_model(self) -> bool:
        """Load the sentence-transformer model on first use.

        Returns False if sentence-transformers is not installed or
        the model cannot be loaded.
        """
        if self._model is not None:
            return True
        if not is_embeddings_available():
            log.debug("sentence-transformers not installed — embeddings disabled")
            return False

        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(
                self.model_name,
                device=self._device,
            )
            actual_dim = self._model.get_sentence_embedding_dimension()
            if actual_dim is not None:
                self._embedding_dim = int(actual_dim)
            log.info(
                "TextEmbedder ready: model=%s, dim=%d",
                self.model_name,
                self._embedding_dim,
            )
            return True
        except Exception:  # noqa: BLE001 — model load may fail for various reasons
            log.warning(
                "Failed to load sentence-transformer '%s'",
                self.model_name,
                exc_info=True,
            )
            self._model = None
            return False

    # ── public properties ─────────────────────────────────────────

    @property
    def available(self) -> bool:
        """True if the model loaded successfully."""
        return self._model is not None

    @property
    def embedding_dim(self) -> int:
        """Dimensionality of the output embedding vector."""
        return self._embedding_dim

    # ── core embedding ────────────────────────────────────────────

    def embed(self, text: str) -> np.ndarray:
        """Compute a dense embedding for a single text string.

        Parameters
        ----------
        text : str
            Input text (block content, region header, etc.).

        Returns
        -------
        numpy.ndarray
            1-D float32 array of shape ``(embedding_dim,)``.
            Returns zeros if the model is unavailable or text is empty.
        """
        if not text or not text.strip():
            return np.zeros(self._embedding_dim, dtype=np.float32)

        if not self._ensure_model():
            return np.zeros(self._embedding_dim, dtype=np.float32)

        try:
            vec = self._model.encode(
                text,
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=True,
            )
            return np.asarray(vec, dtype=np.float32).ravel()
        except Exception:  # noqa: BLE001 — graceful degradation on encode failure
            log.debug("embed() failed for text len=%d", len(text), exc_info=True)
            return np.zeros(self._embedding_dim, dtype=np.float32)

    def embed_batch(self, texts: Sequence[str]) -> list[np.ndarray]:
        """Compute embeddings for a batch of texts.

        Parameters
        ----------
        texts : sequence[str]
            Input texts.

        Returns
        -------
        list[numpy.ndarray]
            List of 1-D float32 arrays, one per input text.
            Zero vectors for empty texts or if model unavailable.
        """
        if not texts:
            return []

        # Identify non-empty texts for batch encoding
        valid_indices: list[int] = []
        valid_texts: list[str] = []
        for i, t in enumerate(texts):
            if t and t.strip():
                valid_indices.append(i)
                valid_texts.append(t)

        # Pre-fill all with zeros
        results = [np.zeros(self._embedding_dim, dtype=np.float32) for _ in texts]

        if not valid_texts or not self._ensure_model():
            return results

        try:
            embeddings = self._model.encode(
                valid_texts,
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=True,
                batch_size=32,
            )
            for j, idx in enumerate(valid_indices):
                results[idx] = np.asarray(embeddings[j], dtype=np.float32).ravel()
        except Exception:  # noqa: BLE001 — graceful degradation on batch encode failure
            log.debug(
                "embed_batch() failed for %d texts", len(valid_texts), exc_info=True
            )

        return results


# ── Module-level singleton ─────────────────────────────────────────────

_embedder: Optional[TextEmbedder] = None


def get_embedder(
    model_name: str = _DEFAULT_MODEL_NAME,
    device: Optional[str] = None,
) -> TextEmbedder:
    """Return (or create) the module-level singleton text embedder."""
    global _embedder
    if _embedder is None or _embedder.model_name != model_name:
        _embedder = TextEmbedder(model_name=model_name, device=device)
    return _embedder


def embed(text: str, model_name: str = _DEFAULT_MODEL_NAME) -> np.ndarray:
    """Convenience: embed a single text string via the singleton."""
    return get_embedder(model_name=model_name).embed(text)


def embed_batch(
    texts: Sequence[str], model_name: str = _DEFAULT_MODEL_NAME
) -> list[np.ndarray]:
    """Convenience: embed a batch of texts via the singleton."""
    return get_embedder(model_name=model_name).embed_batch(texts)
