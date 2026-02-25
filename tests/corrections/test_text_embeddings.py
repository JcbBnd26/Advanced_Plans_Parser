"""Tests for Phase 3.1 — Text Embeddings (sentence-transformer).

Covers:
- Graceful degradation when sentence-transformers is not installed
- TextEmbedder API (embed, embed_batch, zero-vector fallback)
- encode_features() with optional text_embedding
- Config fields (ml_embeddings_enabled, ml_embeddings_model)
- Pipeline integration (text embedder initialisation)

Note: Tests mock sentence-transformers to avoid needing heavy deps in CI.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from plancheck.config import GroupingConfig
from plancheck.corrections.classifier import _NUMERIC_KEYS, ZONE_VALUES, encode_features
from plancheck.corrections.text_embeddings import (
    TEXT_EMBEDDING_DIM,
    TextEmbedder,
    is_embeddings_available,
)

# ── Config fields ──────────────────────────────────────────────────────


class TestEmbeddingsConfig:
    """Config has ml_embeddings_enabled and ml_embeddings_model."""

    def test_defaults(self):
        cfg = GroupingConfig()
        assert cfg.ml_embeddings_enabled is False
        assert cfg.ml_embeddings_model == "all-MiniLM-L6-v2"

    def test_custom_model(self):
        cfg = GroupingConfig(ml_embeddings_model="paraphrase-MiniLM-L3-v2")
        assert cfg.ml_embeddings_model == "paraphrase-MiniLM-L3-v2"


# ── Availability probe ────────────────────────────────────────────────


class TestAvailability:
    """is_embeddings_available() reflects import status."""

    def test_returns_bool(self):
        result = is_embeddings_available()
        assert isinstance(result, bool)


# ── TextEmbedder with mock ────────────────────────────────────────────


class TestTextEmbedder:
    """TextEmbedder API with mocked SentenceTransformer."""

    def _make_mock_model(self):
        mock = MagicMock()
        # encode returns a 2-D array
        mock.encode.return_value = np.random.randn(1, TEXT_EMBEDDING_DIM).astype(
            np.float32
        )
        return mock

    def _make_embedder(self, model=None):
        embedder = TextEmbedder.__new__(TextEmbedder)
        embedder._model = model or self._make_mock_model()
        embedder._embedding_dim = TEXT_EMBEDDING_DIM
        embedder.model_name = "test-model"
        embedder._device = None
        return embedder

    def test_embed_returns_correct_shape(self):
        embedder = self._make_embedder()
        result = embedder.embed("some construction note text")
        assert result.shape == (TEXT_EMBEDDING_DIM,)
        assert result.dtype == np.float32

    def test_embed_empty_string_returns_zeros(self):
        embedder = self._make_embedder()
        result = embedder.embed("")
        assert result.shape == (TEXT_EMBEDDING_DIM,)
        assert np.allclose(result, 0.0)

    def test_embed_batch(self):
        mock_model = MagicMock()
        mock_model.encode.return_value = np.random.randn(3, TEXT_EMBEDDING_DIM).astype(
            np.float32
        )
        embedder = self._make_embedder(mock_model)
        results = embedder.embed_batch(["text1", "text2", "text3"])
        assert len(results) == 3
        for r in results:
            assert r.shape == (TEXT_EMBEDDING_DIM,)

    def test_embed_batch_empty_list(self):
        embedder = self._make_embedder()
        results = embedder.embed_batch([])
        assert results == []

    def test_embed_normalises(self):
        """Embeddings returned by the mock model (with normalize_embeddings=True
        passed through) should be returned as-is by embed(). When using a real
        SentenceTransformer the library normalises internally, so we verify the
        pass-through shape & dtype."""
        mock = MagicMock()
        # Simulate a normalised vector returned by SentenceTransformer
        unit = np.zeros(TEXT_EMBEDDING_DIM, dtype=np.float32)
        unit[0] = 0.6
        unit[1] = 0.8
        mock.encode.return_value = unit
        embedder = self._make_embedder(mock)
        result = embedder.embed("test")
        norm = np.linalg.norm(result)
        assert abs(norm - 1.0) < 1e-5, f"Expected unit norm, got {norm}"


# ── TextEmbedder without model (graceful degradation) ─────────────────


class TestTextEmbedderFallback:
    """When sentence-transformers is not installed, embed returns zeros."""

    def _make_embedder_no_model(self):
        embedder = TextEmbedder.__new__(TextEmbedder)
        embedder._model = None
        embedder._embedding_dim = TEXT_EMBEDDING_DIM
        embedder.model_name = "test-model"
        embedder._device = None
        return embedder

    def test_embed_returns_zeros_when_no_model(self):
        embedder = self._make_embedder_no_model()
        result = embedder.embed("some text")
        assert result.shape == (TEXT_EMBEDDING_DIM,)
        assert np.allclose(result, 0.0)

    def test_embed_batch_returns_zeros_when_no_model(self):
        embedder = self._make_embedder_no_model()
        results = embedder.embed_batch(["a", "b"])
        assert len(results) == 2
        for r in results:
            assert np.allclose(r, 0.0)


# ── encode_features with text embedding ───────────────────────────────


class TestEncodeWithTextEmbedding:
    """encode_features() optionally appends text embedding."""

    def _sample_features(self) -> dict:
        d = {k: 0.0 for k in _NUMERIC_KEYS}
        d["zone"] = "unknown"
        return d

    def test_without_text_embedding(self):
        vec = encode_features(self._sample_features())
        expected_len = len(_NUMERIC_KEYS) + len(ZONE_VALUES)
        assert vec.shape == (expected_len,)

    def test_with_text_embedding(self):
        text_emb = np.random.randn(TEXT_EMBEDDING_DIM).astype(np.float32)
        vec = encode_features(self._sample_features(), text_embedding=text_emb)
        expected_len = len(_NUMERIC_KEYS) + len(ZONE_VALUES) + TEXT_EMBEDDING_DIM
        assert vec.shape == (expected_len,)

    def test_text_embedding_appended_at_end(self):
        text_emb = np.ones(TEXT_EMBEDDING_DIM, dtype=np.float32) * 42.0
        vec = encode_features(self._sample_features(), text_embedding=text_emb)
        # Last TEXT_EMBEDDING_DIM entries should be 42.0
        assert np.allclose(vec[-TEXT_EMBEDDING_DIM:], 42.0)

    def test_with_both_image_and_text(self):
        """Image features + text embedding both appended."""
        from plancheck.corrections.image_features import IMAGE_FEATURE_DIM

        img_feat = np.random.randn(IMAGE_FEATURE_DIM).astype(np.float32)
        text_emb = np.random.randn(TEXT_EMBEDDING_DIM).astype(np.float32)
        vec = encode_features(
            self._sample_features(),
            image_features=img_feat,
            text_embedding=text_emb,
        )
        expected_len = (
            len(_NUMERIC_KEYS)
            + len(ZONE_VALUES)
            + IMAGE_FEATURE_DIM
            + TEXT_EMBEDDING_DIM
        )
        assert vec.shape == (expected_len,)


# ── Constants ─────────────────────────────────────────────────────────


class TestConstants:
    def test_embedding_dim(self):
        assert TEXT_EMBEDDING_DIM == 384

    def test_embedding_dim_positive(self):
        assert TEXT_EMBEDDING_DIM > 0
