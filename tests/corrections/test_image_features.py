"""Tests for Phase 2.1 — Image Feature Extraction (CNN / timm).

Covers:
- Graceful degradation when torch/timm are not installed
- ImageFeatureExtractor API (zero-vector fallback for degenerate crops)
- encode_features() with optional image features
- Config fields (ml_vision_enabled, ml_vision_backbone)

Note: The tests mock torch/timm to avoid needing heavy deps in CI.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from plancheck.config import GroupingConfig
from plancheck.corrections.classifier import _NUMERIC_KEYS, ZONE_VALUES, encode_features
from plancheck.corrections.image_features import (
    IMAGE_FEATURE_DIM,
    ImageFeatureExtractor,
    is_vision_available,
)

# ── Config fields ──────────────────────────────────────────────────────


class TestVisionConfig:
    """Config has ml_vision_enabled and ml_vision_backbone."""

    def test_defaults(self):
        cfg = GroupingConfig()
        assert cfg.ml_vision_enabled is False
        assert cfg.ml_vision_backbone == "resnet18"

    def test_custom_backbone(self):
        cfg = GroupingConfig(ml_vision_backbone="efficientnet_b0")
        assert cfg.ml_vision_backbone == "efficientnet_b0"


# ── encode_features with image features ────────────────────────────────


class TestEncodeWithImageFeatures:
    """encode_features() optionally appends CNN embedding."""

    def _sample_features(self) -> dict:
        d = {k: 0.0 for k in _NUMERIC_KEYS}
        d["zone"] = "unknown"
        return d

    def test_without_image_features(self):
        vec = encode_features(self._sample_features())
        expected_len = len(_NUMERIC_KEYS) + len(ZONE_VALUES)
        assert vec.shape == (expected_len,)

    def test_with_image_features(self):
        img_feat = np.random.randn(IMAGE_FEATURE_DIM).astype(np.float32)
        vec = encode_features(self._sample_features(), image_features=img_feat)
        expected_len = len(_NUMERIC_KEYS) + len(ZONE_VALUES) + IMAGE_FEATURE_DIM
        assert vec.shape == (expected_len,)

    def test_image_features_appended_at_end(self):
        img_feat = np.ones(IMAGE_FEATURE_DIM, dtype=np.float32) * 42.0
        vec = encode_features(self._sample_features(), image_features=img_feat)
        base_len = len(_NUMERIC_KEYS) + len(ZONE_VALUES)
        assert np.all(vec[base_len:] == 42.0)

    def test_none_image_features_is_base_only(self):
        vec = encode_features(self._sample_features(), image_features=None)
        base_len = len(_NUMERIC_KEYS) + len(ZONE_VALUES)
        assert vec.shape == (base_len,)

    def test_empty_image_features_is_base_only(self):
        vec = encode_features(
            self._sample_features(),
            image_features=np.array([], dtype=np.float32),
        )
        base_len = len(_NUMERIC_KEYS) + len(ZONE_VALUES)
        assert vec.shape == (base_len,)


# ── ImageFeatureExtractor fallback ─────────────────────────────────────


class TestImageFeatureExtractorFallback:
    """Extractor returns zero vec when torch/timm are not available."""

    def test_returns_zero_when_unavailable(self):
        extractor = ImageFeatureExtractor()
        # Force unavailable by patching _ensure_model to return False
        extractor._ensure_model = lambda: False
        from PIL import Image

        img = Image.new("RGB", (100, 100), "white")
        result = extractor.extract(img, (10, 10, 50, 50))
        assert result.shape == (IMAGE_FEATURE_DIM,)
        assert np.all(result == 0.0)

    def test_degenerate_bbox_returns_zero(self):
        extractor = ImageFeatureExtractor()
        # Pretend model is loaded — degenerate bbox should short-circuit
        # before any torch import.
        extractor._ensure_model = lambda: True
        from PIL import Image

        img = Image.new("RGB", (100, 100), "white")
        # x1 == x0 → zero width
        result = extractor.extract(img, (50, 50, 50, 80))
        assert result.shape == (IMAGE_FEATURE_DIM,)
        assert np.all(result == 0.0)

    def test_embedding_dim_property(self):
        extractor = ImageFeatureExtractor()
        assert extractor.embedding_dim == IMAGE_FEATURE_DIM

    def test_available_false_when_no_model(self):
        extractor = ImageFeatureExtractor()
        assert extractor.available is False


class TestImageFeatureExtractorMocked:
    """Test extractor with a mocked timm model."""

    def _make_extractor(self):
        """Build an extractor with a fake model that returns predictable output."""
        import torch

        extractor = ImageFeatureExtractor()
        # Create a mock model that returns a fixed embedding
        fake_embedding = torch.ones(1, IMAGE_FEATURE_DIM) * 0.5
        mock_model = MagicMock(return_value=fake_embedding)
        mock_model.to = MagicMock(return_value=mock_model)
        mock_model.eval = MagicMock()
        extractor._model = mock_model
        extractor._device = torch.device("cpu")

        # Simple transform: convert PIL → tensor (mock)
        def fake_transform(pil_img):
            return torch.zeros(3, 224, 224)

        extractor._transform = fake_transform
        return extractor

    @pytest.fixture(autouse=True)
    def _skip_if_no_torch(self):
        pytest.importorskip("torch")

    def test_extract_returns_correct_shape(self):
        from PIL import Image

        extractor = self._make_extractor()
        img = Image.new("RGB", (200, 200), "white")
        result = extractor.extract(img, (10, 10, 100, 100))
        assert result.shape == (IMAGE_FEATURE_DIM,)
        assert result.dtype == np.float32

    def test_extract_batch(self):
        import torch
        from PIL import Image

        extractor = self._make_extractor()
        # Make batch model return stacked embeddings
        batch_emb = torch.ones(3, IMAGE_FEATURE_DIM) * 0.5
        extractor._model.return_value = batch_emb

        img = Image.new("RGB", (200, 200), "white")
        bboxes = [(10, 10, 50, 50), (60, 60, 100, 100), (110, 110, 190, 190)]
        results = extractor.extract_batch(img, bboxes)
        assert len(results) == 3
        for r in results:
            assert r.shape == (IMAGE_FEATURE_DIM,)

    def test_extract_batch_skips_degenerate(self):
        import torch
        from PIL import Image

        extractor = self._make_extractor()
        # Only 1 valid bbox → model called with batch of 1
        batch_emb = torch.ones(1, IMAGE_FEATURE_DIM) * 0.7
        extractor._model.return_value = batch_emb

        img = Image.new("RGB", (200, 200), "white")
        bboxes = [(50, 50, 50, 80), (10, 10, 50, 50)]  # first is degenerate
        results = extractor.extract_batch(img, bboxes)
        assert len(results) == 2
        assert np.all(results[0] == 0.0)  # degenerate → zeros
        assert results[1].shape == (IMAGE_FEATURE_DIM,)


# ── is_vision_available() ─────────────────────────────────────────────


class TestIsVisionAvailable:
    """Check the availability probe."""

    def test_returns_bool(self):
        result = is_vision_available()
        assert isinstance(result, bool)
