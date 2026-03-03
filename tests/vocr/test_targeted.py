"""Tests for plancheck.vocr.targeted — targeted VOCR patch extraction."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from plancheck.config import GroupingConfig
from plancheck.models import GlyphBox, VocrCandidate
from plancheck.vocr.targeted import _crop_patch, _ocr_crop

# ── _crop_patch ────────────────────────────────────────────────────────


class TestCropPatch:
    @pytest.fixture
    def page_image(self):
        """100×100 white image."""
        return Image.new("RGB", (100, 100), (255, 255, 255))

    def test_valid_crop(self, page_image):
        cand = VocrCandidate(page=0, x0=10, y0=20, x1=50, y1=60)
        arr = _crop_patch(page_image, cand, 100.0, 100.0)
        assert arr is not None
        assert arr.shape[1] == 40  # 50-10
        assert arr.shape[0] == 40  # 60-20

    def test_degenerate_crop_returns_none(self, page_image):
        cand = VocrCandidate(page=0, x0=10, y0=20, x1=12, y1=22)  # 2×2 < 4px threshold
        arr = _crop_patch(page_image, cand, 100.0, 100.0)
        assert arr is None

    def test_scaled_coordinates(self):
        # Image is 200×300 but page is 100×150 → 2× scale
        img = Image.new("RGB", (200, 300), (255, 255, 255))
        cand = VocrCandidate(page=0, x0=10, y0=20, x1=50, y1=70)
        arr = _crop_patch(img, cand, 100.0, 150.0)
        assert arr is not None
        # Crop should be (50-10)*2=80 × (70-20)*2=100
        assert arr.shape[1] == 80
        assert arr.shape[0] == 100

    def test_clamped_to_image_bounds(self, page_image):
        cand = VocrCandidate(page=0, x0=-10, y0=-10, x1=200, y1=200)
        arr = _crop_patch(page_image, cand, 100.0, 100.0)
        assert arr is not None
        assert arr.shape[1] == 100
        assert arr.shape[0] == 100

    def test_grayscale_page_image_returns_rgb_array(self):
        # Regression: grayscale ('L') images produced 2D arrays (H, W)
        # which crash PaddleX text detection.
        img = Image.new("L", (100, 100), 255)
        cand = VocrCandidate(page=0, x0=10, y0=20, x1=50, y1=60)
        arr = _crop_patch(img, cand, 100.0, 100.0)
        assert arr is not None
        assert arr.ndim == 3
        assert arr.shape[2] == 3


# ── _ocr_crop ──────────────────────────────────────────────────────────


class TestOcrCrop:
    def test_extracts_results(self):
        mock_ocr = MagicMock()
        mock_ocr.predict.return_value = [
            {
                "dt_polys": [[[0, 0], [20, 0], [20, 10], [0, 10]]],
                "rec_texts": ["45°"],
                "rec_scores": [0.95],
            }
        ]
        crop = np.zeros((20, 40, 3), dtype=np.uint8)
        results = _ocr_crop(mock_ocr, crop, 0.5)
        args, _kwargs = mock_ocr.predict.call_args
        assert isinstance(args[0], list)
        assert len(args[0]) == 1
        assert len(results) == 1
        text, conf, bbox = results[0]
        assert text == "45°"
        assert conf == 0.95
        assert bbox == [0, 0, 20, 10]

    def test_filters_low_confidence(self):
        mock_ocr = MagicMock()
        mock_ocr.predict.return_value = [
            {
                "dt_polys": [[[0, 0], [20, 0], [20, 10], [0, 10]]],
                "rec_texts": ["noise"],
                "rec_scores": [0.1],
            }
        ]
        crop = np.zeros((20, 40, 3), dtype=np.uint8)
        results = _ocr_crop(mock_ocr, crop, 0.5)
        assert results == []

    def test_filters_empty_text(self):
        mock_ocr = MagicMock()
        mock_ocr.predict.return_value = [
            {
                "dt_polys": [[[0, 0], [20, 0], [20, 10], [0, 10]]],
                "rec_texts": [""],
                "rec_scores": [0.95],
            }
        ]
        crop = np.zeros((20, 40, 3), dtype=np.uint8)
        results = _ocr_crop(mock_ocr, crop, 0.5)
        assert results == []

    def test_empty_prediction(self):
        mock_ocr = MagicMock()
        mock_ocr.predict.return_value = [
            {"dt_polys": None, "rec_texts": None, "rec_scores": None}
        ]
        crop = np.zeros((20, 40, 3), dtype=np.uint8)
        results = _ocr_crop(mock_ocr, crop, 0.5)
        assert results == []


# ── extract_vocr_targeted (integration, mocked OCR) ───────────────────


class TestExtractVocrTargeted:
    @patch("plancheck.vocr.engine._get_ocr")
    def test_hit_outcome(self, mock_get_ocr):
        """Candidate with OCR finding a symbol → outcome='hit'."""
        mock_ocr = MagicMock()
        mock_ocr.predict.return_value = [
            {
                "dt_polys": [[[2, 2], [18, 2], [18, 12], [2, 12]]],
                "rec_texts": ["45°"],
                "rec_scores": [0.92],
            }
        ]
        mock_get_ocr.return_value = mock_ocr

        from plancheck.vocr.targeted import extract_vocr_targeted

        img = Image.new("RGB", (612, 792))
        cand = VocrCandidate(
            page=0,
            x0=100,
            y0=100,
            x1=150,
            y1=130,
            trigger_methods=["template_adjacency"],
            predicted_symbol="°",
        )
        cfg = GroupingConfig()
        tokens, confs, updated = extract_vocr_targeted(
            img, [cand], 0, 612.0, 792.0, cfg
        )
        assert len(tokens) >= 1
        assert tokens[0].origin == "ocr_targeted"
        assert updated[0].outcome == "hit"
        assert "°" in updated[0].found_symbol

    @patch("plancheck.vocr.engine._get_ocr")
    def test_miss_outcome(self, mock_get_ocr):
        """Candidate with no OCR results → outcome='miss'."""
        mock_ocr = MagicMock()
        mock_ocr.predict.return_value = [
            {"dt_polys": [], "rec_texts": [], "rec_scores": []}
        ]
        mock_get_ocr.return_value = mock_ocr

        from plancheck.vocr.targeted import extract_vocr_targeted

        img = Image.new("RGB", (612, 792))
        cand = VocrCandidate(
            page=0,
            x0=100,
            y0=100,
            x1=150,
            y1=130,
            trigger_methods=["intraline_gap"],
        )
        cfg = GroupingConfig()
        _, _, updated = extract_vocr_targeted(img, [cand], 0, 612.0, 792.0, cfg)
        assert updated[0].outcome == "miss"

    @patch("plancheck.vocr.engine._get_ocr")
    def test_empty_candidates(self, mock_get_ocr):
        """No candidates → empty output, OCR never called."""
        from plancheck.vocr.targeted import extract_vocr_targeted

        img = Image.new("RGB", (100, 100))
        cfg = GroupingConfig()
        tokens, confs, updated = extract_vocr_targeted(img, [], 0, 100.0, 100.0, cfg)
        assert tokens == []
        assert confs == []
        assert updated == []
        mock_get_ocr.assert_not_called()

    @patch("plancheck.vocr.engine._get_ocr")
    def test_degenerate_patch_is_miss(self, mock_get_ocr):
        """Tiny candidate whose crop is degenerate → miss."""
        mock_get_ocr.return_value = MagicMock()

        from plancheck.vocr.targeted import extract_vocr_targeted

        img = Image.new("RGB", (612, 792))
        cand = VocrCandidate(page=0, x0=100, y0=100, x1=101, y1=101)
        cfg = GroupingConfig()
        _, _, updated = extract_vocr_targeted(img, [cand], 0, 612.0, 792.0, cfg)
        assert updated[0].outcome == "miss"
