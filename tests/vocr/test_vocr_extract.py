"""Tests for plancheck.vocr.extract — VOCR token extraction and tiling."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from plancheck.config import GroupingConfig
from plancheck.models import GlyphBox
from plancheck.vocr.extract import _dedup_tiles, _iou, _ocr_one_tile

# ── IoU helper ─────────────────────────────────────────────────────────


class TestIou:
    def test_identical_boxes(self):
        a = GlyphBox(page=0, x0=0, y0=0, x1=10, y1=10)
        b = GlyphBox(page=0, x0=0, y0=0, x1=10, y1=10)
        assert _iou(a, b) == pytest.approx(1.0)

    def test_no_overlap(self):
        a = GlyphBox(page=0, x0=0, y0=0, x1=10, y1=10)
        b = GlyphBox(page=0, x0=20, y0=20, x1=30, y1=30)
        assert _iou(a, b) == 0.0

    def test_partial_overlap(self):
        a = GlyphBox(page=0, x0=0, y0=0, x1=10, y1=10)
        b = GlyphBox(page=0, x0=5, y0=5, x1=15, y1=15)
        # Intersection = 5×5 = 25, Union = 100 + 100 - 25 = 175
        assert _iou(a, b) == pytest.approx(25.0 / 175.0, rel=1e-3)

    def test_one_inside_another(self):
        a = GlyphBox(page=0, x0=0, y0=0, x1=20, y1=20)
        b = GlyphBox(page=0, x0=5, y0=5, x1=10, y1=10)
        # Intersection = 25, Union = 400 + 25 - 25 = 400
        assert _iou(a, b) == pytest.approx(25.0 / 400.0, rel=1e-3)

    def test_zero_area_boxes(self):
        a = GlyphBox(page=0, x0=0, y0=0, x1=0, y1=10)
        b = GlyphBox(page=0, x0=0, y0=0, x1=10, y1=10)
        assert _iou(a, b) == 0.0


# ── Tile dedup ─────────────────────────────────────────────────────────


class TestDedupTiles:
    def test_no_duplicates(self):
        tokens = [
            GlyphBox(page=0, x0=0, y0=0, x1=10, y1=10, text="A"),
            GlyphBox(page=0, x0=50, y0=50, x1=60, y1=60, text="B"),
        ]
        confs = [0.9, 0.8]
        out_t, out_c = _dedup_tiles(tokens, confs)
        assert len(out_t) == 2
        assert len(out_c) == 2

    def test_removes_lower_confidence_duplicate(self):
        tokens = [
            GlyphBox(page=0, x0=0, y0=0, x1=10, y1=10, text="A"),
            GlyphBox(page=0, x0=0, y0=0, x1=10, y1=10, text="A"),  # exact dup
        ]
        confs = [0.9, 0.7]
        out_t, out_c = _dedup_tiles(tokens, confs, dedup_iou=0.5)
        assert len(out_t) == 1
        assert out_c[0] == 0.9  # Higher confidence kept

    def test_keeps_higher_confidence_when_second_is_better(self):
        tokens = [
            GlyphBox(page=0, x0=0, y0=0, x1=10, y1=10, text="A"),
            GlyphBox(page=0, x0=0, y0=0, x1=10, y1=10, text="A"),
        ]
        confs = [0.5, 0.95]
        out_t, out_c = _dedup_tiles(tokens, confs, dedup_iou=0.5)
        assert len(out_t) == 1
        assert out_c[0] == 0.95

    def test_single_token(self):
        tokens = [GlyphBox(page=0, x0=0, y0=0, x1=10, y1=10, text="A")]
        confs = [0.9]
        out_t, out_c = _dedup_tiles(tokens, confs)
        assert len(out_t) == 1

    def test_empty_list(self):
        out_t, out_c = _dedup_tiles([], [])
        assert out_t == []
        assert out_c == []

    def test_high_iou_threshold_keeps_both(self):
        """Partial overlap below the threshold should not trigger dedup."""
        tokens = [
            GlyphBox(page=0, x0=0, y0=0, x1=10, y1=10, text="A"),
            GlyphBox(page=0, x0=7, y0=7, x1=17, y1=17, text="B"),
        ]
        confs = [0.9, 0.8]
        # IoU of these ~ 0.053, well below 0.5
        out_t, out_c = _dedup_tiles(tokens, confs, dedup_iou=0.5)
        assert len(out_t) == 2

    def test_three_way_dedup(self):
        """Three tokens at the same location — only best survives."""
        tokens = [
            GlyphBox(page=0, x0=0, y0=0, x1=10, y1=10, text="A"),
            GlyphBox(page=0, x0=0, y0=0, x1=10, y1=10, text="A"),
            GlyphBox(page=0, x0=0, y0=0, x1=10, y1=10, text="A"),
        ]
        confs = [0.7, 0.9, 0.6]
        out_t, out_c = _dedup_tiles(tokens, confs, dedup_iou=0.5)
        assert len(out_t) == 1
        assert out_c[0] == 0.9


# ── _ocr_one_tile ─────────────────────────────────────────────────────


class TestOcrOneTile:
    def _make_mock_ocr(self, polys, texts, scores):
        """Build a mock PaddleOCR that yields one page result."""
        page_result = {
            "dt_polys": polys,
            "rec_texts": texts,
            "rec_scores": scores,
        }
        mock_ocr = MagicMock()
        mock_ocr.predict.return_value = [page_result]
        return mock_ocr

    def test_basic_extraction(self):
        """OCR returns two tokens; both should convert to GlyphBox in PDF-point space."""
        polys = [
            [[0, 0], [100, 0], [100, 20], [0, 20]],
            [[200, 50], [300, 50], [300, 70], [200, 70]],
        ]
        texts = ["HELLO", "WORLD"]
        scores = [0.95, 0.88]
        ocr = self._make_mock_ocr(polys, texts, scores)

        # sx=2, sy=2 means image is 2x PDF points
        tokens, confs = _ocr_one_tile(
            ocr,
            "fake_array",
            offset_x=0,
            offset_y=0,
            sx=2.0,
            sy=2.0,
            page_num=0,
            min_conf=0.5,
        )
        assert len(tokens) == 2
        assert tokens[0].text == "HELLO"
        assert tokens[0].x0 == pytest.approx(0.0)
        assert tokens[0].x1 == pytest.approx(50.0)  # 100 / 2
        assert tokens[1].text == "WORLD"
        assert confs == [0.95, 0.88]

    def test_offset_applied(self):
        """Tile offset should shift coordinates before PDF-point conversion."""
        polys = [[[0, 0], [100, 0], [100, 20], [0, 20]]]
        texts = ["TEST"]
        scores = [0.9]
        ocr = self._make_mock_ocr(polys, texts, scores)

        tokens, _ = _ocr_one_tile(
            ocr,
            "fake_array",
            offset_x=200,
            offset_y=100,
            sx=2.0,
            sy=2.0,
            page_num=0,
            min_conf=0.5,
        )
        assert tokens[0].x0 == pytest.approx(100.0)  # (0+200) / 2
        assert tokens[0].y0 == pytest.approx(50.0)  # (0+100) / 2

    def test_low_confidence_filtered(self):
        """Tokens below min_conf should be dropped."""
        polys = [
            [[0, 0], [10, 0], [10, 10], [0, 10]],
            [[20, 0], [30, 0], [30, 10], [20, 10]],
        ]
        texts = ["GOOD", "BAD"]
        scores = [0.9, 0.3]
        ocr = self._make_mock_ocr(polys, texts, scores)

        tokens, confs = _ocr_one_tile(
            ocr,
            "fake_array",
            offset_x=0,
            offset_y=0,
            sx=1.0,
            sy=1.0,
            page_num=0,
            min_conf=0.5,
        )
        assert len(tokens) == 1
        assert tokens[0].text == "GOOD"

    def test_empty_text_filtered(self):
        """Tokens with empty text should be dropped."""
        polys = [[[0, 0], [10, 0], [10, 10], [0, 10]]]
        texts = [""]
        scores = [0.9]
        ocr = self._make_mock_ocr(polys, texts, scores)

        tokens, _ = _ocr_one_tile(
            ocr,
            "fake_array",
            offset_x=0,
            offset_y=0,
            sx=1.0,
            sy=1.0,
            page_num=0,
            min_conf=0.5,
        )
        assert len(tokens) == 0

    def test_origin_is_ocr_full(self):
        """All returned tokens should have origin='ocr_full'."""
        polys = [[[0, 0], [10, 0], [10, 10], [0, 10]]]
        texts = ["X"]
        scores = [0.9]
        ocr = self._make_mock_ocr(polys, texts, scores)

        tokens, _ = _ocr_one_tile(
            ocr,
            "fake_array",
            offset_x=0,
            offset_y=0,
            sx=1.0,
            sy=1.0,
            page_num=3,
            min_conf=0.5,
        )
        assert tokens[0].origin == "ocr_full"
        assert tokens[0].page == 3

    def test_none_attributes_skipped(self):
        """If PaddleOCR returns None for polys/texts/scores, return nothing."""
        page_result = {"dt_polys": None, "rec_texts": None, "rec_scores": None}
        mock_ocr = MagicMock()
        mock_ocr.predict.return_value = [page_result]

        tokens, _ = _ocr_one_tile(
            mock_ocr,
            "fake_array",
            offset_x=0,
            offset_y=0,
            sx=1.0,
            sy=1.0,
            page_num=0,
            min_conf=0.5,
        )
        assert len(tokens) == 0

    def test_object_style_result(self):
        """PaddleOCR may return an object with attributes instead of a dict."""

        class FakeResult:
            dt_polys = [[[0, 0], [10, 0], [10, 10], [0, 10]]]
            rec_texts = ["OBJ"]
            rec_scores = [0.9]

            # Not a dict, so hasattr(x, "get") is False
            pass

        mock_ocr = MagicMock()
        mock_ocr.predict.return_value = [FakeResult()]

        tokens, confs = _ocr_one_tile(
            mock_ocr,
            "fake_array",
            offset_x=0,
            offset_y=0,
            sx=1.0,
            sy=1.0,
            page_num=0,
            min_conf=0.5,
        )
        assert len(tokens) == 1
        assert tokens[0].text == "OBJ"


# ── extract_vocr_tokens (integration-level, mocked engine) ────────────


class TestExtractVocrTokens:
    """Test the public entry point with a mocked PaddleOCR engine."""

    def _make_cfg(self, **overrides):
        defaults = dict(
            enable_vocr=True,
            vocr_max_tile_px=5000,  # large enough to avoid tiling for test images
            vocr_tile_overlap=0.05,
            vocr_min_confidence=0.5,
            vocr_tile_dedup_iou=0.5,
            vocr_min_text_length=0,
            vocr_strip_whitespace=True,
            vocr_heartbeat_interval=60,
        )
        defaults.update(overrides)
        return GroupingConfig(**defaults)

    def _fake_predict(self, polys, texts, scores):
        """Return a callable that yields one page result dict."""

        def predict(img_array):
            return [{"dt_polys": polys, "rec_texts": texts, "rec_scores": scores}]

        return predict

    @patch("plancheck.vocr.extract._extract_ocr_tokens")
    def test_extract_vocr_tokens_delegates(self, mock_inner):
        """extract_vocr_tokens should delegate to _extract_ocr_tokens."""
        from plancheck.vocr.extract import extract_vocr_tokens

        mock_inner.return_value = ([], [])
        img = Image.new("RGB", (100, 100))
        cfg = self._make_cfg()
        extract_vocr_tokens(img, 0, 100.0, 100.0, cfg)
        mock_inner.assert_called_once()

    @patch("plancheck.vocr.engine._get_ocr")
    def test_single_pass_no_tiling(self, mock_get_ocr):
        """Small image should run in single pass (no tiling)."""
        from plancheck.vocr.extract import extract_vocr_tokens

        polys = [[[0, 0], [20, 0], [20, 10], [0, 10]]]
        texts = ["HELLO"]
        scores = [0.9]

        mock_ocr = MagicMock()
        mock_ocr.predict.side_effect = self._fake_predict(polys, texts, scores)
        mock_get_ocr.return_value = mock_ocr

        img = Image.new("RGB", (200, 100))
        cfg = self._make_cfg(vocr_max_tile_px=5000)
        tokens, confs = extract_vocr_tokens(img, 0, 100.0, 50.0, cfg)

        assert len(tokens) >= 1
        assert tokens[0].text == "HELLO"

    @patch("plancheck.vocr.engine._get_ocr")
    def test_whitespace_tokens_stripped(self, mock_get_ocr):
        """Whitespace-only tokens should be removed when strip_ws=True."""
        from plancheck.vocr.extract import extract_vocr_tokens

        polys = [
            [[0, 0], [10, 0], [10, 10], [0, 10]],
            [[20, 0], [30, 0], [30, 10], [20, 10]],
        ]
        texts = ["   ", "REAL"]
        scores = [0.9, 0.9]

        mock_ocr = MagicMock()
        mock_ocr.predict.side_effect = self._fake_predict(polys, texts, scores)
        mock_get_ocr.return_value = mock_ocr

        img = Image.new("RGB", (100, 100))
        cfg = self._make_cfg(vocr_strip_whitespace=True)
        tokens, confs = extract_vocr_tokens(img, 0, 100.0, 100.0, cfg)

        texts_out = [t.text for t in tokens]
        assert "   " not in texts_out
        assert "REAL" in texts_out

    @patch("plancheck.vocr.engine._get_ocr")
    def test_min_text_length_filter(self, mock_get_ocr):
        """Tokens shorter than min_text_length should be dropped."""
        from plancheck.vocr.extract import extract_vocr_tokens

        polys = [
            [[0, 0], [10, 0], [10, 10], [0, 10]],
            [[20, 0], [30, 0], [30, 10], [20, 10]],
        ]
        texts = ["A", "HELLO"]
        scores = [0.9, 0.9]

        mock_ocr = MagicMock()
        mock_ocr.predict.side_effect = self._fake_predict(polys, texts, scores)
        mock_get_ocr.return_value = mock_ocr

        img = Image.new("RGB", (100, 100))
        cfg = self._make_cfg(vocr_min_text_length=3)
        tokens, confs = extract_vocr_tokens(img, 0, 100.0, 100.0, cfg)

        assert all(len(t.text) >= 3 for t in tokens)

    @patch("plancheck.vocr.engine._get_ocr")
    def test_rgba_image_converted(self, mock_get_ocr):
        """RGBA input should be converted to RGB without error."""
        from plancheck.vocr.extract import extract_vocr_tokens

        mock_ocr = MagicMock()
        mock_ocr.predict.side_effect = self._fake_predict([], [], [])
        mock_get_ocr.return_value = mock_ocr

        img = Image.new("RGBA", (100, 100))
        cfg = self._make_cfg()
        tokens, confs = extract_vocr_tokens(img, 0, 100.0, 100.0, cfg)
        # Should not raise; empty result is fine
        assert tokens == []

    @patch("plancheck.vocr.engine._get_ocr")
    def test_exception_returns_empty(self, mock_get_ocr):
        """If PaddleOCR raises an exception, return empty lists gracefully."""
        from plancheck.vocr.extract import extract_vocr_tokens

        mock_ocr = MagicMock()
        mock_ocr.predict.side_effect = RuntimeError("GPU OOM")
        mock_get_ocr.return_value = mock_ocr

        img = Image.new("RGB", (100, 100))
        cfg = self._make_cfg()
        tokens, confs = extract_vocr_tokens(img, 0, 100.0, 100.0, cfg)
        assert tokens == []
        assert confs == []
