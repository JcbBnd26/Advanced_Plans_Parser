"""Tests for plancheck.ocr_preprocess_pipeline — standalone OCR image preprocessing."""

from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image

from plancheck.vocrpp.preprocess import (
    OcrPreprocessConfig,
    _measure_metrics,
    _require_odd,
    preprocess_image_for_ocr,
)


@pytest.fixture
def sample_image():
    return Image.linear_gradient("L").resize((100, 100))


# ── _require_odd ───────────────────────────────────────────────────────


class TestRequireOdd:
    def test_even_becomes_next_odd(self):
        assert _require_odd(4) == 5

    def test_odd_unchanged(self):
        assert _require_odd(5) == 5

    def test_below_floor_clamped(self):
        assert _require_odd(1) == 3

    def test_even_below_floor_clamped_then_odd(self):
        assert _require_odd(2) == 3


# ── _measure_metrics ──────────────────────────────────────────────────


class TestMeasureMetrics:
    def test_rgb_image_returns_all_keys(self):
        img = Image.new("RGB", (50, 50), (100, 150, 200))
        metrics = _measure_metrics(img)
        for key in ("width", "height", "mean", "std", "min", "max"):
            assert key in metrics, f"missing key: {key}"
            assert isinstance(metrics[key], float), f"{key} is not float"

    def test_solid_white_known_values(self):
        img = Image.new("L", (10, 10), 255)
        metrics = _measure_metrics(img)
        assert metrics["width"] == pytest.approx(10.0)
        assert metrics["height"] == pytest.approx(10.0)
        assert metrics["mean"] == pytest.approx(255.0)
        assert metrics["std"] == pytest.approx(0.0)
        assert metrics["min"] == pytest.approx(255.0)
        assert metrics["max"] == pytest.approx(255.0)


# ── preprocess_image_for_ocr — disabled path ─────────────────────────


class TestDisabledPath:
    def test_disabled_returns_unchanged(self, sample_image):
        cfg = OcrPreprocessConfig(enabled=False)
        result = preprocess_image_for_ocr(sample_image, cfg)
        assert result.applied_steps == ["disabled"]
        assert result.image.size == sample_image.size


# ── preprocess_image_for_ocr — grayscale ─────────────────────────────


class TestGrayscale:
    def test_rgb_input_converts_to_grayscale(self, sample_image):
        rgb = sample_image.convert("RGB")
        cfg = OcrPreprocessConfig(clahe=False)
        result = preprocess_image_for_ocr(rgb, cfg)
        assert result.image.mode == "L"
        assert "grayscale" in result.applied_steps

    def test_already_gray_no_double_convert(self, sample_image):
        cfg = OcrPreprocessConfig(clahe=False)
        result = preprocess_image_for_ocr(sample_image, cfg)
        assert "grayscale" not in result.applied_steps


# ── preprocess_image_for_ocr — autocontrast ──────────────────────────


class TestAutocontrast:
    def test_autocontrast_spreads_range(self):
        # Create a low-contrast image (all pixels between 100 and 150)
        arr = np.random.randint(100, 151, (100, 100), dtype=np.uint8)
        img = Image.fromarray(arr, mode="L")
        cfg = OcrPreprocessConfig(autocontrast=True, clahe=False)
        result = preprocess_image_for_ocr(img, cfg)
        assert "autocontrast" in result.applied_steps
        out_arr = np.array(result.image)
        # Autocontrast should spread the range wider than the original 50
        assert int(out_arr.max()) - int(out_arr.min()) > 50


# ── preprocess_image_for_ocr — CLAHE ─────────────────────────────────


class TestCLAHE:
    def test_clahe_with_cv2(self, sample_image):
        cv2 = pytest.importorskip("cv2")
        cfg = OcrPreprocessConfig(clahe=True)
        result = preprocess_image_for_ocr(sample_image, cfg)
        assert "clahe" in result.applied_steps

    def test_clahe_without_cv2(self, sample_image):
        with patch.dict("sys.modules", {"cv2": None}):
            cfg = OcrPreprocessConfig(clahe=True)
            result = preprocess_image_for_ocr(sample_image, cfg)
        assert "clahe_skipped_no_cv2" in result.applied_steps


# ── preprocess_image_for_ocr — median denoise ────────────────────────


class TestMedianDenoise:
    def test_median_denoise_produces_valid_image(self, sample_image):
        cfg = OcrPreprocessConfig(median_denoise=True, clahe=False)
        result = preprocess_image_for_ocr(sample_image, cfg)
        assert "median_denoise" in result.applied_steps
        assert result.image.size == sample_image.size


# ── preprocess_image_for_ocr — adaptive binarize ─────────────────────


class TestAdaptiveBinarize:
    def test_adaptive_binarize_with_cv2(self, sample_image):
        cv2 = pytest.importorskip("cv2")
        cfg = OcrPreprocessConfig(adaptive_binarize=True, clahe=False)
        result = preprocess_image_for_ocr(sample_image, cfg)
        assert "adaptive_binarize" in result.applied_steps
        out_arr = np.array(result.image)
        assert set(np.unique(out_arr)).issubset({0, 255})

    def test_adaptive_binarize_without_cv2(self, sample_image):
        with patch.dict("sys.modules", {"cv2": None}):
            cfg = OcrPreprocessConfig(adaptive_binarize=True, clahe=False)
            result = preprocess_image_for_ocr(sample_image, cfg)
        assert "adaptive_binarize_skipped_no_cv2" in result.applied_steps


# ── preprocess_image_for_ocr — sharpen ────────────────────────────────


class TestSharpen:
    def test_sharpen_adds_unsharp_mask_step(self, sample_image):
        cfg = OcrPreprocessConfig(sharpen=True, clahe=False)
        result = preprocess_image_for_ocr(sample_image, cfg)
        assert "unsharp_mask" in result.applied_steps


# ── preprocess_image_for_ocr — full pipeline ─────────────────────────


class TestFullPipeline:
    def test_all_steps_enabled_in_order(self, sample_image):
        _cv2_available = True
        try:
            import cv2  # noqa: F401
        except Exception:
            _cv2_available = False

        rgb = sample_image.convert("RGB")
        cfg = OcrPreprocessConfig(
            grayscale=True,
            autocontrast=True,
            clahe=True,
            median_denoise=True,
            adaptive_binarize=True,
            sharpen=True,
        )
        result = preprocess_image_for_ocr(rgb, cfg)
        if _cv2_available:
            expected = [
                "grayscale",
                "autocontrast",
                "clahe",
                "median_denoise",
                "adaptive_binarize",
                "unsharp_mask",
            ]
        else:
            expected = [
                "grayscale",
                "autocontrast",
                "clahe_skipped_no_cv2",
                "median_denoise",
                "adaptive_binarize_skipped_no_cv2",
                "unsharp_mask",
            ]
        assert result.applied_steps == expected


# ── preprocess_image_for_ocr — save intermediate ─────────────────────


class TestSaveIntermediate:
    def test_save_intermediate_creates_files(self, sample_image, tmp_path):
        rgb = sample_image.convert("RGB")
        cfg = OcrPreprocessConfig(
            save_intermediate=True,
            grayscale=True,
            clahe=False,
        )
        preprocess_image_for_ocr(rgb, cfg, intermediate_dir=str(tmp_path))
        pngs = list(tmp_path.glob("*.png"))
        assert len(pngs) >= 1
