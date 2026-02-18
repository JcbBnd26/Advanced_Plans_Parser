"""Backward compatibility — imports moved to plancheck.vocrpp.preprocess."""

from .vocrpp.preprocess import (
    OcrPreprocessConfig,  # noqa: F401
    OcrPreprocessResult,
    _as_uint8_gray,
    _measure_metrics,
    _require_odd,
    preprocess_image_for_ocr,
)
