"""Shared PaddleOCR singleton — used by ocr_reconcile (and formerly ocr_fill)."""

from __future__ import annotations

_ocr_instance = None


def _get_ocr():
    """Return a lazily-initialised PaddleOCR recogniser (singleton)."""
    global _ocr_instance
    if _ocr_instance is None:
        import os

        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

        from paddleocr import PaddleOCR

        _ocr_instance = PaddleOCR(
            lang="en",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )
    return _ocr_instance
