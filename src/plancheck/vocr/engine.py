"""OCR engine factory — returns the configured OCR backend.

This module provides the main entry point for obtaining an OCR backend.
The actual implementations live in the `backends` subpackage.

The old PaddleOCR-specific `_get_ocr` function is preserved for backward
compatibility but simply delegates to the new backend system.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from .backends import OCRBackend, clear_backend_cache, get_ocr_backend

if TYPE_CHECKING:
    from ..config import GroupingConfig

log = logging.getLogger(__name__)


def clear_ocr_cache() -> int:
    """Drop all cached OCR backend instances, freeing memory.

    Returns the number of entries that were evicted.

    Deprecated: Use `clear_backend_cache()` from `backends` instead.
    """
    return clear_backend_cache()


def _get_ocr(cfg: "GroupingConfig | None" = None) -> OCRBackend:
    """Return a lazily-initialised OCR backend.

    Instances are cached by the VOCR-relevant config fields so that
    a new engine is created only when settings actually change.

    Parameters
    ----------
    cfg : GroupingConfig, optional
        When *None*, uses default backend (Surya) with CPU.

    Returns
    -------
    OCRBackend
        Configured OCR backend instance with `predict(image)` method.

    Note
    ----
    This function is preserved for backward compatibility. New code
    should use `get_ocr_backend(cfg)` from `backends` directly.
    """
    return get_ocr_backend(cfg)
