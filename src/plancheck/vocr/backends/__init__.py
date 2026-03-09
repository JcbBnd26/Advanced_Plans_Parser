"""OCR backend abstraction layer.

This module provides a pluggable interface for OCR engines, allowing
easy swapping between different providers (Surya, Tesseract, etc.)
without changing downstream code.

Public API
----------
OCRBackend      — Abstract base class for OCR backends
TextBox         — Dataclass for OCR result items
get_ocr_backend — Factory function to get configured backend
clear_backend_cache — Clear cached backend instances
"""
from __future__ import annotations

from .base import OCRBackend, TextBox, clear_backend_cache, get_ocr_backend

__all__ = [
    "OCRBackend",
    "TextBox",
    "get_ocr_backend",
    "clear_backend_cache",
]
