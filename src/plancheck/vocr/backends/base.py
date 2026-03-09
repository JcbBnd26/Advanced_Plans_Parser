"""OCR backend abstract base class and common types.

This module defines the interface that all OCR backends must implement,
plus the common data structures used to pass results between layers.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING, List

import numpy as np

if TYPE_CHECKING:
    from ...config import GroupingConfig

log = logging.getLogger(__name__)


@dataclass
class TextBox:
    """Single OCR detection result.

    Coordinates are in image pixels (tile-local or full-image depending
    on context). The caller is responsible for converting to PDF points.

    Attributes
    ----------
    polygon : List[List[float]]
        Four corner points as [[x0,y0], [x1,y1], [x2,y2], [x3,y3]].
        Typically top-left, top-right, bottom-right, bottom-left.
    text : str
        Recognized text content.
    confidence : float
        Recognition confidence in [0, 1] range.
    """
    polygon: List[List[float]]
    text: str
    confidence: float

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        """Return axis-aligned bounding box (x0, y0, x1, y1)."""
        xs = [p[0] for p in self.polygon]
        ys = [p[1] for p in self.polygon]
        return (min(xs), min(ys), max(xs), max(ys))


class OCRBackend(ABC):
    """Abstract base class for OCR backends.

    Implementations must provide a `predict` method that takes an RGB
    numpy array and returns a list of TextBox detections.

    Backends should be thread-safe for concurrent `predict` calls.
    """

    @abstractmethod
    def predict(self, image: np.ndarray) -> List[TextBox]:
        """Run OCR on an image.

        Parameters
        ----------
        image : np.ndarray
            RGB image as HxWx3 numpy array (uint8).

        Returns
        -------
        List[TextBox]
            List of detected text boxes with coordinates in pixels.
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Backend identifier for logging/debugging."""
        ...


# Cache: config-key → OCRBackend instance (LRU, capped)
_MAX_CACHE = 2
_backend_cache: OrderedDict[tuple, OCRBackend] = OrderedDict()


def clear_backend_cache() -> int:
    """Drop all cached OCR backend instances, freeing memory.

    Returns the number of entries that were evicted.
    """
    n = len(_backend_cache)
    _backend_cache.clear()
    log.info("Cleared %d cached OCR backend(s)", n)
    return n


def _backend_key(cfg: "GroupingConfig | None") -> tuple:
    """Derive a hashable cache key from VOCR-relevant config fields."""
    if cfg is None:
        return ("surya", "cpu")
    return (
        getattr(cfg, "vocr_backend", "surya"),
        getattr(cfg, "vocr_device", "cpu"),
    )


def get_ocr_backend(cfg: "GroupingConfig | None" = None) -> OCRBackend:
    """Return a lazily-initialized OCR backend.

    Instances are cached by configuration key so that a new backend
    is created only when settings actually change.

    Parameters
    ----------
    cfg : GroupingConfig, optional
        When *None*, uses Surya with CPU defaults.

    Returns
    -------
    OCRBackend
        Configured OCR backend instance.

    Raises
    ------
    ValueError
        If the configured backend is not available.
    """
    key = _backend_key(cfg)

    if key in _backend_cache:
        _backend_cache.move_to_end(key)
        return _backend_cache[key]

    backend_type = key[0]
    device = key[1]

    if backend_type == "surya":
        from .surya import SuryaOCRBackend
        backend = SuryaOCRBackend(device=device, cfg=cfg)
    else:
        raise ValueError(
            f"Unknown OCR backend: {backend_type!r}. "
            f"Available backends: 'surya'"
        )

    _backend_cache[key] = backend

    # Evict oldest entry if cache exceeds max size
    while len(_backend_cache) > _MAX_CACHE:
        evicted_key, _ = _backend_cache.popitem(last=False)
        log.info(
            "Evicted OCR backend %s from cache (max=%d)",
            evicted_key,
            _MAX_CACHE,
        )

    log.info("Created OCR backend: %s (device=%s)", backend.name, device)
    return backend
