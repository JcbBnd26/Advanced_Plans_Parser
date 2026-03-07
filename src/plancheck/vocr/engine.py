"""Shared PaddleOCR singleton — used by ocr_reconcile (and formerly ocr_fill).

The engine is cached by configuration key so that changing
``vocr_model_tier`` or the orientation/unwarping flags in
:class:`~plancheck.config.GroupingConfig` transparently returns
a matching PaddleOCR instance.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..config import GroupingConfig

log = logging.getLogger(__name__)

# Model-name lookup by tier.
_MODEL_TIERS: dict[str, tuple[str, str]] = {
    "mobile": ("PP-OCRv5_mobile_det", "en_PP-OCRv5_mobile_rec"),
    "server": ("PP-OCRv5_server_det", "en_PP-OCRv5_server_rec"),
}

# Cache: config-key → PaddleOCR instance  (LRU, capped at _MAX_CACHE entries).
_MAX_CACHE = 3
_ocr_cache: OrderedDict[tuple, object] = OrderedDict()


def clear_ocr_cache() -> int:
    """Drop all cached PaddleOCR instances, freeing GPU/CPU memory.

    Returns the number of entries that were evicted.
    """
    n = len(_ocr_cache)
    _ocr_cache.clear()
    log.info("Cleared %d cached PaddleOCR engine(s)", n)
    return n


def _engine_key(cfg: "GroupingConfig | None") -> tuple:
    """Derive a hashable cache key from the VOCR-relevant config fields."""
    if cfg is None:
        return ("mobile", "gpu", False, False, False)
    return (
        cfg.vocr_model_tier,
        getattr(cfg, "vocr_device", "gpu"),
        cfg.vocr_use_orientation_classify,
        cfg.vocr_use_doc_unwarping,
        cfg.vocr_use_textline_orientation,
    )


def _get_ocr(cfg: "GroupingConfig | None" = None):
    """Return a lazily-initialised PaddleOCR recogniser.

    Instances are cached by the VOCR-relevant config fields so that
    a new engine is created only when settings actually change.

    Parameters
    ----------
    cfg : GroupingConfig, optional
        When *None*, uses mobile-tier defaults (backward compat).
    """
    key = _engine_key(cfg)
    if key in _ocr_cache:
        # Move to end (most recently used)
        _ocr_cache.move_to_end(key)
        return _ocr_cache[key]

    import os

    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

    from paddleocr import PaddleOCR

    tier = key[0] if key[0] in _MODEL_TIERS else "mobile"
    det_model, rec_model = _MODEL_TIERS[tier]
    device = key[1] if key[1] in ("gpu", "cpu") else "gpu"

    _ocr_cache[key] = PaddleOCR(
        text_detection_model_name=det_model,
        text_recognition_model_name=rec_model,
        use_doc_orientation_classify=key[2],
        use_doc_unwarping=key[3],
        use_textline_orientation=key[4],
        device=device,
    )
    # Evict oldest entry if cache exceeds max size
    while len(_ocr_cache) > _MAX_CACHE:
        evicted_key, _ = _ocr_cache.popitem(last=False)
        log.info(
            "Evicted PaddleOCR engine %s from cache (max=%d)",
            evicted_key,
            _MAX_CACHE,
        )
    return _ocr_cache[key]
