"""Text-layer OCR (TOCR) — PDF text extraction and preprocessing.

Public API
----------
- :func:`extract_tocr_page` — extract tokens from a PDF path + page index
- :func:`extract_tocr_from_page` — extract from an open pdfplumber Page
- :class:`TocrPageResult` — extraction result container
- :func:`nms_prune` — non-maximum suppression for overlapping boxes
- :func:`estimate_skew_degrees` / :func:`rotate_boxes` — deskew helpers
"""

from .extract import TocrPageResult, extract_tocr_from_page, extract_tocr_page
from .preprocess import estimate_skew_degrees, nms_prune, rotate_boxes

__all__ = [
    "TocrPageResult",
    "extract_tocr_page",
    "extract_tocr_from_page",
    "nms_prune",
    "estimate_skew_degrees",
    "rotate_boxes",
]
