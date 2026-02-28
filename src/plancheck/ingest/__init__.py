"""Ingest stage — PDF file validation, metadata, and rendering.

Public API
----------
- :func:`ingest_pdf` — open + validate a PDF, return :class:`PdfMeta`
- :func:`render_page_image` — render one page to PIL Image at a given DPI
- :func:`build_page_context` — single-open: extract all page data at once
- :class:`PdfMeta` — PDF-level metadata container
- :class:`PageInfo` — per-page dimensions
- :class:`PageContext` — pre-extracted per-page data
- :class:`IngestError` — raised on validation failures
"""

from .ingest import (
    IngestError,
    PageContext,
    PageInfo,
    PdfMeta,
    build_page_context,
    ingest_pdf,
    render_page_image,
)

__all__ = [
    "IngestError",
    "PageContext",
    "PageInfo",
    "PdfMeta",
    "build_page_context",
    "ingest_pdf",
    "render_page_image",
]
