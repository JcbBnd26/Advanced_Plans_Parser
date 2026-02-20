"""Ingest stage — PDF file validation, metadata, and rendering.

Public API
----------
- :func:`ingest_pdf` — open + validate a PDF, return :class:`PdfMeta`
- :func:`render_page_image` — render one page to PIL Image at a given DPI
- :class:`PdfMeta` — PDF-level metadata container
- :class:`PageInfo` — per-page dimensions
- :class:`IngestError` — raised on validation failures
"""

from .ingest import IngestError, PageInfo, PdfMeta, ingest_pdf, render_page_image

__all__ = [
    "IngestError",
    "PageInfo",
    "PdfMeta",
    "ingest_pdf",
    "render_page_image",
]
