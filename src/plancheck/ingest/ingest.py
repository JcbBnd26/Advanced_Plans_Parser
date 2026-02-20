"""Ingest stage — PDF file validation, page metadata, and image rendering.

Centralises PDF opening, file validation, page-dimension extraction, and
page-image rendering so that downstream stages and runner scripts never
call ``pdfplumber.open()`` or ``.to_image()`` directly.

Public API
----------
- :func:`ingest_pdf` — open + validate a PDF, return a :class:`PdfMeta`
- :func:`render_page_image` — render one page to a PIL Image at a given DPI
- :class:`PdfMeta` — lightweight PDF-level metadata container
- :class:`PageInfo` — per-page dimensions
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import pdfplumber
from PIL import Image

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class PageInfo:
    """Dimension metadata for a single PDF page."""

    index: int  # zero-based page number
    width: float  # points
    height: float  # points

    @property
    def area_sqin(self) -> float:
        """Page area in square inches (72 pts = 1 in)."""
        return (self.width / 72.0) * (self.height / 72.0)

    def to_dict(self) -> dict:
        """Serialize page info to a JSON-compatible dict."""
        return {
            "index": self.index,
            "width": round(self.width, 3),
            "height": round(self.height, 3),
            "area_sqin": round(self.area_sqin, 2),
        }


@dataclass
class PdfMeta:
    """PDF-level metadata returned by :func:`ingest_pdf`.

    This is a lightweight descriptor; it does **not** hold the
    ``pdfplumber.PDF`` handle open.  Use :func:`render_page_image` with
    the original path when you need a rendered image.
    """

    path: Path
    num_pages: int
    pages: List[PageInfo] = field(default_factory=list)
    file_size_bytes: int = 0
    pdf_metadata: dict = field(default_factory=dict)  # PDF info dict
    error: Optional[str] = None

    def page(self, index: int) -> PageInfo:
        """Return :class:`PageInfo` for *index* (zero-based)."""
        return self.pages[index]

    def to_dict(self) -> dict:
        """Serialize PDF metadata to a JSON-compatible dict."""
        d: dict = {
            "path": str(self.path),
            "num_pages": self.num_pages,
            "file_size_bytes": self.file_size_bytes,
        }
        if self.pdf_metadata:
            d["pdf_metadata"] = self.pdf_metadata
        if self.error:
            d["error"] = self.error
        d["pages"] = [p.to_dict() for p in self.pages]
        return d


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


class IngestError(Exception):
    """Raised when a PDF cannot be ingested."""


def _validate_pdf_path(pdf_path: Path) -> None:
    """Raise :class:`IngestError` for missing / empty / wrong-extension files."""
    if not pdf_path.exists():
        raise IngestError(f"File not found: {pdf_path}")
    if not pdf_path.is_file():
        raise IngestError(f"Not a file: {pdf_path}")
    if pdf_path.stat().st_size == 0:
        raise IngestError(f"Empty file: {pdf_path}")
    if pdf_path.suffix.lower() != ".pdf":
        raise IngestError(f"Not a PDF (suffix={pdf_path.suffix!r}): {pdf_path}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def ingest_pdf(pdf_path: Path | str) -> PdfMeta:
    """Open and validate a PDF, returning a :class:`PdfMeta` descriptor.

    This is the canonical entry point for the **ingest** pipeline stage.
    It performs:

    1. Path validation (exists, non-empty, ``.pdf`` extension).
    2. Open with pdfplumber and read page count + per-page dimensions.
    3. Extract PDF-level metadata (author, title, producer, etc.).

    The returned :class:`PdfMeta` is a lightweight data object; the PDF
    file handle is **closed** before returning.

    Parameters
    ----------
    pdf_path : Path or str
        Path to the PDF file.

    Returns
    -------
    PdfMeta

    Raises
    ------
    IngestError
        When the file is missing, empty, or cannot be opened as a PDF.
    """
    pdf_path = Path(pdf_path)
    _validate_pdf_path(pdf_path)

    try:
        with pdfplumber.open(pdf_path) as pdf:
            # Check for encryption — pdfminer sets is_extractable = False
            # on password-protected documents that cannot be read.
            if hasattr(pdf, "doc") and hasattr(pdf.doc, "is_extractable"):
                if not pdf.doc.is_extractable:
                    raise IngestError(
                        f"PDF is password-protected or encrypted "
                        f"(text extraction not permitted): {pdf_path}"
                    )

            num_pages = len(pdf.pages)
            pages = [
                PageInfo(
                    index=i,
                    width=float(pg.width),
                    height=float(pg.height),
                )
                for i, pg in enumerate(pdf.pages)
            ]
            # Extract metadata dict (may be None for some PDFs)
            raw_meta = pdf.metadata or {}
            # Coerce to simple str dict (some values can be bytes)
            pdf_metadata = {}
            for k, v in raw_meta.items():
                if isinstance(v, bytes):
                    try:
                        v = v.decode("utf-8", errors="replace")
                    except (UnicodeDecodeError, AttributeError):
                        v = repr(v)
                pdf_metadata[str(k)] = str(v) if v is not None else ""

        file_size = pdf_path.stat().st_size

        meta = PdfMeta(
            path=pdf_path.resolve(),
            num_pages=num_pages,
            pages=pages,
            file_size_bytes=file_size,
            pdf_metadata=pdf_metadata,
        )
        log.info(
            "Ingested %s: %d pages, %.1f KB",
            pdf_path.name,
            num_pages,
            file_size / 1024,
        )
        return meta

    except IngestError:
        raise
    except Exception as exc:
        raise IngestError(f"Cannot open PDF: {exc}") from exc


def render_page_image(
    pdf_path: Path | str,
    page_num: int,
    resolution: int = 200,
) -> Image.Image:
    """Render a single PDF page to a PIL Image at *resolution* DPI.

    Parameters
    ----------
    pdf_path : Path or str
        Path to the PDF.
    page_num : int
        Zero-based page index.
    resolution : int
        Render resolution in DPI.

    Returns
    -------
    PIL.Image.Image
        RGB image of the rendered page.
    """
    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[page_num]
        img_page = page.to_image(resolution=resolution)
        img = img_page.original.copy()
    # Ensure RGB (pdfplumber may return RGBA in some cases)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img
