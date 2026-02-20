"""Tests for plancheck.ingest — PDF validation, metadata, and rendering."""

from __future__ import annotations

import struct
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from plancheck.ingest import (
    IngestError,
    PageInfo,
    PdfMeta,
    ingest_pdf,
    render_page_image,
)

# ── PageInfo unit tests ────────────────────────────────────────────────


class TestPageInfo:
    def test_area_sqin_letter(self):
        """8.5×11 in letter page = ~93.5 sq in."""
        p = PageInfo(index=0, width=612.0, height=792.0)
        assert abs(p.area_sqin - 93.5) < 0.1

    def test_area_sqin_large_sheet(self):
        """34×22 in CAD sheet."""
        p = PageInfo(index=0, width=2448.0, height=1584.0)
        assert abs(p.area_sqin - 748.0) < 1.0

    def test_to_dict(self):
        p = PageInfo(index=2, width=612.0, height=792.0)
        d = p.to_dict()
        assert d["index"] == 2
        assert d["width"] == 612.0
        assert d["height"] == 792.0
        assert "area_sqin" in d


# ── PdfMeta unit tests ────────────────────────────────────────────────


class TestPdfMeta:
    def test_page_accessor(self):
        pages = [PageInfo(0, 100, 200), PageInfo(1, 300, 400)]
        meta = PdfMeta(path=Path("test.pdf"), num_pages=2, pages=pages)
        assert meta.page(0).width == 100
        assert meta.page(1).height == 400

    def test_page_accessor_out_of_range(self):
        meta = PdfMeta(
            path=Path("test.pdf"), num_pages=1, pages=[PageInfo(0, 100, 200)]
        )
        with pytest.raises(IndexError):
            meta.page(5)

    def test_to_dict_keys(self):
        meta = PdfMeta(
            path=Path("test.pdf"),
            num_pages=1,
            pages=[PageInfo(0, 612, 792)],
            file_size_bytes=12345,
        )
        d = meta.to_dict()
        assert d["path"] == "test.pdf"
        assert d["num_pages"] == 1
        assert d["file_size_bytes"] == 12345
        assert len(d["pages"]) == 1

    def test_to_dict_with_error(self):
        meta = PdfMeta(path=Path("bad.pdf"), num_pages=0, error="corrupted")
        d = meta.to_dict()
        assert d["error"] == "corrupted"

    def test_to_dict_without_metadata(self):
        meta = PdfMeta(path=Path("x.pdf"), num_pages=0)
        d = meta.to_dict()
        assert "pdf_metadata" not in d


# ── Validation tests ──────────────────────────────────────────────────


class TestValidation:
    def test_missing_file(self, tmp_path):
        with pytest.raises(IngestError, match="not found"):
            ingest_pdf(tmp_path / "nonexistent.pdf")

    def test_directory_not_file(self, tmp_path):
        d = tmp_path / "subdir.pdf"
        d.mkdir()
        with pytest.raises(IngestError, match="Not a file"):
            ingest_pdf(d)

    def test_empty_file(self, tmp_path):
        f = tmp_path / "empty.pdf"
        f.write_bytes(b"")
        with pytest.raises(IngestError, match="Empty file"):
            ingest_pdf(f)

    def test_wrong_extension(self, tmp_path):
        f = tmp_path / "data.txt"
        f.write_text("hello")
        with pytest.raises(IngestError, match="Not a PDF"):
            ingest_pdf(f)

    def test_corrupt_pdf(self, tmp_path):
        """A file with .pdf extension but invalid contents."""
        f = tmp_path / "corrupt.pdf"
        f.write_bytes(b"this is not a pdf file at all")
        with pytest.raises(IngestError, match="Cannot open PDF"):
            ingest_pdf(f)


# ── ingest_pdf with mock pdfplumber ───────────────────────────────────


class TestIngestPdf:
    def _make_fake_pdf(self, tmp_path, num_pages=3, widths=None, heights=None):
        """Create a minimal valid-ish PDF file and mock pdfplumber to read it."""
        f = tmp_path / "test.pdf"
        # Write a minimal PDF header so the extension check passes
        f.write_bytes(b"%PDF-1.4\n%%EOF")
        return f

    def test_basic_ingest(self, tmp_path):
        pdf_path = self._make_fake_pdf(tmp_path)
        # Mock pdfplumber.open to return a fake PDF with 3 pages
        mock_page_0 = MagicMock(width=612.0, height=792.0)
        mock_page_1 = MagicMock(width=2448.0, height=1584.0)
        mock_page_2 = MagicMock(width=612.0, height=792.0)
        mock_pdf = MagicMock()
        mock_pdf.pages = [mock_page_0, mock_page_1, mock_page_2]
        mock_pdf.metadata = {"Title": "Test Plan", "Author": "Engineer"}
        mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
        mock_pdf.__exit__ = MagicMock(return_value=False)

        with patch("plancheck.ingest.ingest.pdfplumber.open", return_value=mock_pdf):
            meta = ingest_pdf(pdf_path)

        assert meta.num_pages == 3
        assert meta.page(0).width == 612.0
        assert meta.page(1).width == 2448.0
        assert meta.pdf_metadata["Title"] == "Test Plan"
        assert meta.error is None

    def test_ingest_no_metadata(self, tmp_path):
        pdf_path = self._make_fake_pdf(tmp_path)
        mock_page = MagicMock(width=100.0, height=200.0)
        mock_pdf = MagicMock()
        mock_pdf.pages = [mock_page]
        mock_pdf.metadata = None
        mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
        mock_pdf.__exit__ = MagicMock(return_value=False)

        with patch("plancheck.ingest.ingest.pdfplumber.open", return_value=mock_pdf):
            meta = ingest_pdf(pdf_path)

        assert meta.num_pages == 1
        assert meta.pdf_metadata == {}

    def test_ingest_bytes_metadata(self, tmp_path):
        """Metadata values that are bytes should be decoded."""
        pdf_path = self._make_fake_pdf(tmp_path)
        mock_page = MagicMock(width=100.0, height=200.0)
        mock_pdf = MagicMock()
        mock_pdf.pages = [mock_page]
        mock_pdf.metadata = {"Producer": b"AutoCAD\x00"}
        mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
        mock_pdf.__exit__ = MagicMock(return_value=False)

        with patch("plancheck.ingest.ingest.pdfplumber.open", return_value=mock_pdf):
            meta = ingest_pdf(pdf_path)

        assert "AutoCAD" in meta.pdf_metadata["Producer"]

    def test_ingest_file_size(self, tmp_path):
        pdf_path = self._make_fake_pdf(tmp_path)
        mock_page = MagicMock(width=100.0, height=200.0)
        mock_pdf = MagicMock()
        mock_pdf.pages = [mock_page]
        mock_pdf.metadata = {}
        mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
        mock_pdf.__exit__ = MagicMock(return_value=False)

        with patch("plancheck.ingest.ingest.pdfplumber.open", return_value=mock_pdf):
            meta = ingest_pdf(pdf_path)

        assert meta.file_size_bytes > 0

    def test_ingest_string_path(self, tmp_path):
        """Accept str paths, not just Path objects."""
        pdf_path = self._make_fake_pdf(tmp_path)
        mock_page = MagicMock(width=100.0, height=200.0)
        mock_pdf = MagicMock()
        mock_pdf.pages = [mock_page]
        mock_pdf.metadata = {}
        mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
        mock_pdf.__exit__ = MagicMock(return_value=False)

        with patch("plancheck.ingest.ingest.pdfplumber.open", return_value=mock_pdf):
            meta = ingest_pdf(str(pdf_path))

        assert meta.num_pages == 1

    def test_ingest_encrypted_pdf_raises(self, tmp_path):
        """Password-protected PDFs raise IngestError."""
        pdf_path = self._make_fake_pdf(tmp_path)
        mock_doc = MagicMock()
        mock_doc.is_extractable = False
        mock_pdf = MagicMock()
        mock_pdf.doc = mock_doc
        mock_pdf.pages = [MagicMock(width=100.0, height=200.0)]
        mock_pdf.metadata = {}
        mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
        mock_pdf.__exit__ = MagicMock(return_value=False)

        with patch("plancheck.ingest.ingest.pdfplumber.open", return_value=mock_pdf):
            with pytest.raises(IngestError, match="password-protected"):
                ingest_pdf(pdf_path)

    def test_ingest_extractable_pdf_ok(self, tmp_path):
        """Non-encrypted PDFs with is_extractable=True pass fine."""
        pdf_path = self._make_fake_pdf(tmp_path)
        mock_doc = MagicMock()
        mock_doc.is_extractable = True
        mock_pdf = MagicMock()
        mock_pdf.doc = mock_doc
        mock_pdf.pages = [MagicMock(width=100.0, height=200.0)]
        mock_pdf.metadata = {}
        mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
        mock_pdf.__exit__ = MagicMock(return_value=False)

        with patch("plancheck.ingest.ingest.pdfplumber.open", return_value=mock_pdf):
            meta = ingest_pdf(pdf_path)

        assert meta.num_pages == 1


# ── render_page_image with mock ──────────────────────────────────────


class TestRenderPageImage:
    def test_render_returns_pil_image(self):
        from PIL import Image as PILImage

        fake_img = PILImage.new("RGB", (200, 100), color=(255, 255, 255))
        mock_img_page = MagicMock()
        mock_img_page.original = fake_img
        mock_page = MagicMock()
        mock_page.to_image.return_value = mock_img_page
        mock_pdf = MagicMock()
        mock_pdf.pages = [mock_page]
        mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
        mock_pdf.__exit__ = MagicMock(return_value=False)

        with patch("plancheck.ingest.ingest.pdfplumber.open", return_value=mock_pdf):
            img = render_page_image(Path("dummy.pdf"), 0, resolution=150)

        assert isinstance(img, PILImage.Image)
        assert img.mode == "RGB"

    def test_render_converts_rgba_to_rgb(self):
        from PIL import Image as PILImage

        fake_img = PILImage.new("RGBA", (200, 100), color=(255, 255, 255, 128))
        mock_img_page = MagicMock()
        mock_img_page.original = fake_img
        mock_page = MagicMock()
        mock_page.to_image.return_value = mock_img_page
        mock_pdf = MagicMock()
        mock_pdf.pages = [mock_page]
        mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
        mock_pdf.__exit__ = MagicMock(return_value=False)

        with patch("plancheck.ingest.ingest.pdfplumber.open", return_value=mock_pdf):
            img = render_page_image(Path("dummy.pdf"), 0)

        assert img.mode == "RGB"

    def test_render_passes_resolution(self):
        from PIL import Image as PILImage

        fake_img = PILImage.new("RGB", (200, 100))
        mock_img_page = MagicMock()
        mock_img_page.original = fake_img
        mock_page = MagicMock()
        mock_page.to_image.return_value = mock_img_page
        mock_pdf = MagicMock()
        mock_pdf.pages = [mock_page]
        mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
        mock_pdf.__exit__ = MagicMock(return_value=False)

        with patch("plancheck.ingest.ingest.pdfplumber.open", return_value=mock_pdf):
            render_page_image(Path("dummy.pdf"), 0, resolution=300)

        mock_page.to_image.assert_called_once_with(resolution=300)
