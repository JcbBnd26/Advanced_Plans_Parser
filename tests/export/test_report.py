"""Tests for plancheck.export.report — HTML and JSON report generation."""

from __future__ import annotations

import json

import pytest

from plancheck.export.report import generate_html_report, generate_json_report

# ── Helpers ────────────────────────────────────────────────────────────


def _fake_page_result(page=0, findings=None, title_blocks=None):
    """Minimal PageResult-like object for report tests."""

    class _PR:
        pass

    pr = _PR()
    pr.page = page
    pr.page_width = 612.0
    pr.page_height = 792.0
    pr.page_quality = 0.85
    pr.semantic_findings = findings or []
    pr.title_blocks = title_blocks or []
    pr.stages = {}
    return pr


def _fake_finding(check_id="TEST", severity="warning", message="Test msg", page=0):
    class _F:
        pass

    f = _F()
    f.check_id = check_id
    f.severity = severity
    f.message = message
    f.page = page

    def to_dict():
        return {
            "check_id": check_id,
            "severity": severity,
            "message": message,
            "page": page,
        }

    f.to_dict = to_dict
    return f


def _fake_title_block(
    sheet_number="C-1",
    project_name="Test",
    date="01/01/2024",
    scale="NTS",
    confidence=0.75,
):
    from plancheck.analysis.title_block import TitleBlockField, TitleBlockInfo

    return TitleBlockInfo(
        page=0,
        fields=[
            TitleBlockField(label="sheet_number", value=sheet_number),
            TitleBlockField(label="project_name", value=project_name),
            TitleBlockField(label="date", value=date),
            TitleBlockField(label="scale", value=scale),
        ],
        confidence=confidence,
    )


def _fake_doc_result(pages=None, doc_findings=None, pdf_path="test.pdf"):
    class _DR:
        pass

    dr = _DR()
    dr.pages = pages or []
    dr.document_findings = doc_findings or []
    dr.pdf_path = pdf_path

    def to_summary_dict():
        return {
            "pdf": pdf_path,
            "pages_processed": len(dr.pages),
            "total_page_findings": sum(len(pr.semantic_findings) for pr in dr.pages),
            "document_findings": len(dr.document_findings),
            "pages": [],
            "document_level_findings": [],
        }

    dr.to_summary_dict = to_summary_dict
    return dr


# ── HTML report tests ─────────────────────────────────────────────────


class TestHtmlReport:
    def test_empty_report(self):
        dr = _fake_doc_result()
        html = generate_html_report(dr)
        assert "<html" in html
        assert "0" in html  # 0 pages

    def test_report_with_pages(self):
        pr = _fake_page_result(page=0)
        dr = _fake_doc_result(pages=[pr])
        html = generate_html_report(dr)
        assert "Page 0" in html

    def test_report_with_findings(self):
        finding = _fake_finding(
            check_id="NOTES_DUP", severity="error", message="Dup note"
        )
        pr = _fake_page_result(page=0, findings=[finding])
        dr = _fake_doc_result(pages=[pr])
        html = generate_html_report(dr)
        assert "NOTES_DUP" in html
        assert "error" in html.lower()

    def test_report_with_title_blocks(self):
        tb = _fake_title_block(sheet_number="C-3")
        pr = _fake_page_result(page=0, title_blocks=[tb])
        dr = _fake_doc_result(pages=[pr])
        html = generate_html_report(dr)
        assert "C-3" in html
        assert "Title Block" in html

    def test_report_writes_file(self, tmp_path):
        dr = _fake_doc_result()
        out = tmp_path / "report.html"
        html = generate_html_report(dr, output_path=out)
        assert out.exists()
        assert out.read_text() == html

    def test_custom_title(self):
        dr = _fake_doc_result()
        html = generate_html_report(dr, title="Custom Title")
        assert "Custom Title" in html

    def test_html_escaping(self):
        finding = _fake_finding(message="<script>alert('xss')</script>")
        pr = _fake_page_result(findings=[finding])
        dr = _fake_doc_result(pages=[pr])
        html = generate_html_report(dr)
        assert "<script>" not in html
        assert "&lt;script&gt;" in html

    def test_report_with_doc_findings(self):
        """Document-level findings appear in the report."""
        df = _fake_finding(
            check_id="DOC_SCALE", severity="warning", message="Scale mismatch"
        )
        dr = _fake_doc_result(doc_findings=[df])
        html_out = generate_html_report(dr)
        assert "DOC_SCALE" in html_out
        assert "Scale mismatch" in html_out

    def test_multiple_pages(self):
        """Multiple pages each appear with their own section."""
        pr0 = _fake_page_result(page=0)
        pr1 = _fake_page_result(page=1)
        pr2 = _fake_page_result(page=2)
        dr = _fake_doc_result(pages=[pr0, pr1, pr2])
        html_out = generate_html_report(dr)
        assert "Page 0" in html_out
        assert "Page 1" in html_out
        assert "Page 2" in html_out
        assert "3" in html_out  # 3 pages in summary

    def test_stage_timing_display(self):
        """Stage timing info appears when present."""
        pr = _fake_page_result(page=0)

        class _SR:
            status = "ok"
            duration_ms = 42

        pr.stages = {"ingest": _SR(), "grouping": _SR()}
        dr = _fake_doc_result(pages=[pr])
        html_out = generate_html_report(dr)
        assert "ingest" in html_out
        assert "42 ms" in html_out

    def test_severity_badges(self):
        """Each severity level gets the correct badge class."""
        findings = [
            _fake_finding(severity="error"),
            _fake_finding(severity="warning"),
            _fake_finding(severity="info"),
        ]
        pr = _fake_page_result(findings=findings)
        dr = _fake_doc_result(pages=[pr])
        html_out = generate_html_report(dr)
        assert "badge-error" in html_out
        assert "badge-warning" in html_out
        assert "badge-info" in html_out

    def test_no_findings_message(self):
        """When no findings exist, 'No findings' message shown."""
        pr = _fake_page_result(page=0, findings=[])
        dr = _fake_doc_result(pages=[pr])
        html_out = generate_html_report(dr)
        assert "No findings" in html_out

    def test_pdf_name_in_header(self):
        """PDF filename appears in the report header."""
        dr = _fake_doc_result(pdf_path="/some/path/plan_set.pdf")
        html_out = generate_html_report(dr)
        assert "plan_set.pdf" in html_out


# ── JSON report tests ─────────────────────────────────────────────────


class TestJsonReport:
    def test_empty_report(self):
        dr = _fake_doc_result()
        j = generate_json_report(dr)
        data = json.loads(j)
        assert data["pages_processed"] == 0
        assert "generated_at" in data

    def test_writes_file(self, tmp_path):
        dr = _fake_doc_result()
        out = tmp_path / "report.json"
        j = generate_json_report(dr, output_path=out)
        assert out.exists()
        data = json.loads(out.read_text())
        assert "generated_at" in data
