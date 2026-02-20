"""Summary report generation for plan-check runs.

Produces human-readable HTML reports from :class:`PageResult` and
:class:`DocumentResult` objects.  The reports can be opened in any
browser and contain:

* Per-page overview (quality score, stage timings, region counts)
* Semantic findings grouped by severity
* Document-level cross-page findings
* Title-block field summary table
"""

from __future__ import annotations

import html
import json
from datetime import datetime
from pathlib import Path
from string import Template
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence

if TYPE_CHECKING:
    from ..pipeline import DocumentResult

# ── HTML templates ─────────────────────────────────────────────────────

_BASE_TEMPLATE = Template(
    """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>$title</title>
<style>
  body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
         margin: 2em; background: #f9f9f9; color: #333; }
  h1 { color: #1a5276; border-bottom: 2px solid #1a5276; padding-bottom: .3em; }
  h2 { color: #2c3e50; margin-top: 1.5em; }
  table { border-collapse: collapse; width: 100%; margin: .5em 0 1.5em; }
  th, td { border: 1px solid #ccc; padding: 6px 10px; text-align: left; }
  th { background: #2c3e50; color: #fff; font-weight: 600; }
  tr:nth-child(even) { background: #f0f0f0; }
  .badge { display: inline-block; padding: 2px 8px; border-radius: 4px;
            font-size: .8em; font-weight: 600; color: #fff; }
  .badge-error { background: #e74c3c; }
  .badge-warning { background: #f39c12; }
  .badge-info { background: #3498db; }
  .summary-box { display: inline-block; padding: .5em 1.5em; margin: .3em;
                  border-radius: 6px; text-align: center; }
  .summary-box h3 { margin: 0; font-size: 2em; }
  .summary-box p { margin: .2em 0 0; font-size: .9em; }
  .sb-pages { background: #d5e8d4; border: 1px solid #82b366; }
  .sb-errors { background: #f8d7da; border: 1px solid #e74c3c; }
  .sb-warnings { background: #fff3cd; border: 1px solid #f39c12; }
  .sb-info { background: #d1ecf1; border: 1px solid #3498db; }
</style>
</head>
<body>
<h1>$title</h1>
<p><strong>PDF:</strong> $pdf_name &nbsp;|&nbsp;
   <strong>Generated:</strong> $timestamp &nbsp;|&nbsp;
   <strong>Pages:</strong> $page_count</p>

<div>
  <div class="summary-box sb-pages"><h3>$page_count</h3><p>Pages</p></div>
  <div class="summary-box sb-errors"><h3>$error_count</h3><p>Errors</p></div>
  <div class="summary-box sb-warnings"><h3>$warning_count</h3><p>Warnings</p></div>
  <div class="summary-box sb-info"><h3>$info_count</h3><p>Info</p></div>
</div>

$title_block_section
$findings_section
$page_details_section
</body></html>"""
)

_TITLE_BLOCK_ROW = Template(
    "<tr><td>$page</td><td>$sheet_number</td>"
    "<td>$project_name</td><td>$date</td>"
    "<td>$scale</td><td>$confidence</td></tr>"
)

_FINDING_ROW = Template(
    '<tr><td><span class="badge badge-$severity">$severity_upper</span></td>'
    "<td>$check_id</td><td>$page</td><td>$message</td></tr>"
)

_STAGE_ROW = Template("<tr><td>$name</td><td>$status</td><td>$duration_ms ms</td></tr>")

_STAGE_NAMES = (
    "ingest",
    "tocr",
    "vocrpp",
    "vocr",
    "reconcile",
    "grouping",
    "analysis",
    "checks",
    "export",
)


# ── Section builders ──────────────────────────────────────────────────


def _build_title_block_section(pages: Sequence[Any]) -> str:
    """Build the title-block summary table HTML."""
    rows: List[str] = []
    for pr in pages:
        for tb in pr.title_blocks:
            rows.append(
                _TITLE_BLOCK_ROW.substitute(
                    page=_esc(str(pr.page)),
                    sheet_number=_esc(tb.sheet_number),
                    project_name=_esc(tb.project_name),
                    date=_esc(tb.date),
                    scale=_esc(tb.scale),
                    confidence=_esc(f"{tb.confidence:.0%}"),
                )
            )
    if not rows:
        return ""
    header = (
        "<h2>Title Block Summary</h2>\n<table>"
        "<tr><th>Page</th><th>Sheet #</th><th>Project</th>"
        "<th>Date</th><th>Scale</th><th>Confidence</th></tr>"
    )
    return header + "\n".join(rows) + "\n</table>"


def _build_findings_section(
    page_findings: List[Dict[str, Any]],
    doc_findings: Sequence[Any],
) -> str:
    """Build the semantic findings table HTML."""
    if not page_findings and not doc_findings:
        return "<h2>Semantic Findings</h2><p>No findings.</p>"

    rows: List[str] = []
    for f in page_findings:
        sev = f.get("severity", "info")
        rows.append(
            _FINDING_ROW.substitute(
                severity=sev,
                severity_upper=sev.upper(),
                check_id=_esc(f.get("check_id", "")),
                page=f.get("page", ""),
                message=_esc(f.get("message", "")),
            )
        )
    for f in doc_findings:
        d = f.to_dict() if hasattr(f, "to_dict") else {"message": str(f)}
        sev = d.get("severity", "info")
        rows.append(
            _FINDING_ROW.substitute(
                severity=sev,
                severity_upper=sev.upper(),
                check_id=_esc(d.get("check_id", "")),
                page="doc",
                message=_esc(d.get("message", "")),
            )
        )

    header = (
        "<h2>Semantic Findings</h2>\n<table>"
        "<tr><th>Severity</th><th>Check</th><th>Page</th><th>Message</th></tr>"
    )
    return header + "\n".join(rows) + "\n</table>"


def _build_page_details_section(pages: Sequence[Any]) -> str:
    """Build per-page detail sections with stage timings."""
    parts: List[str] = ["<h2>Per-Page Details</h2>"]
    for pr in pages:
        q = getattr(pr, "page_quality", 0.0)
        parts.append(f"<h3>Page {pr.page}</h3>")
        parts.append(
            f"<p>Size: {pr.page_width:.0f} &times; {pr.page_height:.0f} pts "
            f"&nbsp;|&nbsp; Quality: {q:.0%} "
            f"&nbsp;|&nbsp; Findings: {len(pr.semantic_findings)}</p>"
        )
        stage_rows: List[str] = []
        for name in _STAGE_NAMES:
            sr = pr.stages.get(name)
            if sr:
                stage_rows.append(
                    _STAGE_ROW.substitute(
                        name=name,
                        status=sr.status,
                        duration_ms=sr.duration_ms,
                    )
                )
        parts.append("<table><tr><th>Stage</th><th>Status</th><th>Duration</th></tr>")
        parts.extend(stage_rows)
        parts.append("</table>")
    return "\n".join(parts)


# ── Public API ─────────────────────────────────────────────────────────


def generate_html_report(
    document_result: DocumentResult,
    *,
    output_path: Optional[Path] = None,
    title: str = "Plan Check Report",
) -> str:
    """Generate an HTML summary report from a DocumentResult.

    Parameters
    ----------
    document_result : DocumentResult
        The result from :func:`run_document`.
    output_path : Path, optional
        If given, the HTML is also written to this file.
    title : str
        Report title shown in the header.

    Returns
    -------
    str
        The HTML report as a string.
    """
    pages = document_result.pages
    doc_findings = getattr(document_result, "document_findings", [])
    pdf_name = (
        Path(document_result.pdf_path).name if document_result.pdf_path else "Unknown"
    )

    # Collect all findings
    all_page_findings: List[Dict[str, Any]] = []
    for pr in pages:
        for f in pr.semantic_findings:
            d = f.to_dict() if hasattr(f, "to_dict") else {"message": str(f)}
            d.setdefault("page", pr.page)
            all_page_findings.append(d)

    errors = [f for f in all_page_findings if f.get("severity") == "error"]
    warnings = [f for f in all_page_findings if f.get("severity") == "warning"]
    infos = [f for f in all_page_findings if f.get("severity") == "info"]

    report_html = _BASE_TEMPLATE.substitute(
        title=_esc(title),
        pdf_name=_esc(pdf_name),
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        page_count=len(pages),
        error_count=len(errors),
        warning_count=len(warnings),
        info_count=len(infos),
        title_block_section=_build_title_block_section(pages),
        findings_section=_build_findings_section(all_page_findings, doc_findings),
        page_details_section=_build_page_details_section(pages),
    )

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report_html, encoding="utf-8")

    return report_html


def generate_json_report(
    document_result: DocumentResult,
    *,
    output_path: Optional[Path] = None,
) -> str:
    """Generate a JSON summary report from a DocumentResult.

    Parameters
    ----------
    document_result : DocumentResult
        The result from :func:`run_document`.
    output_path : Path, optional
        If given, the JSON is written to this file.

    Returns
    -------
    str
        The JSON report as a string.
    """
    data = document_result.to_summary_dict()
    data["generated_at"] = datetime.now().isoformat()
    json_str = json.dumps(data, indent=2, default=str)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json_str, encoding="utf-8")

    return json_str


# ── Helpers ────────────────────────────────────────────────────────────


def _esc(text: str) -> str:
    """HTML-escape a string."""
    return html.escape(str(text))
