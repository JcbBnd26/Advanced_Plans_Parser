"""Export module: produce structured output from pipeline results.

Generates CSV and consolidated JSON exports from the per-page artifacts
that the batch pipeline produces.  Exports are written to the ``exports/``
subdirectory of each run folder.

Usage from the batch pipeline::

    from plancheck.export import export_page_results
    export_page_results(page_result, run_dir, pdf_stem)

Or standalone from an existing manifest::

    from plancheck.export import export_from_manifest
    export_from_manifest(Path("runs/run_.../manifest.json"))
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def _safe_str(val: Any) -> str:
    """Convert to string, handling None gracefully."""
    if val is None:
        return ""
    return str(val)


def _bbox_str(bbox: list | tuple | None) -> str:
    """Format a bbox as a compact string."""
    if not bbox:
        return ""
    return f"({bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f})"


# ── Per-page export ────────────────────────────────────────────────────


def export_page_summary_csv(
    page_result: Dict[str, Any],
    out_path: Path,
) -> Path:
    """Write a single-row CSV with page-level summary counts.

    Columns include page number, box/line/block counts, region counts,
    and OCR reconciliation stats.
    """
    counts = page_result.get("counts", {})
    stages = page_result.get("stages", {})

    row = {
        "page": page_result.get("page", 0),
        "page_width": page_result.get("page_width", 0),
        "page_height": page_result.get("page_height", 0),
        "skew_degrees": page_result.get("skew_degrees", 0),
        "boxes": counts.get("boxes", 0),
        "lines": counts.get("lines", 0),
        "blocks": counts.get("blocks", 0),
        "tables": counts.get("tables", 0),
        "notes_columns": counts.get("notes_columns", 0),
        "legend_regions": counts.get("legend_regions", 0),
        "legend_entries": counts.get("legend_entries", 0),
        "abbreviation_regions": counts.get("abbreviation_regions", 0),
        "abbreviation_entries": counts.get("abbreviation_entries", 0),
        "revision_regions": counts.get("revision_regions", 0),
        "revision_entries": counts.get("revision_entries", 0),
        "misc_title_regions": counts.get("misc_title_regions", 0),
        "standard_detail_regions": counts.get("standard_detail_regions", 0),
        "standard_detail_entries": counts.get("standard_detail_entries", 0),
        "ocr_reconcile_accepted": counts.get("ocr_reconcile_accepted", 0),
        "tocr_status": stages.get("tocr", {}).get("status", "n/a"),
        "vocr_status": stages.get("vocr", {}).get("status", "n/a"),
        "reconcile_status": stages.get("reconcile", {}).get("status", "n/a"),
    }

    write_header = not out_path.exists()
    with open(out_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    return out_path


def export_notes_csv(
    columns_json: Path,
    blocks_json: Path,
    out_path: Path,
    page_num: int = 0,
) -> Path:
    """Export notes columns and their content to CSV.

    Reads the columns and blocks JSON artifacts and writes one row per
    notes column with header text, block count, and continuation info.
    """
    columns_data = json.loads(columns_json.read_text()) if columns_json.exists() else []

    rows = []
    for col in columns_data:
        rows.append(
            {
                "page": page_num,
                "header_text": col.get("header_text", ""),
                "base_header_text": col.get("base_header_text", ""),
                "is_continuation": col.get("is_continuation", False),
                "column_group_id": col.get("column_group_id", ""),
                "continues_from": col.get("continues_from", ""),
                "notes_count": col.get("notes_count", 0),
                "bbox": _bbox_str(col.get("bbox")),
            }
        )

    write_header = not out_path.exists()
    with open(out_path, "a", newline="", encoding="utf-8") as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            if write_header:
                writer.writeheader()
            writer.writerows(rows)

    return out_path


def export_abbreviations_csv(
    abbrev_json: Path,
    out_path: Path,
    page_num: int = 0,
) -> Path:
    """Export abbreviation entries to CSV (one row per abbreviation)."""
    data = json.loads(abbrev_json.read_text()) if abbrev_json.exists() else []

    rows = []
    for region in data:
        region_header = region.get("header_text", "")
        for entry in region.get("entries", []):
            rows.append(
                {
                    "page": page_num,
                    "region_header": region_header,
                    "code": entry.get("code", ""),
                    "meaning": entry.get("meaning", ""),
                }
            )

    write_header = not out_path.exists()
    with open(out_path, "a", newline="", encoding="utf-8") as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            if write_header:
                writer.writeheader()
            writer.writerows(rows)

    return out_path


def export_legends_csv(
    legends_json: Path,
    out_path: Path,
    page_num: int = 0,
) -> Path:
    """Export legend entries to CSV (one row per legend entry)."""
    data = json.loads(legends_json.read_text()) if legends_json.exists() else []

    rows = []
    for region in data:
        region_header = region.get("header_text", "")
        for entry in region.get("entries", []):
            rows.append(
                {
                    "page": page_num,
                    "region_header": region_header,
                    "description": entry.get("description", ""),
                    "symbol_bbox": _bbox_str(entry.get("symbol_bbox")),
                }
            )

    write_header = not out_path.exists()
    with open(out_path, "a", newline="", encoding="utf-8") as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            if write_header:
                writer.writeheader()
            writer.writerows(rows)

    return out_path


def export_standard_details_csv(
    details_json: Path,
    out_path: Path,
    page_num: int = 0,
) -> Path:
    """Export standard detail entries to CSV."""
    data = json.loads(details_json.read_text()) if details_json.exists() else []

    rows = []
    for region in data:
        region_header = region.get("header_text", "")
        for entry in region.get("entries", []):
            rows.append(
                {
                    "page": page_num,
                    "region_header": region_header,
                    "sheet_number": entry.get("sheet_number", ""),
                    "description": entry.get("description", ""),
                }
            )

    write_header = not out_path.exists()
    with open(out_path, "a", newline="", encoding="utf-8") as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            if write_header:
                writer.writeheader()
            writer.writerows(rows)

    return out_path


def export_revisions_csv(
    revisions_json: Path,
    out_path: Path,
    page_num: int = 0,
) -> Path:
    """Export revision entries to CSV."""
    data = json.loads(revisions_json.read_text()) if revisions_json.exists() else []

    rows = []
    for region in data:
        region_header = region.get("header_text", "")
        for entry in region.get("entries", []):
            rows.append(
                {
                    "page": page_num,
                    "region_header": region_header,
                    "number": entry.get("number", ""),
                    "description": entry.get("description", ""),
                    "date": entry.get("date", ""),
                }
            )

    write_header = not out_path.exists()
    with open(out_path, "a", newline="", encoding="utf-8") as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            if write_header:
                writer.writeheader()
            writer.writerows(rows)

    return out_path


def export_blocks_csv(
    blocks_json: Path,
    out_path: Path,
    page_num: int = 0,
) -> Path:
    """Export block-level data to CSV (one row per block)."""
    data = json.loads(blocks_json.read_text()) if blocks_json.exists() else []

    rows = []
    for i, blk in enumerate(data):
        bbox = blk.get("bbox", [0, 0, 0, 0])
        line_count = len(blk.get("lines", []))
        row_count = len(blk.get("rows", []))
        # Collect all text from lines or rows
        if blk.get("lines"):
            text_parts = [ln.get("text", "") for ln in blk["lines"]]
        else:
            text_parts = []
            for row in blk.get("rows", []):
                text_parts.extend(row.get("texts", []))
        full_text = " ".join(t for t in text_parts if t).strip()

        rows.append(
            {
                "page": page_num,
                "block_index": i,
                "label": blk.get("label", ""),
                "is_table": blk.get("is_table", False),
                "is_notes": blk.get("is_notes", False),
                "lines": line_count,
                "rows": row_count,
                "bbox": _bbox_str(bbox),
                "text_preview": full_text[:200],
            }
        )

    write_header = not out_path.exists()
    with open(out_path, "a", newline="", encoding="utf-8") as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            if write_header:
                writer.writeheader()
            writer.writerows(rows)

    return out_path


# ── Consolidated export ────────────────────────────────────────────────


def export_page_results(
    page_result: Dict[str, Any],
    run_dir: Path,
    pdf_stem: str,
) -> Dict[str, str]:
    """Export all CSV files for a single page.

    Called from the batch pipeline after ``process_page()``.  Reads the
    JSON artifacts and writes CSVs to ``run_dir / exports /``.

    Returns a dict of export names → file paths.
    """
    exports_dir = run_dir / "exports"
    exports_dir.mkdir(parents=True, exist_ok=True)
    page_num = page_result.get("page", 0)
    artifacts = page_result.get("artifacts", {})

    exported: Dict[str, str] = {}

    # Page summary
    summary_path = exports_dir / f"{pdf_stem}_page_summary.csv"
    export_page_summary_csv(page_result, summary_path)
    exported["page_summary"] = str(summary_path)

    # Blocks
    blocks_json = (
        Path(artifacts.get("boxes_json", "")).parent
        / f"{pdf_stem}_page_{page_num}_blocks.json"
    )
    if blocks_json.exists():
        blocks_csv = exports_dir / f"{pdf_stem}_blocks.csv"
        export_blocks_csv(blocks_json, blocks_csv, page_num)
        exported["blocks"] = str(blocks_csv)

    # Notes columns
    columns_json = blocks_json.parent / f"{pdf_stem}_page_{page_num}_columns.json"
    if columns_json.exists():
        notes_csv = exports_dir / f"{pdf_stem}_notes.csv"
        export_notes_csv(columns_json, blocks_json, notes_csv, page_num)
        exported["notes"] = str(notes_csv)

    # Abbreviations
    abbrev_json_path = artifacts.get("abbreviations_json", "")
    if abbrev_json_path and Path(abbrev_json_path).exists():
        abbrev_csv = exports_dir / f"{pdf_stem}_abbreviations.csv"
        export_abbreviations_csv(Path(abbrev_json_path), abbrev_csv, page_num)
        exported["abbreviations"] = str(abbrev_csv)

    # Legends
    legends_json_path = artifacts.get("legends_json", "")
    if legends_json_path and Path(legends_json_path).exists():
        legends_csv = exports_dir / f"{pdf_stem}_legends.csv"
        export_legends_csv(Path(legends_json_path), legends_csv, page_num)
        exported["legends"] = str(legends_csv)

    # Standard details
    details_json_path = artifacts.get("standard_details_json", "")
    if details_json_path and Path(details_json_path).exists():
        details_csv = exports_dir / f"{pdf_stem}_standard_details.csv"
        export_standard_details_csv(Path(details_json_path), details_csv, page_num)
        exported["standard_details"] = str(details_csv)

    # Revisions
    revisions_json_path = artifacts.get("revisions_json", "")
    if revisions_json_path and Path(revisions_json_path).exists():
        revisions_csv = exports_dir / f"{pdf_stem}_revisions.csv"
        export_revisions_csv(Path(revisions_json_path), revisions_csv, page_num)
        exported["revisions"] = str(revisions_csv)

    return exported


def export_from_manifest(manifest_path: Path) -> Dict[str, str]:
    """Export from an existing run manifest.

    Reads the manifest JSON, iterates over pages, and generates CSVs.
    Returns dict of export names → file paths.
    """
    manifest = json.loads(manifest_path.read_text())
    run_dir = manifest_path.parent
    pdf_name = manifest.get("pdf_name", "unknown")
    pdf_stem = Path(pdf_name).stem.replace(" ", "_")

    exported: Dict[str, str] = {}
    for page_result in manifest.get("pages", []):
        page_exports = export_page_results(page_result, run_dir, pdf_stem)
        exported.update(page_exports)

    return exported
