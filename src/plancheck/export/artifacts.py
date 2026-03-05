"""Artifact serialization and persistence for pipeline results.

This module provides reusable helpers for serializing page results to JSON
artifacts. It extracts logic from run_pdf_batch.py to enable consistent
artifact output across different runners and tools.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from ..models import BlockCluster, GlyphBox, NotesColumn
from ..pipeline import PageResult, StageResult

# ── Serialization helpers ───────────────────────────────────────────────


def serialize_block(blk: BlockCluster) -> Dict[str, Any]:
    """Serialize a BlockCluster to a JSON-compatible dict."""
    x0, y0, x1, y1 = blk.bbox()
    result = {
        "page": blk.page,
        "bbox": [x0, y0, x1, y1],
        "rows": [
            {"bbox": list(row.bbox()), "texts": [b.text for b in row.boxes]}
            for row in blk.rows
        ],
        "label": blk.label,
        "is_table": bool(blk.is_table),
        "is_notes": bool(blk.is_notes),
    }
    if blk.lines and blk._tokens:
        result["lines"] = [
            {
                "line_id": line.line_id,
                "baseline_y": line.baseline_y,
                "bbox": list(line.bbox(blk._tokens)),
                "text": line.text(blk._tokens),
                "spans": [
                    {
                        "col_id": span.col_id,
                        "bbox": list(span.bbox(blk._tokens)),
                        "text": span.text(blk._tokens),
                    }
                    for span in line.spans
                ],
            }
            for line in blk.lines
        ]
    return result


def serialize_column(col: NotesColumn) -> Dict[str, Any]:
    """Serialize a NotesColumn to a JSON-compatible dict."""
    return {
        "header_text": col.header_text(),
        "base_header_text": col.base_header_text(),
        "is_continuation": col.is_continuation(),
        "column_group_id": col.column_group_id,
        "continues_from": col.continues_from,
        "notes_count": len(col.notes_blocks),
        "bbox": list(col.bbox()),
    }


def serialize_structural_boxes(pr: PageResult) -> List[Dict[str, Any]]:
    """Serialize structural boxes from a PageResult."""
    return [
        {
            "box_type": sb.box_type.value,
            "bbox": list(sb.bbox()),
            "confidence": round(sb.confidence, 3),
            "is_synthetic": sb.is_synthetic,
            "contained_blocks": len(sb.contained_block_indices),
            "contained_text_preview": (
                sb.contained_text[:120] if sb.contained_text else ""
            ),
        }
        for sb in pr.structural_boxes
    ]


def serialize_semantic_regions(pr: PageResult) -> List[Dict[str, Any]]:
    """Serialize semantic regions from a PageResult."""
    return [
        {
            "label": sr.label,
            "bbox": list(sr.bbox()),
            "confidence": round(sr.confidence, 3),
            "has_enclosing_box": sr.enclosing_box is not None,
            "anchor_text": (
                " ".join(
                    b.text for r in sr.anchor_block.rows for b in r.boxes if b.text
                )[:80]
                if sr.anchor_block
                else ""
            ),
            "child_blocks": len(sr.child_blocks),
        }
        for sr in pr.semantic_regions
    ]


def serialize_abbreviation_regions(pr: PageResult) -> List[Dict[str, Any]]:
    """Serialize abbreviation regions from a PageResult."""
    return [
        {
            "header_text": ab.header_text(),
            "is_boxed": ab.is_boxed,
            "box_bbox": list(ab.box_bbox) if ab.box_bbox else None,
            "bbox": list(ab.bbox()),
            "confidence": ab.confidence,
            "entries_count": len(ab.entries),
            "entries": [
                {
                    "code": e.code,
                    "meaning": e.meaning,
                    "code_bbox": list(e.code_bbox) if e.code_bbox else None,
                    "meaning_bbox": list(e.meaning_bbox) if e.meaning_bbox else None,
                }
                for e in ab.entries
            ],
        }
        for ab in pr.abbreviation_regions
    ]


def serialize_legend_regions(pr: PageResult) -> List[Dict[str, Any]]:
    """Serialize legend regions from a PageResult."""
    return [
        {
            "header_text": leg.header_text(),
            "is_boxed": leg.is_boxed,
            "box_bbox": list(leg.box_bbox) if leg.box_bbox else None,
            "bbox": list(leg.bbox()),
            "confidence": leg.confidence,
            "entries_count": len(leg.entries),
            "entries": [
                {
                    "symbol_bbox": list(e.symbol_bbox) if e.symbol_bbox else None,
                    "description": e.description,
                    "description_bbox": (
                        list(e.description_bbox) if e.description_bbox else None
                    ),
                }
                for e in leg.entries
            ],
        }
        for leg in pr.legend_regions
    ]


def serialize_revision_regions(pr: PageResult) -> List[Dict[str, Any]]:
    """Serialize revision regions from a PageResult."""
    return [
        {
            "header_text": rev.header_text(),
            "is_boxed": rev.is_boxed,
            "box_bbox": list(rev.box_bbox) if rev.box_bbox else None,
            "bbox": list(rev.bbox()),
            "confidence": rev.confidence,
            "entries_count": len(rev.entries),
            "entries": [
                {
                    "number": e.number,
                    "description": e.description,
                    "date": e.date,
                    "row_bbox": list(e.row_bbox) if e.row_bbox else None,
                }
                for e in rev.entries
            ],
        }
        for rev in pr.revision_regions
    ]


def serialize_misc_title_regions(pr: PageResult) -> List[Dict[str, Any]]:
    """Serialize misc title regions from a PageResult."""
    return [
        {
            "text": mt.text,
            "is_boxed": mt.is_boxed,
            "box_bbox": list(mt.box_bbox) if mt.box_bbox else None,
            "bbox": list(mt.bbox()),
            "confidence": mt.confidence,
        }
        for mt in pr.misc_title_regions
    ]


def serialize_standard_detail_regions(pr: PageResult) -> List[Dict[str, Any]]:
    """Serialize standard detail regions from a PageResult."""
    return [
        {
            "header_text": sd.header_text(),
            "subheader": sd.subheader,
            "subheader_bbox": list(sd.subheader_bbox) if sd.subheader_bbox else None,
            "is_boxed": sd.is_boxed,
            "box_bbox": list(sd.box_bbox) if sd.box_bbox else None,
            "bbox": list(sd.bbox()),
            "confidence": sd.confidence,
            "entries_count": len(sd.entries),
            "entries": [
                {
                    "sheet_number": e.sheet_number,
                    "description": e.description,
                    "sheet_bbox": list(e.sheet_bbox) if e.sheet_bbox else None,
                    "description_bbox": (
                        list(e.description_bbox) if e.description_bbox else None
                    ),
                }
                for e in sd.entries
            ],
        }
        for sd in pr.standard_detail_regions
    ]


# ── Quality scoring ─────────────────────────────────────────────────────


def compute_page_quality(pr: PageResult) -> float:
    """Compute a 0–1 page quality score from a PageResult.

    Scoring:
    - 40% from token density (capped at 2.0 tokens/sq-in)
    - 40% from region count (capped at 3 regions)
    - 20% baseline, reduced by encoding error penalty
    """
    tocr_counts = pr.stages.get("tocr", StageResult(stage="tocr")).counts or {}
    token_density = tocr_counts.get("token_density_per_sqin", 0.0)
    encoding_issues = tocr_counts.get("char_encoding_issues", 0)
    tokens_total = tocr_counts.get("tokens_total", 0)

    density_score = min(1.0, token_density / 2.0)
    region_count = (
        len(pr.notes_columns)
        + len(pr.abbreviation_regions)
        + len(pr.legend_regions)
        + len(pr.revision_regions)
        + len(pr.standard_detail_regions)
        + len(pr.misc_title_regions)
    )
    region_score = min(1.0, region_count / 3.0)
    error_frac = encoding_issues / tokens_total if tokens_total > 0 else 0.0
    error_penalty = min(1.0, error_frac * 10.0)
    return round(
        max(
            0.0,
            (density_score * 0.4 + region_score * 0.4) * (1.0 - error_penalty) + 0.2,
        ),
        3,
    )


# ── File I/O helpers ────────────────────────────────────────────────────


def save_json_artifact(
    data: Any, art_dir: Path, pdf_stem: str, page_num: int, suffix: str
) -> Path:
    """Save JSON artifact with consistent naming: {stem}_page_{n}_{suffix}.json."""
    art_dir.mkdir(parents=True, exist_ok=True)
    path = art_dir / f"{pdf_stem}_page_{page_num}_{suffix}.json"
    path.write_text(json.dumps(data, indent=2))
    return path
