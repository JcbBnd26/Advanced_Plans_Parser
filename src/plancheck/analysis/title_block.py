"""Title-block field extraction.

Given a :class:`StructuralBox` with ``box_type == BoxType.title_block``
(already detected by the structural-box layer), this module parses the
contained text into named fields commonly found in construction-plan
title blocks:

* project_name
* sheet_number / sheet_title
* date / revision_date
* engineer / designer / checker
* scale
* drawing_number

The extraction is heuristic-based: it scans the text blocks inside the
title-block rectangle for label → value pairs using keyword matching and
spatial proximity.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..models import BlockCluster, GlyphBox

# ── Data model ─────────────────────────────────────────────────────────


@dataclass
class TitleBlockField:
    """A single extracted field from the title block."""

    label: str  # normalised key, e.g. "project_name"
    value: str  # extracted text value
    bbox: Optional[Tuple[float, float, float, float]] = None
    confidence: float = 0.0


@dataclass
class TitleBlockInfo:
    """Parsed title-block contents for one page."""

    page: int
    bbox: Tuple[float, float, float, float] = (0, 0, 0, 0)
    fields: List[TitleBlockField] = field(default_factory=list)
    raw_text: str = ""
    confidence: float = 0.0

    # ── Convenience accessors ──────────────────────────────────────
    def get(self, key: str, default: str = "") -> str:
        """Return the value for *key*, or *default* if not found."""
        for f in self.fields:
            if f.label == key:
                return f.value
        return default

    @property
    def project_name(self) -> str:
        """Project name from title block fields."""
        return self.get("project_name")

    @property
    def sheet_number(self) -> str:
        """Sheet number from title block fields."""
        return self.get("sheet_number")

    @property
    def sheet_title(self) -> str:
        """Sheet title from title block fields."""
        return self.get("sheet_title")

    @property
    def date(self) -> str:
        """Date from title block fields."""
        return self.get("date")

    @property
    def engineer(self) -> str:
        """Engineer name from title block fields."""
        return self.get("engineer")

    @property
    def scale(self) -> str:
        """Drawing scale from title block fields."""
        return self.get("scale")

    @property
    def drawing_number(self) -> str:
        """Drawing number from title block fields."""
        return self.get("drawing_number")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize title block to a JSON-compatible dict."""
        return {
            "page": self.page,
            "bbox": list(self.bbox),
            "confidence": round(self.confidence, 3),
            "raw_text": self.raw_text,
            "fields": [
                {
                    "label": f.label,
                    "value": f.value,
                    "bbox": list(f.bbox) if f.bbox else None,
                    "confidence": round(f.confidence, 3),
                }
                for f in self.fields
            ],
        }


# ── Label patterns ─────────────────────────────────────────────────────

# Maps a normalised field key to regex patterns that match the label text
# preceding the value.  Patterns are tried in order; first match wins.
_FIELD_PATTERNS: List[Tuple[str, re.Pattern[str]]] = [
    ("project_name", re.compile(r"PROJECT\s*(?:NAME|TITLE)\s*[:.]?", re.I)),
    ("sheet_number", re.compile(r"SHEET\s*(?:NO\.?|NUMBER|#)\s*[:.]?", re.I)),
    ("sheet_title", re.compile(r"SHEET\s*(?:TITLE|NAME)\s*[:.]?", re.I)),
    (
        "drawing_number",
        re.compile(r"(?:DWG|DRAWING)\s*(?:NO\.?|NUMBER|#)\s*[:.]?", re.I),
    ),
    ("date", re.compile(r"(?<!REVISION\s)DATE\s*[:.]?", re.I)),
    ("revision_date", re.compile(r"REVISION\s*DATE\s*[:.]?", re.I)),
    ("engineer", re.compile(r"(?:ENGINEER|DESIGNED\s+BY|DESIGN(?:ER)?)\s*[:.]?", re.I)),
    (
        "checker",
        re.compile(r"(?:CHECK(?:ED)?\s+BY|CHECKER|REVIEWED\s+BY)\s*[:.]?", re.I),
    ),
    ("designer", re.compile(r"(?:DRAWN\s+BY|DRAFTER|DRAFTSMAN)\s*[:.]?", re.I)),
    ("scale", re.compile(r"SCALE\s*[:.]?", re.I)),
    ("job_number", re.compile(r"(?:JOB|PROJECT)\s*(?:NO\.?|NUMBER|#)\s*[:.]?", re.I)),
    ("client", re.compile(r"(?:CLIENT|OWNER)\s*[:.]?", re.I)),
    ("county", re.compile(r"COUNTY\s*[:.]?", re.I)),
    ("state", re.compile(r"STATE\s*[:.]?", re.I)),
]

# When no label is found, try to recognise value patterns directly.
_SCALE_RE = re.compile(r'(?:1\s*["\u201D]?\s*=\s*\d+|NTS|NOT\s+TO\s+SCALE)', re.I)
_SHEET_NUM_RE = re.compile(r"^[A-Z]{0,3}-?\d{1,4}[A-Z]?$", re.I)
_DATE_VALUE_RE = re.compile(
    r"\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}"
    r"|(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\w*\s+\d{1,2},?\s*\d{4}",
    re.I,
)


# ── Public API ─────────────────────────────────────────────────────────


def parse_title_block(
    *,
    contained_blocks: Sequence[BlockCluster],
    page: int,
    box_bbox: Tuple[float, float, float, float],
    tokens: Optional[Sequence[GlyphBox]] = None,
) -> TitleBlockInfo:
    """Extract structured fields from blocks inside a title-block rectangle.

    Parameters
    ----------
    contained_blocks : Sequence[BlockCluster]
        The text blocks whose centres fall inside the title-block box.
    page : int
        Page number (0-based).
    box_bbox : (x0, y0, x1, y1)
        Bounding box of the title-block structural box.
    tokens : Sequence[GlyphBox], optional
        Full token list (needed for line-based blocks).

    Returns
    -------
    TitleBlockInfo
        Parsed title-block data with extracted fields.
    """
    info = TitleBlockInfo(page=page, bbox=box_bbox)

    # Collect all text rows from contained blocks, sorted top-to-bottom
    rows: List[Tuple[str, Tuple[float, float, float, float]]] = []
    for blk in contained_blocks:
        all_boxes = blk.get_all_boxes(tokens)
        if not all_boxes:
            continue
        # Group boxes by approximate y-position into visual rows
        row_groups = _group_into_rows(all_boxes)
        for rg in row_groups:
            text = " ".join(b.text for b in rg if b.text).strip()
            if text:
                bbox = (
                    min(b.x0 for b in rg),
                    min(b.y0 for b in rg),
                    max(b.x1 for b in rg),
                    max(b.y1 for b in rg),
                )
                rows.append((text, bbox))

    # Sort rows top-to-bottom
    rows.sort(key=lambda r: r[1][1])

    # Build raw text
    info.raw_text = "\n".join(text for text, _ in rows)

    # First pass: label → value extraction from rows
    used_indices: set = set()
    for idx, (text, bbox) in enumerate(rows):
        for field_key, pattern in _FIELD_PATTERNS:
            m = pattern.search(text)
            if m:
                # The value is the text after the label on the same line
                value = text[m.end() :].strip().rstrip(":.")
                if not value and idx + 1 < len(rows) and (idx + 1) not in used_indices:
                    # Value might be on the next line
                    value = rows[idx + 1][0].strip()
                    used_indices.add(idx + 1)
                if value:
                    # Don't add duplicate fields
                    if not any(f.label == field_key for f in info.fields):
                        info.fields.append(
                            TitleBlockField(
                                label=field_key,
                                value=value,
                                bbox=bbox,
                                confidence=0.85,
                            )
                        )
                used_indices.add(idx)
                break

    # Second pass: pattern-based value detection for unfound fields
    for idx, (text, bbox) in enumerate(rows):
        if idx in used_indices:
            continue
        text_stripped = text.strip()

        # Scale detection
        if not any(f.label == "scale" for f in info.fields):
            if _SCALE_RE.search(text_stripped):
                info.fields.append(
                    TitleBlockField(
                        label="scale", value=text_stripped, bbox=bbox, confidence=0.6
                    )
                )
                used_indices.add(idx)
                continue

        # Date detection
        if not any(f.label == "date" for f in info.fields):
            dm = _DATE_VALUE_RE.search(text_stripped)
            if dm:
                info.fields.append(
                    TitleBlockField(
                        label="date", value=dm.group(), bbox=bbox, confidence=0.55
                    )
                )
                used_indices.add(idx)
                continue

    # Overall confidence: fraction of key fields populated
    _important = {"project_name", "sheet_number", "date", "scale"}
    found = sum(1 for f in info.fields if f.label in _important)
    info.confidence = round(found / len(_important), 2) if _important else 0.0

    return info


def extract_title_blocks(
    *,
    structural_boxes: Sequence[Any],
    blocks: Sequence[BlockCluster],
    tokens: Optional[Sequence[GlyphBox]] = None,
    page: int = 0,
) -> List[TitleBlockInfo]:
    """Extract title-block info from all title-block structural boxes.

    Parameters
    ----------
    structural_boxes : Sequence[StructuralBox]
        All structural boxes detected on the page.
    blocks : Sequence[BlockCluster]
        All text blocks on the page.
    tokens : Sequence[GlyphBox], optional
        Full token list (needed for line-based blocks).
    page : int
        Page number (0-based).

    Returns
    -------
    List[TitleBlockInfo]
        One entry per title-block box found.
    """
    from ..analysis.structural_boxes import BoxType

    results: List[TitleBlockInfo] = []
    for sb in structural_boxes:
        if sb.box_type != BoxType.title_block:
            continue

        # Gather blocks whose centres fall inside this box
        contained: List[BlockCluster] = []
        bx0, by0, bx1, by1 = sb.bbox()
        for blk in blocks:
            cx = (blk.bbox()[0] + blk.bbox()[2]) / 2
            cy = (blk.bbox()[1] + blk.bbox()[3]) / 2
            if bx0 <= cx <= bx1 and by0 <= cy <= by1:
                contained.append(blk)

        info = parse_title_block(
            contained_blocks=contained,
            page=page,
            box_bbox=sb.bbox(),
            tokens=tokens,
        )
        results.append(info)

    return results


# ── Helpers ────────────────────────────────────────────────────────────


def _group_into_rows(
    boxes: List[GlyphBox], y_tolerance: float = 5.0
) -> List[List[GlyphBox]]:
    """Group glyph boxes into visual rows by y-overlap."""
    if not boxes:
        return []
    sorted_boxes = sorted(boxes, key=lambda b: (b.y0, b.x0))
    rows: List[List[GlyphBox]] = [[sorted_boxes[0]]]
    for box in sorted_boxes[1:]:
        last_row = rows[-1]
        # Check y-overlap with last row
        row_y0 = min(b.y0 for b in last_row)
        row_y1 = max(b.y1 for b in last_row)
        mid_y = (box.y0 + box.y1) / 2
        if row_y0 - y_tolerance <= mid_y <= row_y1 + y_tolerance:
            last_row.append(box)
        else:
            rows.append([box])
    # Sort each row left-to-right
    for row in rows:
        row.sort(key=lambda b: b.x0)
    return rows
