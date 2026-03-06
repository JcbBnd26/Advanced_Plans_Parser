"""Block-level models: BlockCluster, NotesColumn, HeaderTextMixin."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional, Tuple

if TYPE_CHECKING:
    from .tokens import GlyphBox

from .tokens import GlyphBox, Line, RowBand


class HeaderTextMixin:
    """Shared helper for region dataclasses that have a ``header`` block."""

    header: Optional["BlockCluster"]

    def header_text(self) -> str:
        """Concatenated text from the first header row."""
        if not self.header or not self.header.rows:
            return ""
        texts = [b.text for b in self.header.rows[0].boxes if b.text]
        return " ".join(texts).strip()


@dataclass
class BlockCluster:
    """Row bands merged into a logical block (note/legend entry/table region).

    Supports both old (rows-based) and new (lines-based) grouping pipelines.
    When using the new pipeline, `lines` is populated and `rows` may be empty.
    Use `get_all_boxes()` for a unified way to access glyph boxes from either.
    """

    page: int
    rows: List[RowBand] = field(default_factory=list)
    lines: List[Line] = field(default_factory=list)  # New: line-based grouping
    _tokens: Optional[List[GlyphBox]] = field(
        default=None, repr=False
    )  # Token reference for line-based access
    label: Optional[str] = None
    is_table: bool = False
    is_notes: bool = False
    is_header: bool = False
    parent_block_index: Optional[int] = None  # subheader → parent header index

    def bbox(self) -> Tuple[float, float, float, float]:
        """Bounding box enclosing all lines or rows in this block."""
        xs0: List[float] = []
        ys0: List[float] = []
        xs1: List[float] = []
        ys1: List[float] = []

        # Prefer lines if available
        if self.lines and self._tokens:
            for line in self.lines:
                x0, y0, x1, y1 = line.bbox(self._tokens)
                xs0.append(x0)
                ys0.append(y0)
                xs1.append(x1)
                ys1.append(y1)
        else:
            for row in self.rows:
                x0, y0, x1, y1 = row.bbox()
                xs0.append(x0)
                ys0.append(y0)
                xs1.append(x1)
                ys1.append(y1)

        if not xs0:
            return (0, 0, 0, 0)
        return (min(xs0), min(ys0), max(xs1), max(ys1))

    def get_all_boxes(self, tokens: Optional[List[GlyphBox]] = None) -> List[GlyphBox]:
        """Get all glyph boxes from this block (supports both old and new pipelines).

        Args:
            tokens: Token list (required for line-based blocks, optional for row-based)

        Returns:
            List of GlyphBox in reading order (y then x)
        """
        tokens = tokens or self._tokens

        if self.lines and tokens:
            # New pipeline: collect from lines
            boxes = []
            for line in self.lines:
                boxes.extend(tokens[i] for i in line.token_indices)
            return sorted(boxes, key=lambda b: (b.y0, b.x0))
        else:
            # Old pipeline: collect from rows
            boxes = []
            for row in self.rows:
                boxes.extend(row.boxes)
            return sorted(boxes, key=lambda b: (b.y0, b.x0))

    def to_dict(self) -> dict:
        """Serialize to JSON-friendly dict.  Eagerly evaluates bbox()."""
        all_boxes = self.get_all_boxes()
        text_preview = " ".join(b.text for b in all_boxes)
        return {
            "page": self.page,
            "bbox": [round(v, 3) for v in self.bbox()],
            "label": self.label,
            "is_table": self.is_table,
            "is_notes": self.is_notes,
            "is_header": self.is_header,
            "parent_block_index": self.parent_block_index,
            "text": text_preview,
            "lines": [ln.to_dict() for ln in self.lines],
        }

    @classmethod
    def from_dict(cls, d: dict, tokens: List[GlyphBox]) -> "BlockCluster":
        """Reconstruct a BlockCluster from a serialized dict + token list."""
        lines = [Line.from_dict(ld) for ld in d.get("lines", [])]
        blk = cls(
            page=d["page"],
            lines=lines,
            _tokens=tokens,
            label=d.get("label"),
            is_table=d.get("is_table", False),
            is_notes=d.get("is_notes", False),
            is_header=d.get("is_header", False),
            parent_block_index=d.get("parent_block_index"),
        )
        blk.populate_rows_from_lines()
        return blk

    def populate_rows_from_lines(self) -> None:
        """Populate .rows from .lines for backward compatibility.

        Converts each Line into a RowBand so all existing code that reads
        .rows[0].boxes etc. continues to work without modification.
        Requires self._tokens to be set.
        """
        if not self.lines or not self._tokens:
            return

        self.rows = []
        for line in self.lines:
            boxes = [
                self._tokens[i]
                for i in sorted(line.token_indices, key=lambda i: self._tokens[i].x0)
            ]
            row = RowBand(page=line.page, boxes=boxes, column_id=None)
            # Set column_id from the leftmost span if available
            if line.spans:
                row.column_id = line.spans[0].col_id
            self.rows.append(row)


@dataclass
class NotesColumn(HeaderTextMixin):
    """A notes column: header block + associated notes blocks grouped together."""

    page: int
    header: Optional[BlockCluster] = None
    notes_blocks: List[BlockCluster] = field(default_factory=list)
    # For linking continued columns (e.g., "SITE NOTES" and "SITE NOTES (CONT'D)")
    column_group_id: Optional[str] = None  # Shared ID for linked columns
    continues_from: Optional[str] = None  # Header text of the column this continues

    def base_header_text(self) -> str:
        """Get header text without continuation suffixes like (CONT'D)."""
        text = self.header_text().upper()
        # Remove common continuation patterns
        text = re.sub(r"\s*\(CONT['\u2019]?D\)\s*:?\s*$", "", text)
        text = re.sub(r"\s*\(CONTINUED\)\s*:?\s*$", "", text)
        text = re.sub(r"\s*CONT['\u2019]?D\s*:?\s*$", "", text)
        text = re.sub(r"\s*CONTINUED\s*:?\s*$", "", text)
        # Normalize trailing colon
        text = text.rstrip(":").strip()
        return text

    def is_continuation(self) -> bool:
        """Check if this column is a continuation of another."""
        text = self.header_text().upper()
        return bool(
            re.search(
                r"\(CONT['\u2019]?D\)|\(CONTINUED\)|CONT['\u2019]?D\s*:|CONTINUED\s*:",
                text,
            )
        )

    def bbox(self) -> Tuple[float, float, float, float]:
        """Bounding box encompassing the header and all notes blocks.

        Uses median-width clipping: if a block is much wider than the
        column's median width it is clipped to ``median_x0 + median_width``
        so that a single over-wide block cannot inflate the column bbox
        into an adjacent visual column.
        """
        all_bboxes: List[Tuple[float, float, float, float]] = []
        if self.header:
            all_bboxes.append(self.header.bbox())
        for blk in self.notes_blocks:
            all_bboxes.append(blk.bbox())
        if not all_bboxes:
            return (0, 0, 0, 0)

        # Compute median width from notes blocks only (≥2) so the header
        # cannot inflate the median and escape clipping.  Fall back to
        # all bboxes when there aren't enough notes blocks.
        notes_bboxes = [blk.bbox() for blk in self.notes_blocks]
        ref_bboxes = notes_bboxes if len(notes_bboxes) >= 2 else all_bboxes
        ref_widths = sorted(bx[2] - bx[0] for bx in ref_bboxes)
        median_w = ref_widths[len(ref_widths) // 2]

        # Identify outlier-width blocks: any block wider than 1.4× the
        # notes-block median is considered an outlier.  The column x1 is
        # capped at the maximum x1 among non-outlier blocks so that a
        # single wide block cannot push the column boundary into an
        # adjacent column.
        outlier_threshold = median_w * 1.4
        non_outlier_x1 = [
            bx[2] for bx in all_bboxes if (bx[2] - bx[0]) <= outlier_threshold
        ]
        # Fall back to raw max if everything is "outlier" (e.g. single block)
        clip_x1 = (
            max(non_outlier_x1) if non_outlier_x1 else max(bx[2] for bx in all_bboxes)
        )

        xs0: List[float] = []
        ys0: List[float] = []
        xs1: List[float] = []
        ys1: List[float] = []
        for x0, y0, x1, y1 in all_bboxes:
            xs0.append(x0)
            ys0.append(y0)
            xs1.append(min(x1, clip_x1))
            ys1.append(y1)
        return (min(xs0), min(ys0), max(xs1), max(ys1))

    def all_blocks(self) -> List[BlockCluster]:
        """Return header + notes blocks as a flat list."""
        result = []
        if self.header:
            result.append(self.header)
        result.extend(self.notes_blocks)
        return result

    def to_dict(self, blocks: List[BlockCluster]) -> dict:
        """Serialize to JSON-friendly dict using block indices."""
        header_idx = None
        if blocks is not None and self.header is not None:
            try:
                header_idx = blocks.index(self.header)
            except ValueError:
                pass
        notes_indices = []
        for nb in self.notes_blocks:
            try:
                notes_indices.append(blocks.index(nb))
            except ValueError:
                pass
        return {
            "page": self.page,
            "bbox": [round(v, 3) for v in self.bbox()],
            "header_block_index": header_idx,
            "notes_block_indices": notes_indices,
            "column_group_id": self.column_group_id,
            "continues_from": self.continues_from,
            "header_text": self.header_text(),
        }

    @classmethod
    def from_dict(cls, d: dict, blocks: List[BlockCluster]) -> "NotesColumn":
        """Reconstruct a NotesColumn from a serialized dict + block list."""
        hi = d.get("header_block_index")
        header = blocks[hi] if hi is not None and 0 <= hi < len(blocks) else None
        notes_blocks = [
            blocks[i] for i in d.get("notes_block_indices", []) if 0 <= i < len(blocks)
        ]
        return cls(
            page=d["page"],
            header=header,
            notes_blocks=notes_blocks,
            column_group_id=d.get("column_group_id"),
            continues_from=d.get("continues_from"),
        )
