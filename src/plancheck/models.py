from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


def _region_bbox(
    header: Optional["BlockCluster"],
    entries: list,
    box_bbox: Optional[Tuple[float, float, float, float]] = None,
) -> Tuple[float, float, float, float]:
    """Compute bbox for a region with an optional header and iterable entries.

    If *box_bbox* is pre-set (e.g. from a detected enclosing rectangle)
    it is returned directly.  Otherwise the bbox is the union of the
    header bbox and every entry bbox.  Returns ``(0, 0, 0, 0)`` when
    there are no components.
    """
    if box_bbox:
        return box_bbox
    xs0: list[float] = []
    ys0: list[float] = []
    xs1: list[float] = []
    ys1: list[float] = []
    if header:
        x0, y0, x1, y1 = header.bbox()
        xs0.append(x0)
        ys0.append(y0)
        xs1.append(x1)
        ys1.append(y1)
    for entry in entries:
        x0, y0, x1, y1 = entry.bbox()
        xs0.append(x0)
        ys0.append(y0)
        xs1.append(x1)
        ys1.append(y1)
    if not xs0:
        return (0, 0, 0, 0)
    return (min(xs0), min(ys0), max(xs1), max(ys1))


def _multi_bbox(
    bboxes: List[Optional[Tuple[float, float, float, float]]],
) -> Tuple[float, float, float, float]:
    """Compute the union bbox from a list of optional bounding boxes.

    ``None`` entries are skipped.  Returns ``(0, 0, 0, 0)`` when no
    valid bounding boxes are supplied.
    """
    xs0: list[float] = []
    ys0: list[float] = []
    xs1: list[float] = []
    ys1: list[float] = []
    for bb in bboxes:
        if bb is not None:
            xs0.append(bb[0])
            ys0.append(bb[1])
            xs1.append(bb[2])
            ys1.append(bb[3])
    if not xs0:
        return (0, 0, 0, 0)
    return (min(xs0), min(ys0), max(xs1), max(ys1))


@dataclass
class GraphicElement:
    """A graphical element extracted from the PDF (line, rect, curve)."""

    page: int
    element_type: str  # "line", "rect", "curve"
    x0: float
    y0: float
    x1: float
    y1: float
    stroke_color: Optional[Tuple] = None  # RGB or CMYK tuple
    fill_color: Optional[Tuple] = None  # RGB or CMYK tuple (for rects/curves)
    linewidth: float = 1.0
    pts: Optional[List[Tuple[float, float]]] = None  # For curves

    def width(self) -> float:
        """Horizontal extent in points."""
        return self.x1 - self.x0

    def height(self) -> float:
        """Vertical extent in points."""
        return self.y1 - self.y0

    def area(self) -> float:
        """Area in square points, clamped to zero."""
        return max(0.0, self.width()) * max(0.0, self.height())

    def bbox(self) -> Tuple[float, float, float, float]:
        """Bounding box as ``(x0, y0, x1, y1)``."""
        return (self.x0, self.y0, self.x1, self.y1)

    def is_small_symbol(self, max_size: float = 50.0) -> bool:
        """Check if this element is small enough to be a legend symbol."""
        return self.width() <= max_size and self.height() <= max_size


@dataclass
class LegendEntry:
    """A legend entry: symbol graphic + description text."""

    page: int
    symbol: Optional[GraphicElement] = None  # The graphical symbol
    symbol_bbox: Optional[Tuple[float, float, float, float]] = (
        None  # Combined symbol area
    )
    description: str = ""  # The text description
    description_bbox: Optional[Tuple[float, float, float, float]] = None

    def bbox(self) -> Tuple[float, float, float, float]:
        """Combined bounding box of symbol and description."""
        return _multi_bbox([self.symbol_bbox, self.description_bbox])


@dataclass
class LegendRegion:
    """A legend region: header + entries."""

    page: int
    header: Optional[BlockCluster] = None
    entries: List[LegendEntry] = field(default_factory=list)
    is_boxed: bool = False  # Whether legend is enclosed in a rectangle
    box_bbox: Optional[Tuple[float, float, float, float]] = None
    confidence: float = 0.0  # 0–1 detection confidence

    def header_text(self) -> str:
        """Concatenated text from the first header row."""
        if not self.header or not self.header.rows:
            return ""
        texts = [b.text for b in self.header.rows[0].boxes if b.text]
        return " ".join(texts).strip()

    def bbox(self) -> Tuple[float, float, float, float]:
        """Combined bounding box of header, entries, and enclosing box."""
        return _region_bbox(self.header, self.entries, self.box_bbox)


@dataclass
class AbbreviationEntry:
    """An abbreviation entry: short code + full meaning (no graphics)."""

    page: int
    code: str = ""  # Short abbreviation code (e.g., "AI", "BOC")
    meaning: str = ""  # Full meaning (e.g., "AREA INLET", "BACK OF CURB")
    code_bbox: Optional[Tuple[float, float, float, float]] = None
    meaning_bbox: Optional[Tuple[float, float, float, float]] = None

    def bbox(self) -> Tuple[float, float, float, float]:
        """Combined bounding box of code and meaning."""
        return _multi_bbox([self.code_bbox, self.meaning_bbox])


@dataclass
class AbbreviationRegion:
    """An abbreviation region: header + entries (pure text, no graphics)."""

    page: int
    header: Optional[BlockCluster] = None
    entries: List[AbbreviationEntry] = field(default_factory=list)
    is_boxed: bool = False
    box_bbox: Optional[Tuple[float, float, float, float]] = None
    confidence: float = 0.0  # 0–1 detection confidence

    def header_text(self) -> str:
        """Concatenated text from the first header row."""
        if not self.header or not self.header.rows:
            return ""
        texts = [b.text for b in self.header.rows[0].boxes if b.text]
        return " ".join(texts).strip()

    def bbox(self) -> Tuple[float, float, float, float]:
        """Combined bounding box of header, entries, and enclosing box."""
        return _region_bbox(self.header, self.entries, self.box_bbox)


@dataclass
class RevisionEntry:
    """A revision entry: number, description, and date."""

    page: int
    number: str = ""  # Revision number (e.g., "1", "A", etc.)
    description: str = ""  # Description of the revision
    date: str = ""  # Date of the revision
    row_bbox: Optional[Tuple[float, float, float, float]] = None

    def bbox(self) -> Tuple[float, float, float, float]:
        """Bounding box of this revision row."""
        if self.row_bbox:
            return self.row_bbox
        return (0, 0, 0, 0)


@dataclass
class MiscTitleRegion:
    """A miscellaneous title box (e.g., 'OKLAHOMA DEPARTMENT OF TRANSPORTATION')."""

    page: int
    text: str = ""
    text_block: Optional[BlockCluster] = None
    is_boxed: bool = False
    box_bbox: Optional[Tuple[float, float, float, float]] = None
    confidence: float = 0.0  # 0–1 detection confidence

    def bbox(self) -> Tuple[float, float, float, float]:
        """Bounding box from enclosing box or text block."""
        if self.box_bbox:
            return self.box_bbox
        if self.text_block:
            return self.text_block.bbox()
        return (0, 0, 0, 0)


@dataclass
class RevisionRegion:
    """A revisions box: header + entries (typically a table)."""

    page: int
    header: Optional[BlockCluster] = None
    entries: List[RevisionEntry] = field(default_factory=list)
    is_boxed: bool = False
    box_bbox: Optional[Tuple[float, float, float, float]] = None
    confidence: float = 0.0  # 0–1 detection confidence

    def header_text(self) -> str:
        """Concatenated text from the first header row."""
        if not self.header or not self.header.rows:
            return ""
        texts = [b.text for b in self.header.rows[0].boxes if b.text]
        return " ".join(texts).strip()

    def bbox(self) -> Tuple[float, float, float, float]:
        """Combined bounding box of header, entries, and enclosing box."""
        return _region_bbox(self.header, self.entries, self.box_bbox)


@dataclass
class StandardDetailEntry:
    """A standard detail entry: sheet number + description."""

    page: int
    sheet_number: str = ""  # Sheet/detail number (e.g., "SS-1", "621-1")
    description: str = ""  # Description of the detail sheet
    sheet_bbox: Optional[Tuple[float, float, float, float]] = None
    description_bbox: Optional[Tuple[float, float, float, float]] = None

    def bbox(self) -> Tuple[float, float, float, float]:
        """Combined bounding box of sheet number and description."""
        return _multi_bbox([self.sheet_bbox, self.description_bbox])


@dataclass
class StandardDetailRegion:
    """A standard details region: header + entries (sheet numbers and descriptions)."""

    page: int
    header: Optional[BlockCluster] = None
    subheader: Optional[str] = (
        None  # e.g., "THE FOLLOWING ODOT STANDARD DETAILS SHALL BE USED ON THIS PROJECT:"
    )
    subheader_bbox: Optional[Tuple[float, float, float, float]] = None
    entries: List[StandardDetailEntry] = field(default_factory=list)
    is_boxed: bool = False
    box_bbox: Optional[Tuple[float, float, float, float]] = None
    confidence: float = 0.0  # 0–1 detection confidence

    def header_text(self) -> str:
        """Concatenated text from the first header row."""
        if not self.header or not self.header.rows:
            return ""
        texts = [b.text for b in self.header.rows[0].boxes if b.text]
        return " ".join(texts).strip()

    def bbox(self) -> Tuple[float, float, float, float]:
        """Combined bounding box of header, entries, and enclosing box."""
        return _region_bbox(self.header, self.entries, self.box_bbox)


@dataclass
class GlyphBox:
    """Smallest unit: typically a word box from text extraction or OCR."""

    page: int
    x0: float
    y0: float
    x1: float
    y1: float
    text: str = ""
    origin: str = "text"
    fontname: str = ""
    font_size: float = 0.0

    def width(self) -> float:
        """Horizontal extent in points."""
        return self.x1 - self.x0

    def height(self) -> float:
        """Vertical extent in points."""
        return self.y1 - self.y0

    def area(self) -> float:
        """Area in square points, clamped to zero."""
        return max(0.0, self.width()) * max(0.0, self.height())

    def bbox(self) -> Tuple[float, float, float, float]:
        """Bounding box as ``(x0, y0, x1, y1)``."""
        return (self.x0, self.y0, self.x1, self.y1)

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dict."""
        return {
            "page": self.page,
            "x0": round(self.x0, 3),
            "y0": round(self.y0, 3),
            "x1": round(self.x1, 3),
            "y1": round(self.y1, 3),
            "text": self.text,
            "origin": self.origin,
            "fontname": self.fontname,
            "font_size": round(self.font_size, 3),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "GlyphBox":
        """Deserialize from a dict produced by :meth:`to_dict`."""
        return cls(
            page=d["page"],
            x0=d["x0"],
            y0=d["y0"],
            x1=d["x1"],
            y1=d["y1"],
            text=d.get("text", ""),
            origin=d.get("origin", "text"),
            fontname=d.get("fontname", ""),
            font_size=d.get("font_size", 0.0),
        )


@dataclass
class Span:
    """A contiguous run of tokens within a Line, split by large horizontal gaps.

    Spans preserve column/table structure without breaking line integrity.
    Each span may be assigned a col_id after column detection.
    """

    token_indices: List[int] = field(default_factory=list)
    col_id: Optional[int] = None

    def bbox(self, tokens: List[GlyphBox]) -> Tuple[float, float, float, float]:
        """Compute bounding box from referenced tokens."""
        if not self.token_indices:
            return (0, 0, 0, 0)
        boxes = [tokens[i] for i in self.token_indices]
        return (
            min(b.x0 for b in boxes),
            min(b.y0 for b in boxes),
            max(b.x1 for b in boxes),
            max(b.y1 for b in boxes),
        )

    def text(self, tokens: List[GlyphBox]) -> str:
        """Join token text in index order."""
        return " ".join(tokens[i].text for i in self.token_indices if tokens[i].text)

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dict."""
        return {
            "token_indices": list(self.token_indices),
            "col_id": self.col_id,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Span":
        """Deserialize from a dict produced by :meth:`to_dict`."""
        return cls(
            token_indices=d.get("token_indices", []),
            col_id=d.get("col_id"),
        )


@dataclass
class Line:
    """A horizontal line of text: all tokens sharing the same baseline.

    This is the canonical row-truth layer. A Line is never split by column
    detection - columns only label spans/tokens with col_id.
    """

    line_id: int
    page: int
    token_indices: List[int] = field(default_factory=list)
    baseline_y: float = 0.0
    spans: List[Span] = field(default_factory=list)

    def bbox(self, tokens: List[GlyphBox]) -> Tuple[float, float, float, float]:
        """Compute bounding box from referenced tokens."""
        if not self.token_indices:
            return (0, 0, 0, 0)
        boxes = [tokens[i] for i in self.token_indices]
        return (
            min(b.x0 for b in boxes),
            min(b.y0 for b in boxes),
            max(b.x1 for b in boxes),
            max(b.y1 for b in boxes),
        )

    def text(self, tokens: List[GlyphBox]) -> str:
        """Join token text in x-sorted order."""
        if not self.token_indices:
            return ""
        # Sort by x0 to ensure correct reading order
        sorted_indices = sorted(self.token_indices, key=lambda i: tokens[i].x0)
        return " ".join(tokens[i].text for i in sorted_indices if tokens[i].text)

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dict."""
        return {
            "line_id": self.line_id,
            "page": self.page,
            "token_indices": list(self.token_indices),
            "baseline_y": round(self.baseline_y, 3),
            "spans": [s.to_dict() for s in self.spans],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Line":
        """Deserialize from a dict produced by :meth:`to_dict`."""
        spans = [Span.from_dict(s) for s in d.get("spans", [])]
        return cls(
            line_id=d["line_id"],
            page=d["page"],
            token_indices=d.get("token_indices", []),
            baseline_y=d.get("baseline_y", 0.0),
            spans=spans,
        )


@dataclass
class RowBand:
    """Boxes grouped into a single text row or horizontal band."""

    page: int
    boxes: List[GlyphBox] = field(default_factory=list)
    column_id: Optional[int] = None

    def bbox(self) -> Tuple[float, float, float, float]:
        """Bounding box enclosing all boxes in this row."""
        xs0 = [b.x0 for b in self.boxes]
        ys0 = [b.y0 for b in self.boxes]
        xs1 = [b.x1 for b in self.boxes]
        ys1 = [b.y1 for b in self.boxes]
        return (min(xs0), min(ys0), max(xs1), max(ys1))


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
class NotesColumn:
    """A notes column: header block + associated notes blocks grouped together."""

    page: int
    header: Optional[BlockCluster] = None
    notes_blocks: List[BlockCluster] = field(default_factory=list)
    # For linking continued columns (e.g., "SITE NOTES" and "SITE NOTES (CONT'D)")
    column_group_id: Optional[str] = None  # Shared ID for linked columns
    continues_from: Optional[str] = None  # Header text of the column this continues

    def header_text(self) -> str:
        """Extract the header text from the header block."""
        if not self.header or not self.header.rows:
            return ""
        texts = [b.text for b in self.header.rows[0].boxes if b.text]
        return " ".join(texts).strip()

    def base_header_text(self) -> str:
        """Get header text without continuation suffixes like (CONT'D)."""
        import re

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
        import re

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
        header_idx = blocks.index(self.header) if self.header in blocks else None
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


@dataclass
class SuspectRegion:
    """A region flagged for VOCR / LLM inspection.

    These are locations where the text-layer extraction returned
    suspicious results (e.g. fused compound words with missing
    separators) that may need visual OCR rectification.
    """

    page: int
    x0: float
    y0: float
    x1: float
    y1: float
    word_text: str  # The suspicious extracted word
    context: str  # Surrounding text (e.g. full header)
    reason: str  # Why it was flagged
    source_label: str = ""  # e.g. "note_column_header"
    block_index: int = -1  # Index into blocks list

    def bbox(self) -> Tuple[float, float, float, float]:
        """Bounding box as ``(x0, y0, x1, y1)``."""
        return (self.x0, self.y0, self.x1, self.y1)

    def to_dict(self) -> dict:
        """Serialize to JSON-friendly dict for VOCR pipeline consumption."""
        return {
            "page": self.page,
            "bbox": [
                round(self.x0, 2),
                round(self.y0, 2),
                round(self.x1, 2),
                round(self.y1, 2),
            ],
            "word_text": self.word_text,
            "context": self.context,
            "reason": self.reason,
            "source_label": self.source_label,
            "block_index": self.block_index,
        }
