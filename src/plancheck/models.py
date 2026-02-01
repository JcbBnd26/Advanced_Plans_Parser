from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


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
        return self.x1 - self.x0

    def height(self) -> float:
        return self.y1 - self.y0

    def area(self) -> float:
        return max(0.0, self.width()) * max(0.0, self.height())

    def bbox(self) -> Tuple[float, float, float, float]:
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
        xs0, ys0, xs1, ys1 = [], [], [], []
        if self.symbol_bbox:
            xs0.append(self.symbol_bbox[0])
            ys0.append(self.symbol_bbox[1])
            xs1.append(self.symbol_bbox[2])
            ys1.append(self.symbol_bbox[3])
        if self.description_bbox:
            xs0.append(self.description_bbox[0])
            ys0.append(self.description_bbox[1])
            xs1.append(self.description_bbox[2])
            ys1.append(self.description_bbox[3])
        if not xs0:
            return (0, 0, 0, 0)
        return (min(xs0), min(ys0), max(xs1), max(ys1))


@dataclass
class LegendRegion:
    """A legend region: header + entries."""

    page: int
    header: Optional[BlockCluster] = None
    entries: List[LegendEntry] = field(default_factory=list)
    is_boxed: bool = False  # Whether legend is enclosed in a rectangle
    box_bbox: Optional[Tuple[float, float, float, float]] = None

    def header_text(self) -> str:
        if not self.header or not self.header.rows:
            return ""
        texts = [b.text for b in self.header.rows[0].boxes if b.text]
        return " ".join(texts).strip()

    def bbox(self) -> Tuple[float, float, float, float]:
        if self.box_bbox:
            return self.box_bbox
        xs0, ys0, xs1, ys1 = [], [], [], []
        if self.header:
            x0, y0, x1, y1 = self.header.bbox()
            xs0.append(x0)
            ys0.append(y0)
            xs1.append(x1)
            ys1.append(y1)
        for entry in self.entries:
            x0, y0, x1, y1 = entry.bbox()
            xs0.append(x0)
            ys0.append(y0)
            xs1.append(x1)
            ys1.append(y1)
        if not xs0:
            return (0, 0, 0, 0)
        return (min(xs0), min(ys0), max(xs1), max(ys1))


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
        xs0, ys0, xs1, ys1 = [], [], [], []
        if self.code_bbox:
            xs0.append(self.code_bbox[0])
            ys0.append(self.code_bbox[1])
            xs1.append(self.code_bbox[2])
            ys1.append(self.code_bbox[3])
        if self.meaning_bbox:
            xs0.append(self.meaning_bbox[0])
            ys0.append(self.meaning_bbox[1])
            xs1.append(self.meaning_bbox[2])
            ys1.append(self.meaning_bbox[3])
        if not xs0:
            return (0, 0, 0, 0)
        return (min(xs0), min(ys0), max(xs1), max(ys1))


@dataclass
class AbbreviationRegion:
    """An abbreviation region: header + entries (pure text, no graphics)."""

    page: int
    header: Optional[BlockCluster] = None
    entries: List[AbbreviationEntry] = field(default_factory=list)
    is_boxed: bool = False
    box_bbox: Optional[Tuple[float, float, float, float]] = None

    def header_text(self) -> str:
        if not self.header or not self.header.rows:
            return ""
        texts = [b.text for b in self.header.rows[0].boxes if b.text]
        return " ".join(texts).strip()

    def bbox(self) -> Tuple[float, float, float, float]:
        if self.box_bbox:
            return self.box_bbox
        xs0, ys0, xs1, ys1 = [], [], [], []
        if self.header:
            x0, y0, x1, y1 = self.header.bbox()
            xs0.append(x0)
            ys0.append(y0)
            xs1.append(x1)
            ys1.append(y1)
        for entry in self.entries:
            x0, y0, x1, y1 = entry.bbox()
            xs0.append(x0)
            ys0.append(y0)
            xs1.append(x1)
            ys1.append(y1)
        if not xs0:
            return (0, 0, 0, 0)
        return (min(xs0), min(ys0), max(xs1), max(ys1))


@dataclass
class RevisionEntry:
    """A revision entry: number, description, and date."""

    page: int
    number: str = ""  # Revision number (e.g., "1", "A", etc.)
    description: str = ""  # Description of the revision
    date: str = ""  # Date of the revision
    row_bbox: Optional[Tuple[float, float, float, float]] = None

    def bbox(self) -> Tuple[float, float, float, float]:
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

    def bbox(self) -> Tuple[float, float, float, float]:
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

    def header_text(self) -> str:
        if not self.header or not self.header.rows:
            return ""
        texts = [b.text for b in self.header.rows[0].boxes if b.text]
        return " ".join(texts).strip()

    def bbox(self) -> Tuple[float, float, float, float]:
        if self.box_bbox:
            return self.box_bbox
        xs0, ys0, xs1, ys1 = [], [], [], []
        if self.header:
            x0, y0, x1, y1 = self.header.bbox()
            xs0.append(x0)
            ys0.append(y0)
            xs1.append(x1)
            ys1.append(y1)
        for entry in self.entries:
            x0, y0, x1, y1 = entry.bbox()
            xs0.append(x0)
            ys0.append(y0)
            xs1.append(x1)
            ys1.append(y1)
        if not xs0:
            return (0, 0, 0, 0)
        return (min(xs0), min(ys0), max(xs1), max(ys1))


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

    def width(self) -> float:
        return self.x1 - self.x0

    def height(self) -> float:
        return self.y1 - self.y0

    def area(self) -> float:
        return max(0.0, self.width()) * max(0.0, self.height())

    def bbox(self) -> Tuple[float, float, float, float]:
        return (self.x0, self.y0, self.x1, self.y1)


@dataclass
class RowBand:
    """Boxes grouped into a single text row or horizontal band."""

    page: int
    boxes: List[GlyphBox] = field(default_factory=list)
    column_id: Optional[int] = None

    def bbox(self) -> Tuple[float, float, float, float]:
        xs0 = [b.x0 for b in self.boxes]
        ys0 = [b.y0 for b in self.boxes]
        xs1 = [b.x1 for b in self.boxes]
        ys1 = [b.y1 for b in self.boxes]
        return (min(xs0), min(ys0), max(xs1), max(ys1))


@dataclass
class BlockCluster:
    """Row bands merged into a logical block (note/legend entry/table region)."""

    page: int
    rows: List[RowBand] = field(default_factory=list)
    label: Optional[str] = None
    is_table: bool = False
    is_notes: bool = False
    is_header: bool = False

    def bbox(self) -> Tuple[float, float, float, float]:
        xs0: List[float] = []
        ys0: List[float] = []
        xs1: List[float] = []
        ys1: List[float] = []
        for row in self.rows:
            x0, y0, x1, y1 = row.bbox()
            xs0.append(x0)
            ys0.append(y0)
            xs1.append(x1)
            ys1.append(y1)
        return (min(xs0), min(ys0), max(xs1), max(ys1))


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
        """Bounding box encompassing the header and all notes blocks."""
        xs0: List[float] = []
        ys0: List[float] = []
        xs1: List[float] = []
        ys1: List[float] = []
        if self.header:
            x0, y0, x1, y1 = self.header.bbox()
            xs0.append(x0)
            ys0.append(y0)
            xs1.append(x1)
            ys1.append(y1)
        for blk in self.notes_blocks:
            x0, y0, x1, y1 = blk.bbox()
            xs0.append(x0)
            ys0.append(y0)
            xs1.append(x1)
            ys1.append(y1)
        if not xs0:
            return (0, 0, 0, 0)
        return (min(xs0), min(ys0), max(xs1), max(ys1))

    def all_blocks(self) -> List[BlockCluster]:
        """Return header + notes blocks as a flat list."""
        result = []
        if self.header:
            result.append(self.header)
        result.extend(self.notes_blocks)
        return result
