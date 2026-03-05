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

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dict."""
        d: dict = {
            "page": self.page,
            "element_type": self.element_type,
            "x0": round(self.x0, 3),
            "y0": round(self.y0, 3),
            "x1": round(self.x1, 3),
            "y1": round(self.y1, 3),
            "linewidth": round(self.linewidth, 3),
        }
        if self.stroke_color is not None:
            d["stroke_color"] = list(self.stroke_color)
        if self.fill_color is not None:
            d["fill_color"] = list(self.fill_color)
        if self.pts is not None:
            d["pts"] = [[round(x, 3), round(y, 3)] for x, y in self.pts]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "GraphicElement":
        """Deserialize from a dict produced by :meth:`to_dict`."""
        return cls(
            page=d["page"],
            element_type=d["element_type"],
            x0=d["x0"],
            y0=d["y0"],
            x1=d["x1"],
            y1=d["y1"],
            stroke_color=tuple(d["stroke_color"]) if d.get("stroke_color") else None,
            fill_color=tuple(d["fill_color"]) if d.get("fill_color") else None,
            linewidth=d.get("linewidth", 1.0),
            pts=[(p[0], p[1]) for p in d["pts"]] if d.get("pts") else None,
        )


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

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dict."""
        d: dict = {"page": self.page, "description": self.description}
        if self.symbol is not None:
            d["symbol"] = self.symbol.to_dict()
        d["symbol_bbox"] = list(self.symbol_bbox) if self.symbol_bbox else None
        d["description_bbox"] = (
            list(self.description_bbox) if self.description_bbox else None
        )
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "LegendEntry":
        """Deserialize from a dict produced by :meth:`to_dict`."""
        symbol = GraphicElement.from_dict(d["symbol"]) if d.get("symbol") else None
        return cls(
            page=d["page"],
            symbol=symbol,
            symbol_bbox=tuple(d["symbol_bbox"]) if d.get("symbol_bbox") else None,
            description=d.get("description", ""),
            description_bbox=(
                tuple(d["description_bbox"]) if d.get("description_bbox") else None
            ),
        )


@dataclass
class LegendRegion(HeaderTextMixin):
    """A legend region: header + entries."""

    page: int
    header: Optional[BlockCluster] = None
    entries: List[LegendEntry] = field(default_factory=list)
    is_boxed: bool = False  # Whether legend is enclosed in a rectangle
    box_bbox: Optional[Tuple[float, float, float, float]] = None
    confidence: float = 0.0  # 0–1 detection confidence

    def bbox(self) -> Tuple[float, float, float, float]:
        """Combined bounding box of header, entries, and enclosing box."""
        return _region_bbox(self.header, self.entries, self.box_bbox)

    def to_dict(self, blocks: Optional[List["BlockCluster"]] = None) -> dict:
        """Serialize to a JSON-compatible dict."""
        header_idx = None
        if blocks is not None and self.header is not None:
            try:
                header_idx = blocks.index(self.header)
            except ValueError:
                pass
        return {
            "page": self.page,
            "header_block_index": header_idx,
            "header_text": self.header_text(),
            "entries": [e.to_dict() for e in self.entries],
            "is_boxed": self.is_boxed,
            "box_bbox": list(self.box_bbox) if self.box_bbox else None,
            "confidence": round(self.confidence, 4),
        }

    @classmethod
    def from_dict(
        cls, d: dict, blocks: Optional[List["BlockCluster"]] = None
    ) -> "LegendRegion":
        """Deserialize from a dict produced by :meth:`to_dict`."""
        header = None
        hi = d.get("header_block_index")
        if blocks is not None and hi is not None and 0 <= hi < len(blocks):
            header = blocks[hi]
        entries = [LegendEntry.from_dict(e) for e in d.get("entries", [])]
        return cls(
            page=d["page"],
            header=header,
            entries=entries,
            is_boxed=d.get("is_boxed", False),
            box_bbox=tuple(d["box_bbox"]) if d.get("box_bbox") else None,
            confidence=d.get("confidence", 0.0),
        )


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

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dict."""
        return {
            "page": self.page,
            "code": self.code,
            "meaning": self.meaning,
            "code_bbox": list(self.code_bbox) if self.code_bbox else None,
            "meaning_bbox": list(self.meaning_bbox) if self.meaning_bbox else None,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "AbbreviationEntry":
        """Deserialize from a dict produced by :meth:`to_dict`."""
        return cls(
            page=d["page"],
            code=d.get("code", ""),
            meaning=d.get("meaning", ""),
            code_bbox=tuple(d["code_bbox"]) if d.get("code_bbox") else None,
            meaning_bbox=tuple(d["meaning_bbox"]) if d.get("meaning_bbox") else None,
        )


@dataclass
class AbbreviationRegion(HeaderTextMixin):
    """An abbreviation region: header + entries (pure text, no graphics)."""

    page: int
    header: Optional[BlockCluster] = None
    entries: List[AbbreviationEntry] = field(default_factory=list)
    is_boxed: bool = False
    box_bbox: Optional[Tuple[float, float, float, float]] = None
    confidence: float = 0.0  # 0–1 detection confidence

    def bbox(self) -> Tuple[float, float, float, float]:
        """Combined bounding box of header, entries, and enclosing box."""
        return _region_bbox(self.header, self.entries, self.box_bbox)

    def to_dict(self, blocks: Optional[List["BlockCluster"]] = None) -> dict:
        """Serialize to a JSON-compatible dict."""
        header_idx = None
        if blocks is not None and self.header is not None:
            try:
                header_idx = blocks.index(self.header)
            except ValueError:
                pass
        return {
            "page": self.page,
            "header_block_index": header_idx,
            "header_text": self.header_text(),
            "entries": [e.to_dict() for e in self.entries],
            "is_boxed": self.is_boxed,
            "box_bbox": list(self.box_bbox) if self.box_bbox else None,
            "confidence": round(self.confidence, 4),
        }

    @classmethod
    def from_dict(
        cls, d: dict, blocks: Optional[List["BlockCluster"]] = None
    ) -> "AbbreviationRegion":
        """Deserialize from a dict produced by :meth:`to_dict`."""
        header = None
        hi = d.get("header_block_index")
        if blocks is not None and hi is not None and 0 <= hi < len(blocks):
            header = blocks[hi]
        entries = [AbbreviationEntry.from_dict(e) for e in d.get("entries", [])]
        return cls(
            page=d["page"],
            header=header,
            entries=entries,
            is_boxed=d.get("is_boxed", False),
            box_bbox=tuple(d["box_bbox"]) if d.get("box_bbox") else None,
            confidence=d.get("confidence", 0.0),
        )


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

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dict."""
        return {
            "page": self.page,
            "number": self.number,
            "description": self.description,
            "date": self.date,
            "row_bbox": list(self.row_bbox) if self.row_bbox else None,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RevisionEntry":
        """Deserialize from a dict produced by :meth:`to_dict`."""
        return cls(
            page=d["page"],
            number=d.get("number", ""),
            description=d.get("description", ""),
            date=d.get("date", ""),
            row_bbox=tuple(d["row_bbox"]) if d.get("row_bbox") else None,
        )


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

    def to_dict(self, blocks: Optional[List["BlockCluster"]] = None) -> dict:
        """Serialize to a JSON-compatible dict."""
        text_block_idx = None
        if blocks is not None and self.text_block is not None:
            try:
                text_block_idx = blocks.index(self.text_block)
            except ValueError:
                pass
        return {
            "page": self.page,
            "text": self.text,
            "text_block_index": text_block_idx,
            "is_boxed": self.is_boxed,
            "box_bbox": list(self.box_bbox) if self.box_bbox else None,
            "confidence": round(self.confidence, 4),
        }

    @classmethod
    def from_dict(
        cls, d: dict, blocks: Optional[List["BlockCluster"]] = None
    ) -> "MiscTitleRegion":
        """Deserialize from a dict produced by :meth:`to_dict`."""
        text_block = None
        tbi = d.get("text_block_index")
        if blocks is not None and tbi is not None and 0 <= tbi < len(blocks):
            text_block = blocks[tbi]
        return cls(
            page=d["page"],
            text=d.get("text", ""),
            text_block=text_block,
            is_boxed=d.get("is_boxed", False),
            box_bbox=tuple(d["box_bbox"]) if d.get("box_bbox") else None,
            confidence=d.get("confidence", 0.0),
        )


@dataclass
class RevisionRegion(HeaderTextMixin):
    """A revisions box: header + entries (typically a table)."""

    page: int
    header: Optional[BlockCluster] = None
    entries: List[RevisionEntry] = field(default_factory=list)
    is_boxed: bool = False
    box_bbox: Optional[Tuple[float, float, float, float]] = None
    confidence: float = 0.0  # 0–1 detection confidence

    def bbox(self) -> Tuple[float, float, float, float]:
        """Combined bounding box of header, entries, and enclosing box."""
        return _region_bbox(self.header, self.entries, self.box_bbox)

    def to_dict(self, blocks: Optional[List["BlockCluster"]] = None) -> dict:
        """Serialize to a JSON-compatible dict."""
        header_idx = None
        if blocks is not None and self.header is not None:
            try:
                header_idx = blocks.index(self.header)
            except ValueError:
                pass
        return {
            "page": self.page,
            "header_block_index": header_idx,
            "header_text": self.header_text(),
            "entries": [e.to_dict() for e in self.entries],
            "is_boxed": self.is_boxed,
            "box_bbox": list(self.box_bbox) if self.box_bbox else None,
            "confidence": round(self.confidence, 4),
        }

    @classmethod
    def from_dict(
        cls, d: dict, blocks: Optional[List["BlockCluster"]] = None
    ) -> "RevisionRegion":
        """Deserialize from a dict produced by :meth:`to_dict`."""
        header = None
        hi = d.get("header_block_index")
        if blocks is not None and hi is not None and 0 <= hi < len(blocks):
            header = blocks[hi]
        entries = [RevisionEntry.from_dict(e) for e in d.get("entries", [])]
        return cls(
            page=d["page"],
            header=header,
            entries=entries,
            is_boxed=d.get("is_boxed", False),
            box_bbox=tuple(d["box_bbox"]) if d.get("box_bbox") else None,
            confidence=d.get("confidence", 0.0),
        )


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

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dict."""
        return {
            "page": self.page,
            "sheet_number": self.sheet_number,
            "description": self.description,
            "sheet_bbox": list(self.sheet_bbox) if self.sheet_bbox else None,
            "description_bbox": (
                list(self.description_bbox) if self.description_bbox else None
            ),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "StandardDetailEntry":
        """Deserialize from a dict produced by :meth:`to_dict`."""
        return cls(
            page=d["page"],
            sheet_number=d.get("sheet_number", ""),
            description=d.get("description", ""),
            sheet_bbox=tuple(d["sheet_bbox"]) if d.get("sheet_bbox") else None,
            description_bbox=(
                tuple(d["description_bbox"]) if d.get("description_bbox") else None
            ),
        )


@dataclass
class StandardDetailRegion(HeaderTextMixin):
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

    def bbox(self) -> Tuple[float, float, float, float]:
        """Combined bounding box of header, entries, and enclosing box."""
        return _region_bbox(self.header, self.entries, self.box_bbox)

    def to_dict(self, blocks: Optional[List["BlockCluster"]] = None) -> dict:
        """Serialize to a JSON-compatible dict."""
        header_idx = None
        if blocks is not None and self.header is not None:
            try:
                header_idx = blocks.index(self.header)
            except ValueError:
                pass
        return {
            "page": self.page,
            "header_block_index": header_idx,
            "header_text": self.header_text(),
            "subheader": self.subheader,
            "subheader_bbox": (
                list(self.subheader_bbox) if self.subheader_bbox else None
            ),
            "entries": [e.to_dict() for e in self.entries],
            "is_boxed": self.is_boxed,
            "box_bbox": list(self.box_bbox) if self.box_bbox else None,
            "confidence": round(self.confidence, 4),
        }

    @classmethod
    def from_dict(
        cls, d: dict, blocks: Optional[List["BlockCluster"]] = None
    ) -> "StandardDetailRegion":
        """Deserialize from a dict produced by :meth:`to_dict`."""
        header = None
        hi = d.get("header_block_index")
        if blocks is not None and hi is not None and 0 <= hi < len(blocks):
            header = blocks[hi]
        entries = [StandardDetailEntry.from_dict(e) for e in d.get("entries", [])]
        return cls(
            page=d["page"],
            header=header,
            subheader=d.get("subheader"),
            subheader_bbox=(
                tuple(d["subheader_bbox"]) if d.get("subheader_bbox") else None
            ),
            entries=entries,
            is_boxed=d.get("is_boxed", False),
            box_bbox=tuple(d["box_bbox"]) if d.get("box_bbox") else None,
            confidence=d.get("confidence", 0.0),
        )


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
    confidence: float = 1.0

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
            "confidence": round(self.confidence, 4),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "GlyphBox":
        """Deserialize from a dict produced by :meth:`to_dict`."""
        from pydantic import ValidationError

        from .validation.schemas import GlyphBoxSchema

        try:
            v = GlyphBoxSchema.model_validate(d)
        except ValidationError as exc:
            raise ValueError(f"Invalid GlyphBox dict: {exc}") from exc

        return cls(
            page=v.page,
            x0=v.x0,
            y0=v.y0,
            x1=v.x1,
            y1=v.y1,
            text=v.text,
            origin=v.origin,
            fontname=v.fontname,
            font_size=v.font_size,
            confidence=v.confidence,
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
        from pydantic import ValidationError

        from .validation.schemas import SpanSchema

        try:
            v = SpanSchema.model_validate(d)
        except ValidationError as exc:
            raise ValueError(f"Invalid Span dict: {exc}") from exc

        return cls(
            token_indices=v.token_indices,
            col_id=v.col_id,
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
        from pydantic import ValidationError

        from .validation.schemas import LineSchema

        try:
            v = LineSchema.model_validate(d)
        except ValidationError as exc:
            raise ValueError(f"Invalid Line dict: {exc}") from exc

        spans = [Span(token_indices=s.token_indices, col_id=s.col_id) for s in v.spans]
        return cls(
            line_id=v.line_id,
            page=v.page,
            token_indices=v.token_indices,
            baseline_y=v.baseline_y,
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
        if not self.boxes:
            return (0, 0, 0, 0)
        xs0 = [b.x0 for b in self.boxes]
        ys0 = [b.y0 for b in self.boxes]
        xs1 = [b.x1 for b in self.boxes]
        ys1 = [b.y1 for b in self.boxes]
        return (min(xs0), min(ys0), max(xs1), max(ys1))

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dict."""
        return {
            "page": self.page,
            "boxes": [b.to_dict() for b in self.boxes],
            "column_id": self.column_id,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RowBand":
        """Deserialize from a dict produced by :meth:`to_dict`."""
        boxes = [GlyphBox.from_dict(b) for b in d.get("boxes", [])]
        return cls(
            page=d["page"],
            boxes=boxes,
            column_id=d.get("column_id"),
        )


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

    @classmethod
    def from_dict(cls, d: dict) -> "SuspectRegion":
        """Deserialize from a dict produced by :meth:`to_dict`."""
        bbox = d.get("bbox", [0, 0, 0, 0])
        return cls(
            page=d["page"],
            x0=bbox[0],
            y0=bbox[1],
            x1=bbox[2],
            y1=bbox[3],
            word_text=d.get("word_text", ""),
            context=d.get("context", ""),
            reason=d.get("reason", ""),
            source_label=d.get("source_label", ""),
            block_index=d.get("block_index", -1),
        )


# ── VOCR candidate detection ──────────────────────────────────────────

# Canonical trigger-method names used by vocr/candidates.py.
VOCR_TRIGGER_METHODS: Tuple[str, ...] = (
    "char_encoding_failure",
    "unmapped_glyph",
    "placeholder_token",
    "intraline_gap",
    "dense_cluster_hole",
    "baseline_style_gap",
    "template_adjacency",
    "regex_digit_pattern",
    "impossible_sequence",
    "vocab_trigger",
    "keyword_cooccurrence",
    "cross_ref_phrase",
    "near_duplicate_line",
    "font_subset_correlation",
    "token_width_anomaly",
    "vector_circle_near_number",
    "semantic_no_units",
    "dimension_geometry_proximity",
)


@dataclass
class VocrCandidate:
    """A small page region flagged for targeted VOCR scanning.

    Each candidate encodes *where* to look, *why* (trigger methods),
    and *what* we expect to find.  After targeted VOCR runs on the
    patch the ``outcome``, ``found_text`` and ``found_symbol`` fields
    are populated so that per-method hit-rate statistics can be computed.
    """

    page: int
    x0: float
    y0: float
    x1: float
    y1: float

    # Why this region was flagged — one or more trigger method names.
    trigger_methods: List[str] = field(default_factory=list)
    # Best guess of the missing symbol (e.g. "°", "±", "Ø").
    predicted_symbol: str = ""
    # Composite confidence in [0, 1] across all triggers.
    confidence: float = 0.5
    # Free-form context dict (neighbor text, gap size, font, etc.).
    context: dict = field(default_factory=dict)

    # Populated after targeted VOCR runs on this patch.
    outcome: str = "pending"  # "pending" | "hit" | "miss"
    found_text: str = ""
    found_symbol: str = ""

    # ── helpers ────────────────────────────────────────────────────

    def bbox(self) -> Tuple[float, float, float, float]:
        return (self.x0, self.y0, self.x1, self.y1)

    def patch_area(self) -> float:
        return max(0.0, self.x1 - self.x0) * max(0.0, self.y1 - self.y0)

    def to_dict(self) -> dict:
        return {
            "page": self.page,
            "bbox": [
                round(self.x0, 2),
                round(self.y0, 2),
                round(self.x1, 2),
                round(self.y1, 2),
            ],
            "trigger_methods": list(self.trigger_methods),
            "predicted_symbol": self.predicted_symbol,
            "confidence": round(self.confidence, 4),
            "context": dict(self.context),
            "outcome": self.outcome,
            "found_text": self.found_text,
            "found_symbol": self.found_symbol,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "VocrCandidate":
        from pydantic import ValidationError

        from .validation.schemas import VocrCandidateSchema

        try:
            v = VocrCandidateSchema.model_validate(d)
        except ValidationError as exc:
            raise ValueError(f"Invalid VocrCandidate dict: {exc}") from exc

        bbox = v.bbox
        return cls(
            page=v.page,
            x0=bbox[0],
            y0=bbox[1],
            x1=bbox[2],
            y1=bbox[3],
            trigger_methods=v.trigger_methods,
            predicted_symbol=v.predicted_symbol,
            confidence=v.confidence,
            context=v.context,
            outcome=v.outcome,
            found_text=v.found_text,
            found_symbol=v.found_symbol,
        )
