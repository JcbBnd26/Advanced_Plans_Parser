"""Section/region models: Legend, Abbreviation, Revision, StandardDetail, MiscTitle regions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .blocks import BlockCluster, HeaderTextMixin
from .geometry import _multi_bbox, _region_bbox
from .graphics import GraphicElement


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
