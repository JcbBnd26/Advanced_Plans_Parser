"""Token-level text models: GlyphBox, Span, Line, RowBand."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


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

        from ..validation.schemas import GlyphBoxSchema

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

        from ..validation.schemas import SpanSchema

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

    Accumulator fields (_y_sum, _y0_min, _y1_max, _count) enable O(1)
    incremental updates during line building instead of O(k) per comparison.
    """

    line_id: int
    page: int
    token_indices: List[int] = field(default_factory=list)
    baseline_y: float = 0.0
    spans: List[Span] = field(default_factory=list)

    # Accumulators for incremental line building (not serialized)
    _y_sum: float = field(default=0.0, repr=False)
    _y0_min: float = field(default=float("inf"), repr=False)
    _y1_max: float = field(default=float("-inf"), repr=False)
    _count: int = field(default=0, repr=False)

    @property
    def y_center(self) -> float:
        """Return cached y-center from accumulators, or 0 if empty."""
        return self._y_sum / self._count if self._count > 0 else 0.0

    def update_bounds(self, token: "GlyphBox") -> None:
        """Incrementally update accumulators when a token is added."""
        y_center = (token.y0 + token.y1) * 0.5
        self._y_sum += y_center
        self._y0_min = min(self._y0_min, token.y0)
        self._y1_max = max(self._y1_max, token.y1)
        self._count += 1

    def init_bounds(self, token: "GlyphBox") -> None:
        """Initialize accumulators with the first token."""
        y_center = (token.y0 + token.y1) * 0.5
        self._y_sum = y_center
        self._y0_min = token.y0
        self._y1_max = token.y1
        self._count = 1

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

        from ..validation.schemas import LineSchema

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
