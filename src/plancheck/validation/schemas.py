"""Pydantic schemas for validating serialized plancheck model dicts.

These schemas intentionally use ``extra='ignore'`` so older/newer payloads can
round-trip without breaking.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class GlyphBoxSchema(BaseModel):
    model_config = ConfigDict(extra="ignore")

    page: int = Field(ge=0)
    x0: float
    y0: float
    x1: float
    y1: float
    text: str = ""
    origin: str = "text"
    fontname: str = ""
    font_size: float = 0.0
    confidence: float = 1.0


class SpanSchema(BaseModel):
    model_config = ConfigDict(extra="ignore")

    token_indices: List[int] = Field(default_factory=list)
    col_id: Optional[int] = None


class LineSchema(BaseModel):
    model_config = ConfigDict(extra="ignore")

    line_id: int
    page: int = Field(ge=0)
    token_indices: List[int] = Field(default_factory=list)
    baseline_y: float = 0.0
    spans: List[SpanSchema] = Field(default_factory=list)


class VocrCandidateSchema(BaseModel):
    model_config = ConfigDict(extra="ignore")

    page: int = Field(default=0, ge=0)
    bbox: List[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])
    trigger_methods: List[str] = Field(default_factory=list)
    predicted_symbol: str = ""
    confidence: float = 0.5
    context: Dict[str, Any] = Field(default_factory=dict)
    outcome: str = "pending"
    found_text: str = ""
    found_symbol: str = ""

    @field_validator("bbox")
    @classmethod
    def _bbox_len_4(cls, v: List[float]) -> List[float]:
        if len(v) != 4:
            raise ValueError("bbox must have exactly 4 numbers")
        return v
