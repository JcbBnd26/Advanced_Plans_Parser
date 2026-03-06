"""Quality/OCR models: SuspectRegion, VocrCandidate."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

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

        from ..validation.schemas import VocrCandidateSchema

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
