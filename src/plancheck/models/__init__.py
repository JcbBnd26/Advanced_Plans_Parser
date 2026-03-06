"""Models package for plancheck data structures.

Re-exports all models for backward compatibility with:
    from plancheck.models import GlyphBox, BlockCluster, ...
"""

# Blocks
from .blocks import BlockCluster, HeaderTextMixin, NotesColumn

# Geometry utilities
from .geometry import (
    Bbox,
    _multi_bbox,
    _region_bbox,
    bbox_intersection_area,
    bbox_iou,
    bboxes_overlap,
    glyph_iou,
)

# Graphics
from .graphics import GraphicElement

# Quality/OCR
from .quality import VOCR_TRIGGER_METHODS, SuspectRegion, VocrCandidate

# Sections/Regions
from .sections import (
    AbbreviationEntry,
    AbbreviationRegion,
    LegendEntry,
    LegendRegion,
    MiscTitleRegion,
    RevisionEntry,
    RevisionRegion,
    StandardDetailEntry,
    StandardDetailRegion,
)

# Tokens
from .tokens import GlyphBox, Line, RowBand, Span

__all__ = [
    # Geometry
    "_region_bbox",
    "_multi_bbox",
    "Bbox",
    "bbox_intersection_area",
    "bbox_iou",
    "bboxes_overlap",
    "glyph_iou",
    # Graphics
    "GraphicElement",
    # Tokens
    "GlyphBox",
    "Span",
    "Line",
    "RowBand",
    # Blocks
    "HeaderTextMixin",
    "BlockCluster",
    "NotesColumn",
    # Sections
    "LegendEntry",
    "LegendRegion",
    "AbbreviationEntry",
    "AbbreviationRegion",
    "RevisionEntry",
    "RevisionRegion",
    "MiscTitleRegion",
    "StandardDetailEntry",
    "StandardDetailRegion",
    # Quality
    "SuspectRegion",
    "VocrCandidate",
    "VOCR_TRIGGER_METHODS",
]
