"""Core geometry-first grouping package for plan checking."""

from .config import GroupingConfig
from .grouping import (
    build_clusters,
    build_clusters_v2,
    build_lines,
    group_blocks,
    group_blocks_from_lines,
    group_rows,
    mark_notes,
    mark_tables,
)
from .models import BlockCluster, GlyphBox, Line, RowBand, Span
from .overlay import draw_overlay
from .preprocess import estimate_skew_degrees, nms_prune, rotate_boxes
from .zoning import Region, whole_page

__all__ = [
    "GroupingConfig",
    "GlyphBox",
    "RowBand",
    "Line",
    "Span",
    "BlockCluster",
    "nms_prune",
    "estimate_skew_degrees",
    "rotate_boxes",
    "draw_overlay",
    "group_rows",
    "group_blocks",
    "group_blocks_from_lines",
    "build_lines",
    "mark_tables",
    "build_clusters",
    "build_clusters_v2",
    "Region",
    "whole_page",
]
