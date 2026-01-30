"""Core geometry-first grouping package for plan checking."""

from .config import GroupingConfig
from .grouping import build_clusters, group_blocks, group_rows, mark_tables
from .models import BlockCluster, GlyphBox, RowBand
from .overlay import draw_overlay
from .preprocess import estimate_skew_degrees, nms_prune, rotate_boxes
from .zoning import Region, whole_page

__all__ = [
    "GroupingConfig",
    "GlyphBox",
    "RowBand",
    "BlockCluster",
    "nms_prune",
    "estimate_skew_degrees",
    "rotate_boxes",
    "draw_overlay",
    "group_rows",
    "group_blocks",
    "mark_tables",
    "build_clusters",
    "Region",
    "whole_page",
]
