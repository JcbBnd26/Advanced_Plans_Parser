"""Geometry-first clustering: rows → lines → blocks → notes columns."""

from .clustering import (
    build_clusters_v2,
    build_lines,
    compute_median_space_gap,
    flag_suspect_header_words,
    group_blocks,
    group_blocks_from_lines,
    group_notes_columns,
    group_rows,
    link_continued_columns,
    mark_headers,
    mark_notes,
    mark_tables,
    split_line_spans,
    split_wide_lines,
)

__all__ = [
    "build_clusters_v2",
    "build_lines",
    "compute_median_space_gap",
    "flag_suspect_header_words",
    "group_blocks",
    "group_blocks_from_lines",
    "group_notes_columns",
    "group_rows",
    "link_continued_columns",
    "mark_headers",
    "mark_notes",
    "mark_tables",
    "split_line_spans",
    "split_wide_lines",
]
