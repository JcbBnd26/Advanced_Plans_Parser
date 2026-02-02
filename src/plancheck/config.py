from dataclasses import dataclass


@dataclass
class GroupingConfig:
    """Tunables for geometry-first grouping."""

    # IoU threshold for pruning overlapping boxes.
    iou_prune: float = 0.5
    # Skew handling.
    enable_skew: bool = False
    max_skew_degrees: float = 3.0
    # Relative tolerances derived from median text size.
    horizontal_tol_mult: float = 1.2
    vertical_tol_mult: float = 0.45
    row_gap_mult: float = 1.0
    block_gap_mult: float = 0.9
    # Maximum block height relative to median row height; beyond this, start a new block.
    max_block_height_mult: float = 60.0
    # Horizontal gap multiplier to split a row when boxes are too far apart (prevents spanning whole page).
    row_split_gap_mult: float = 3.0
    # Gap between columns (in median word widths) to split page into columns before row grouping.
    column_gap_mult: float = 1.2
    # Enable histogram-based gutter detection before gap-based splitting.
    use_hist_gutter: bool = False
    # Minimum gutter width (in median word widths) for histogram-based column detection.
    gutter_width_mult: float = 2.0
    # Maximum allowed column width (in median word widths) before forcing an internal split.
    max_column_width_mult: float = 15.0
    # Max row width relative to its column span; rows wider than this are split on largest gaps.
    max_row_width_mult: float = 1.1
    # Table regularity tolerance (coefficient of variation threshold).
    table_regular_tol: float = 0.22
