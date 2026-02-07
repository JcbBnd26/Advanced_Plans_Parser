"""Debug script to trace column partitioning"""

import json
import sys

sys.path.insert(0, "src")

from plancheck.config import GroupingConfig
from plancheck.grouping import (
    _median_size,
    _merge_note_number_columns,
    _partition_columns,
)
from plancheck.models import GlyphBox

with open(
    "runs/run_20260204_131148_IFC_Operations_Facil/artifacts/IFC_Operations_Facilities_McClain_County_-_Drawings_25_0915_page_2_boxes.json"
) as f:
    boxes_data = json.load(f)

# Convert to GlyphBox objects
boxes = [
    GlyphBox(
        x0=b["x0"], y0=b["y0"], x1=b["x1"], y1=b["y1"], text=b["text"], page=b["page"]
    )
    for b in boxes_data
]

median_w, median_h = _median_size(boxes)
settings = GroupingConfig()

print(f"Median width: {median_w:.1f}")
print(f"column_gap_mult: {settings.column_gap_mult}")
print(f"Gap threshold: {median_w * settings.column_gap_mult:.1f}")
print()

# Run column partitioning
columns = _partition_columns(boxes, median_w, settings)
columns = _merge_note_number_columns(columns)

print(f"Total columns detected: {len(columns)}")
print()

# Find which column contains "PRACTICE" and "REFERENCE"
for col_idx, (col_boxes, col_min, col_max) in enumerate(columns):
    practice_boxes = [b for b in col_boxes if b.text == "PRACTICE"]
    reference_boxes = [b for b in col_boxes if b.text == "REFERENCE"]

    if practice_boxes or reference_boxes:
        print(f"Column {col_idx}: x_range={col_min:.1f} to {col_max:.1f}")
        if practice_boxes:
            for b in practice_boxes:
                print(f"  Contains 'PRACTICE' at x0={b.x0:.1f}")
        if reference_boxes:
            for b in reference_boxes:
                print(f"  Contains 'REFERENCE' at x0={b.x0:.1f}")

# Also check where BMPR-0 is
print()
for col_idx, (col_boxes, col_min, col_max) in enumerate(columns):
    bmpr_boxes = [b for b in col_boxes if "BMPR" in b.text]
    matrix_boxes = [b for b in col_boxes if b.text == "MATRIX"]

    if bmpr_boxes or matrix_boxes:
        print(f"Column {col_idx}: x_range={col_min:.1f} to {col_max:.1f}")
        for b in bmpr_boxes:
            print(f"  Contains '{b.text}' at x0={b.x0:.1f}")
        for b in matrix_boxes:
            print(f"  Contains 'MATRIX' at x0={b.x0:.1f}")
