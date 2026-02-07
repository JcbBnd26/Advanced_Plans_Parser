"""Debug script to understand row splitting at y=1164.7"""

import json

with open(
    "runs/run_20260204_131148_IFC_Operations_Facil/artifacts/IFC_Operations_Facilities_McClain_County_-_Drawings_25_0915_page_2_boxes.json"
) as f:
    boxes = json.load(f)

# Find all boxes at y ~ 1164.7
print("All glyph boxes at y ~ 1164.7 (first row of ODOT Standard Details):")
print("=" * 100)

target_boxes = []
for box in boxes:
    if 1160 < box["y0"] < 1170:
        target_boxes.append(box)

target_boxes.sort(key=lambda b: b["x0"])

for box in target_boxes:
    print(f"  x0={box['x0']:7.1f}  x1={box['x1']:7.1f}  text='{box['text']}'")

# Look for gaps
print("\n" + "=" * 100)
print("Gaps between consecutive boxes:")
for i in range(len(target_boxes) - 1):
    gap = target_boxes[i + 1]["x0"] - target_boxes[i]["x1"]
    print(
        f"  Between '{target_boxes[i]['text']}' and '{target_boxes[i+1]['text']}': {gap:.1f} pts"
    )

# Find the median width for context
widths = [b["x1"] - b["x0"] for b in boxes]
from statistics import median

med_w = median(widths)
print(f"\nMedian glyph width: {med_w:.1f}")
print(f"Typical column gap threshold (median_w * column_gap_mult): {med_w * 6:.1f}")
