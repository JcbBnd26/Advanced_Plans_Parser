"""Debug: Look at ALL boxes in ODOT Standard Details area (y > 1100) around x=1650-1750"""

import json

with open(
    "runs/run_20260204_131148_IFC_Operations_Facil/artifacts/IFC_Operations_Facilities_McClain_County_-_Drawings_25_0915_page_2_boxes.json"
) as f:
    boxes = json.load(f)

# Find all boxes in ODOT area with x in potential boundary range
target_boxes = [b for b in boxes if b["y0"] > 1100 and 1550 <= b["x0"] <= 1850]
target_boxes.sort(key=lambda b: (b["y0"], b["x0"]))

print("Boxes in ODOT Standard Details area (y>1100) x-range 1550-1850:")
print("=" * 100)

last_y = None
for box in target_boxes[:60]:
    if last_y is not None and box["y0"] - last_y > 5:
        print()  # Add blank line between rows
    y0 = box["y0"]
    x0 = box["x0"]
    x1 = box["x1"]
    text = box["text"]
    print("y0={:7.1f}  x0={:7.1f}  x1={:7.1f}  text='{}'".format(y0, x0, x1, text))
    last_y = box["y0"]

# Now look specifically at the gaps on the ODOT rows
print("\n" + "=" * 100)
print("Full rows in ODOT area with all boxes:")
print("=" * 100)

from collections import defaultdict

rows = defaultdict(list)
for box in boxes:
    if 1160 < box["y0"] < 1350 and box["x0"] > 1300:
        y_key = round(box["y0"] * 10) / 10  # round to .1
        rows[y_key].append(box)

for y_key in sorted(rows.keys())[:12]:
    row_boxes = sorted(rows[y_key], key=lambda b: b["x0"])
    print("\nRow at y={:.1f}:".format(y_key))
    for i, b in enumerate(row_boxes):
        gap = ""
        if i < len(row_boxes) - 1:
            gap_val = row_boxes[i + 1]["x0"] - b["x1"]
            gap = " -> gap={:.1f}".format(gap_val)
        print(
            "  x0={:7.1f}  x1={:7.1f}  '{}' {}".format(b["x0"], b["x1"], b["text"], gap)
        )
