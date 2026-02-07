"""Find where the column boundary around x=1676 comes from"""

import json
from collections import defaultdict

with open(
    "runs/run_20260204_131148_IFC_Operations_Facil/artifacts/IFC_Operations_Facilities_McClain_County_-_Drawings_25_0915_page_2_boxes.json"
) as f:
    boxes = json.load(f)

# Look for gaps in x-range 1650-1700 across ALL rows
# The histogram gutter detection looks at the ENTIRE page

print("Looking for gaps around x=1650-1700 across ALL rows on the page:")
print("=" * 100)

# Group boxes by approximate y
rows = defaultdict(list)
for box in boxes:
    y_key = round(box["y0"] / 15) * 15  # bucket by ~15pt
    rows[y_key].append(box)

# For each row, check if there's content at x < 1650 AND x > 1700 with a gap in between
gaps_found = []
for y_key in sorted(rows.keys()):
    row_boxes = sorted(rows[y_key], key=lambda b: b["x0"])

    # Find boxes that end before 1660 and boxes that start after 1690
    ends_before = [b for b in row_boxes if b["x1"] < 1660 and b["x1"] > 1400]
    starts_after = [b for b in row_boxes if b["x0"] > 1690]

    if ends_before and starts_after:
        # There's potential for a gap
        last_before = max(ends_before, key=lambda b: b["x1"])
        first_after = min(starts_after, key=lambda b: b["x0"])
        gap = first_after["x0"] - last_before["x1"]
        if gap > 20:  # Significant gap
            gaps_found.append((y_key, gap, last_before["text"], first_after["text"]))

print("Rows with gaps > 20pts in x-range 1660-1690:")
for y, gap, before, after in gaps_found[:30]:
    print(
        "  y={:6.0f}: gap={:5.1f} between '{}' and '{}'".format(y, gap, before, after)
    )

print("\nTotal rows with gap in this range: {}".format(len(gaps_found)))

# Also check histogram - what x-values have low density?
print("\n" + "=" * 100)
print("X-center distribution of all boxes (histogram):")
x_centers = [(b["x0"] + b["x1"]) / 2 for b in boxes]
x_min, x_max = min(x_centers), max(x_centers)
print("X range: {:.1f} to {:.1f}".format(x_min, x_max))

# Check density in range 1650-1700
in_range = [x for x in x_centers if 1650 <= x <= 1700]
print("Boxes with center in 1650-1700: {}".format(len(in_range)))

# Compare to other ranges
in_1600_1650 = len([x for x in x_centers if 1600 <= x <= 1650])
in_1700_1750 = len([x for x in x_centers if 1700 <= x <= 1750])
print("Boxes with center in 1600-1650: {}".format(in_1600_1650))
print("Boxes with center in 1700-1750: {}".format(in_1700_1750))
