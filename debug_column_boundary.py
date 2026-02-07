"""Debug: Look at ALL boxes in the x-range 1600-1800 to understand column split"""

import json

with open(
    "runs/run_20260204_131148_IFC_Operations_Facil/artifacts/IFC_Operations_Facilities_McClain_County_-_Drawings_25_0915_page_2_boxes.json"
) as f:
    boxes = json.load(f)

# Find all boxes in x-range 1600-1800
target_boxes = [b for b in boxes if 1600 <= b["x0"] <= 1800 or 1600 <= b["x1"] <= 1800]
target_boxes.sort(key=lambda b: (b["y0"], b["x0"]))

print("Boxes in x-range 1600-1800 (potential column boundary area):")
print("=" * 100)

last_y = None
for box in target_boxes[:50]:  # First 50
    if last_y is not None and box["y0"] - last_y > 5:
        print()  # Add blank line between rows
    y0 = box["y0"]
    x0 = box["x0"]
    x1 = box["x1"]
    text = box["text"]
    print(f"y0={y0:7.1f}  x0={x0:7.1f}  x1={x1:7.1f}  text='{text}'")
    last_y = box["y0"]
