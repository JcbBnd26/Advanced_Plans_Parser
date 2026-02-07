"""Debug script to check specific row pairs."""
import json
from pathlib import Path
from src.plancheck.grouping import group_rows, GroupingConfig
from src.plancheck.models import GlyphBox

# Load boxes from latest run
base = Path("runs/run_20260204_210335_IFC_Operations_Facil/artifacts")
boxes_file = base / "IFC_Operations_Facilities_McClain_County_-_Drawings_25_0915_page_2_boxes.json"

with open(boxes_file) as f:
    boxes_data = json.load(f)

# Convert to GlyphBox objects
boxes = [GlyphBox(
    page=0, 
    x0=float(b["x0"]), 
    y0=float(b["y0"]), 
    x1=float(b["x1"]), 
    y1=float(b["y1"]), 
    text=b["text"]
) for b in boxes_data]

# Get page dimensions (approximate from boxes)
page_width = max(b.x1 for b in boxes) + 50
page_height = max(b.y1 for b in boxes) + 50

print(f"Loaded {len(boxes)} boxes")
print(f"Page size: {page_width:.0f} x {page_height:.0f}")

# Group into rows
settings = GroupingConfig()
rows = group_rows(boxes, settings)
print(f"Grouped into {len(rows)} rows")

# Check for specific text patterns that were previously split
print("\n" + "="*80)
print("SEARCHING FOR PREVIOUSLY SPLIT TEXT (should now be merged)")
print("="*80)

search_patterns = [
    "BEST MANAGEMENT PRACTICE",
    "REFERENCE MATRIX",
    "TEMPORARY EROSION",
    "SEDIMENT CONTROL",
    "SILT FENCE",
    "AND APPLICATIONS",
    "ASPHALT SURFACING",
    "CONSTRUCTION DETAILS",
    "STORM SEWER",
]

for pattern in search_patterns:
    for idx, row in enumerate(rows):
        text = " ".join(b.text for b in row.boxes)
        if pattern in text:
            bbox = row.bbox()
            print(f"\nR{idx+1} contains '{pattern}':")
            print(f"  Full text: {text[:100]}")
            print(f"  bbox: ({bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f})")
            print(f"  col_id: {row.column_id}")
            break
