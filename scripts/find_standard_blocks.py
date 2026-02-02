"""Find blocks containing ODOT STANDARD in IFC page 2."""

import json

with open(
    "runs/run_20260201_182252_IFC_test3/artifacts/IFC_Operations_Facilities_McClain_County_-_Drawings_25_0915_page_2_blocks.json"
) as f:
    data = json.load(f)

# Handle both dict and list formats
blocks = data["blocks"] if isinstance(data, dict) else data

# Block 81 and 82
for i, blk in enumerate(blocks):
    if blk["bbox"][1] > 1160 and blk["bbox"][1] < 1380 and blk["bbox"][0] > 1300:
        print(f"Block {i}, bbox={blk['bbox']}")
        for j, row in enumerate(blk["rows"]):
            row_text = " ".join(row.get("texts", []))
            print(f"  Row {j}: {row_text[:80]}")
        print()
