"""Debug script to look at Block 80 and understand the ODOT Standard Details area."""

import json

with open(
    "runs/run_20260204_131148_IFC_Operations_Facil/artifacts/IFC_Operations_Facilities_McClain_County_-_Drawings_25_0915_page_2_blocks.json"
) as f:
    blocks = json.load(f)

# Block 80 is the ODOT Standard Details content
blk = blocks[80]
bbox = blk["bbox"]
print(f"Block 80 (ODOT Standard Details content)")
print(
    f"Block bbox: x0={bbox[0]:.1f}, y0={bbox[1]:.1f}, x1={bbox[2]:.1f}, y1={bbox[3]:.1f}"
)
print(f"Label: {blk.get('label')}, is_table: {blk.get('is_table')}")
print(f"\nRows in this block:")

for row_idx, row in enumerate(blk.get("rows", [])):
    row_bbox = row["bbox"]
    texts = row.get("texts", [])
    print(f"  Row {row_idx}: y0={row_bbox[1]:.1f}, texts={texts}")

# Now look at Block 81 (B73)
print("\n" + "=" * 80)
blk = blocks[81]
bbox = blk["bbox"]
print(f"Block 81 (B73)")
print(
    f"Block bbox: x0={bbox[0]:.1f}, y0={bbox[1]:.1f}, x1={bbox[2]:.1f}, y1={bbox[3]:.1f}"
)
print(f"\nRows:")
for row_idx, row in enumerate(blk.get("rows", [])):
    row_bbox = row["bbox"]
    texts = row.get("texts", [])
    print(f"  Row {row_idx}: y0={row_bbox[1]:.1f}, x0={row_bbox[0]:.1f}, texts={texts}")

# Look at what's between Block 80's x1 and Block 81's x0
print("\n" + "=" * 80)
print("SPATIAL ANALYSIS:")
print(f"Block 80 x-range: {blocks[80]['bbox'][0]:.1f} to {blocks[80]['bbox'][2]:.1f}")
print(f"Block 81 x-range: {blocks[81]['bbox'][0]:.1f} to {blocks[81]['bbox'][2]:.1f}")
print(f"Gap between them: {blocks[81]['bbox'][0] - blocks[80]['bbox'][2]:.1f} pts")

print("\n" + "=" * 80)
print("Block 80 rows vs Block 81 rows - do they align?")
for i, row80 in enumerate(blocks[80].get("rows", [])[:3]):
    y80 = row80["bbox"][1]
    for j, row81 in enumerate(blocks[81].get("rows", [])):
        y81 = row81["bbox"][1]
        if abs(y80 - y81) < 2:
            print(
                f"  Row {i} of Block 80 (y={y80:.1f}) aligns with Row {j} of Block 81 (y={y81:.1f})"
            )
            print(f"    Block 80: {row80['texts']}")
            print(f"    Block 81: {row81['texts']}")
