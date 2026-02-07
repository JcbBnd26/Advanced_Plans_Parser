"""Debug script to understand why B73 and B82 have unrelated content grouped together."""

import json

# Load boxes and blocks
with open(
    "runs/run_20260204_131148_IFC_Operations_Facil/artifacts/IFC_Operations_Facilities_McClain_County_-_Drawings_25_0915_page_2_boxes.json"
) as f:
    boxes = json.load(f)

with open(
    "runs/run_20260204_131148_IFC_Operations_Facil/artifacts/IFC_Operations_Facilities_McClain_County_-_Drawings_25_0915_page_2_blocks.json"
) as f:
    blocks = json.load(f)

# Find the blocks that become B73 and B82
regular_idx = 0
target_blocks = {}
for i, blk in enumerate(blocks):
    label = blk.get("label")
    is_table = blk.get("is_table", False)

    if label != "note_column_header" and not is_table:
        regular_idx += 1
        if regular_idx in [73, 82]:
            target_blocks[regular_idx] = (i, blk)

print("=" * 80)
print("ANALYSIS OF B73 AND B82")
print("=" * 80)

for b_idx, (raw_idx, blk) in target_blocks.items():
    bbox = blk["bbox"]
    print(f"\n{'='*80}")
    print(f"B{b_idx} (raw block index {raw_idx})")
    print(
        f"Block bbox: x0={bbox[0]:.1f}, y0={bbox[1]:.1f}, x1={bbox[2]:.1f}, y1={bbox[3]:.1f}"
    )
    print(f"Label: {blk.get('label')}, is_table: {blk.get('is_table')}")
    print(f"\nRows in this block:")

    for row_idx, row in enumerate(blk.get("rows", [])):
        row_bbox = row["bbox"]
        texts = row.get("texts", [])
        print(f"\n  Row {row_idx}: y0={row_bbox[1]:.1f}, y1={row_bbox[3]:.1f}")
        print(f"    Texts: {texts}")

        # Find the actual glyph boxes for each text in this row
        print(f"    Glyph boxes in this row:")
        row_y0, row_y1 = row_bbox[1], row_bbox[3]
        row_x0, row_x1 = row_bbox[0], row_bbox[2]

        matching_boxes = []
        for box in boxes:
            # Check if this box is in this row's bounding box
            box_y_center = (box["y0"] + box["y1"]) / 2
            if row_y0 <= box_y_center <= row_y1:
                if row_x0 - 5 <= box["x0"] <= row_x1 + 5:
                    matching_boxes.append(box)

        matching_boxes.sort(key=lambda b: b["x0"])
        for box in matching_boxes:
            print(f"      '{box['text']}' at x0={box['x0']:.1f}, y0={box['y0']:.1f}")

print("\n" + "=" * 80)
print("COLUMN ANALYSIS")
print("=" * 80)

# Look at what other content is in the same y-range as B73 and B82
for b_idx, (raw_idx, blk) in target_blocks.items():
    bbox = blk["bbox"]
    y0, y1 = bbox[1], bbox[3]

    print(f"\nOther blocks in same y-range as B{b_idx} (y={y0:.1f} to {y1:.1f}):")

    for i, other_blk in enumerate(blocks):
        other_bbox = other_blk["bbox"]
        other_y0, other_y1 = other_bbox[1], other_bbox[3]

        # Check for y-overlap
        if other_y0 <= y1 and other_y1 >= y0 and i != raw_idx:
            texts = []
            for row in other_blk.get("rows", []):
                texts.extend(row.get("texts", []))
            text_preview = " ".join(texts)[:60]
            print(
                f"  Block {i}: x0={other_bbox[0]:.1f}, label={other_blk.get('label')}, text='{text_preview}...'"
            )
