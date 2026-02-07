import json

with open(
    "runs/run_20260204_131148_IFC_Operations_Facil/artifacts/IFC_Operations_Facilities_McClain_County_-_Drawings_25_0915_page_2_blocks.json"
) as f:
    blocks = json.load(f)

regular_idx = 0
for i, blk in enumerate(blocks):
    label = blk.get("label")
    is_table = blk.get("is_table", False)

    # Regular blocks: not header, not table
    if label != "note_column_header" and not is_table:
        regular_idx += 1
        if regular_idx in [73, 74, 75, 80, 81, 82, 83]:
            texts = []
            for row in blk.get("rows", []):
                texts.extend(row.get("texts", []))
            text_preview = " ".join(texts)[:100]
            bbox = blk["bbox"]
            print(
                f"B{regular_idx} (raw idx {i}): label={label}, bbox=({bbox[0]:.1f}, {bbox[1]:.1f})"
            )
            print(f"  Text: {text_preview}...")
            print()

print(f"Total regular blocks: {regular_idx}")
