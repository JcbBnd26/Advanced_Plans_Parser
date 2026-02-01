import json

data = json.load(
    open(
        "runs/run_20260201_100543_abbrev_final_p2/artifacts/IFC_Operations_Facilities_McClain_County_-_Drawings_25_0915_page_2_blocks.json"
    )
)
blocks = [
    b
    for b in data
    if b["bbox"][0] > 2140 and b["bbox"][0] < 2220 and b["bbox"][1] > 200
]

print("Blocks in abbreviation x-range:")
for b in blocks:
    print(
        f"  x0={b['bbox'][0]:.1f} y0={b['bbox'][1]:.1f} y1={b['bbox'][3]:.1f} rows={len(b['rows'])} last_row={b['rows'][-1]['texts']}"
    )

# Also check for blocks with abbreviation-like content (short codes)
print("\nAbbreviation region from debug: (2139.2, 223.3, 2448.0, 835.9)")
print("This ends at y=835.9, but blocks continue below...")
