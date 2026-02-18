"""Check character-level spacing to find micro-gap signals in fused words."""

import sys

sys.path.insert(0, "src")
import pdfplumber

pdf = pdfplumber.open(
    "input/IFC Operations Facilities McClain County - Drawings 25_0915.pdf"
)
page = pdf.pages[2]

# SIDEWALKCURB chars
chars = [
    c
    for c in page.chars
    if 1000 < float(c["top"]) < 1025 and 940 < float(c["x0"]) < 1060
]

print("=== Char-by-char advance analysis for SIDEWALKCURB ===")
for i in range(1, len(chars)):
    prev = chars[i - 1]
    cur = chars[i]
    gap = float(cur["x0"]) - float(prev["x1"])
    marker = " <-- GAP" if gap > 0.1 else ""
    print(
        f"  {prev['text']}->{cur['text']}: gap={gap:.3f}  adv_prev={float(prev['x1'])-float(prev['x0']):.2f}  adv_cur={float(cur['x1'])-float(cur['x0']):.2f}{marker}"
    )

# Also check: does pdfplumber have adv or width info per char?
print("\n=== Full char metadata for K and C at boundary ===")
for c in chars:
    if c["text"] in ("K", "C") and float(c["x0"]) > 1010:
        print(
            f"  '{c['text']}': x0={c['x0']}, x1={c['x1']}, width={c.get('width','N/A')}, adv={c.get('adv','N/A')}"
        )
        # Print all keys
        print(f"    keys: {sorted(c.keys())}")

pdf.close()
