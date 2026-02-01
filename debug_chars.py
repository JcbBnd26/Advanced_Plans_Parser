import pdfplumber

with pdfplumber.open(
    "input/IFC Operations Facilities McClain County - Drawings 25_0915.pdf"
) as pdf:
    page = pdf.pages[2]

    # Get individual characters
    chars = page.chars
    transport_chars = [
        c for c in chars if c["top"] > 65 and c["top"] < 85 and c["x0"] > 2360
    ]

    print("Characters in TRANSPORTATION area:")
    for c in sorted(transport_chars, key=lambda x: x["x0"]):
        print(f"  '{c['text']}' x0={c['x0']:.1f} x1={c['x1']:.1f}")

    if transport_chars:
        last_char = max(transport_chars, key=lambda x: x["x1"])
        print(
            f"\nLast character: '{last_char['text']}' ends at x1={last_char['x1']:.1f}"
        )
