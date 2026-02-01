import pdfplumber

with pdfplumber.open(
    "input/IFC Operations Facilities McClain County - Drawings 25_0915.pdf"
) as pdf:
    page = pdf.pages[2]

    print(f"Page width: {page.width}")
    print(f"Page height: {page.height}")
    print(f"Page bbox: {page.bbox}")
    print(f"Page mediabox: {page.mediabox}")
    print()

    # Get ALL characters in the TRANSPORTATION area
    chars = page.chars
    transport_chars = [
        c for c in chars if c["top"] > 65 and c["top"] < 85 and c["x0"] > 2350
    ]

    print(f"Found {len(transport_chars)} characters in area")
    print()
    print("Full character details:")
    for c in sorted(transport_chars, key=lambda x: x["x0"]):
        print(f"  Char: '{c['text']}' (ord={ord(c['text']) if c['text'] else 'N/A'})")
        print(f"    x0={c['x0']:.2f}, x1={c['x1']:.2f}, width={c['x1']-c['x0']:.2f}")
        print(f"    top={c['top']:.2f}, bottom={c['bottom']:.2f}")
        print(f"    fontname={c.get('fontname', 'N/A')}")
        print(f"    size={c.get('size', 'N/A')}")
        print(f"    adv={c.get('adv', 'N/A')}")  # advance width
        print()

    # Check if there are any characters with x1 > page.width
    offpage_chars = [c for c in chars if c["x1"] > page.width]
    print(f"\nCharacters extending past page width ({page.width}):")
    print(f"  Count: {len(offpage_chars)}")
    if offpage_chars:
        for c in offpage_chars[:10]:  # Show first 10
            print(
                f"  '{c['text']}' at x1={c['x1']:.2f} (overhang={c['x1']-page.width:.2f})"
            )
