import pdfplumber

with pdfplumber.open(
    "input/IFC Operations Facilities McClain County - Drawings 25_0915.pdf"
) as pdf:
    page = pdf.pages[2]

    print("Page boxes:")
    print(f"  bbox: {page.bbox}")
    print(f"  mediabox: {page.mediabox}")

    # Check for cropbox, trimbox, artbox, bleedbox
    page_obj = page.page_obj
    print(
        f"\nRaw page object keys: {list(page_obj.keys()) if hasattr(page_obj, 'keys') else 'N/A'}"
    )

    # Try to access various box types
    for box_type in ["MediaBox", "CropBox", "TrimBox", "ArtBox", "BleedBox"]:
        try:
            box = page_obj.get(box_type)
            if box:
                print(f"  {box_type}: {box}")
        except:
            pass

    # Check if there's clipping or transformation
    print(f"\nPage rotation: {page.rotation}")

    # Let's also check how many total characters extend past page
    chars = page.chars
    offpage_chars = [c for c in chars if c["x1"] > page.width]
    print(f"\nTotal characters past page width: {len(offpage_chars)}")

    # Group by word
    words_offpage = set()
    for c in offpage_chars:
        # Find nearby chars to form word
        nearby = [
            ch["text"]
            for ch in chars
            if abs(ch["top"] - c["top"]) < 2 and abs(ch["x0"] - c["x0"]) < 150
        ]
        words_offpage.add("".join(nearby))

    print(f"Unique words with chars past page: {words_offpage}")

    # Check the rightmost characters on the page
    rightmost = sorted(chars, key=lambda x: x["x1"], reverse=True)[:20]
    print(f"\n20 rightmost characters:")
    for c in rightmost:
        status = "OFF PAGE" if c["x1"] > page.width else ""
        print(f"  '{c['text']}' x1={c['x1']:.1f} y={c['top']:.1f} {status}")
