"""Check raw characters around L95 area on correct page."""

import pdfplumber

pdf_path = "input/IFC Operations Facilities McClain County - Drawings 25_0915.pdf"
with pdfplumber.open(pdf_path) as pdf:
    page = pdf.pages[2]  # page 3 = index 2
    chars = page.chars

    print(f"Total chars on page: {len(chars)}")

    # All chars near y=1160-1180, x=970-1145
    print("\n=== All chars y=1155-1180, x=970-1145 ===")
    relevant = [
        c
        for c in chars
        if 1155 < (c["top"] + c["bottom"]) / 2 < 1180 and 970 < c["x0"] < 1145
    ]
    relevant.sort(key=lambda c: c["x0"])
    for c in relevant:
        ym = (c["top"] + c["bottom"]) / 2
        print(
            f"  x0={c['x0']:.2f} x1={c['x1']:.2f} ym={ym:.2f} char={repr(c['text'])} font={c.get('fontname','')} size={c.get('size','')}"
        )

    # Check for % chars anywhere near y=1150-1180
    print("\n=== % chars near y=1150-1180 ===")
    for c in chars:
        ym = (c["top"] + c["bottom"]) / 2
        if 1150 < ym < 1180 and c["text"] in ("%", "/"):
            print(
                f"  x0={c['x0']:.2f} x1={c['x1']:.2f} ym={ym:.2f} char={repr(c['text'])} font={c.get('fontname','')}"
            )

    # Show extract_words for this area
    print("\n=== extract_words y=1155-1180, x=970-1145 ===")
    words = page.extract_words()
    for w in words:
        ym = (w["top"] + w["bottom"]) / 2
        if 1155 < ym < 1180 and 970 < w["x0"] < 1145:
            print(
                f"  x0={w['x0']:.2f} x1={w['x1']:.2f} ym={ym:.2f} text={repr(w['text'])}"
            )
