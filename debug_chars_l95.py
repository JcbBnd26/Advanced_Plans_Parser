"""Check raw characters around L95 area."""

import pdfplumber

pdf_path = "input/IFC Operations Facilities McClain County - Drawings 25_0915.pdf"
with pdfplumber.open(pdf_path) as pdf:
    page = pdf.pages[1]  # page index 1 = page 2

    # Find all chars near y=1160-1175 (L95 baseline area in page_boxes coords)
    # page_boxes uses 'top'/'bottom' directly, so these coords should match
    chars = page.chars

    print(f"Total chars on page: {len(chars)}")

    # Find chars with text '2' near x=980
    print("\n=== Chars with '2' near x=975-995 ===")
    for c in chars:
        ym = (c["top"] + c["bottom"]) / 2
        if c["text"] == "2" and 975 < c["x0"] < 995:
            print(
                f"  x0={c['x0']:.2f} x1={c['x1']:.2f} top={c['top']:.2f} bot={c['bottom']:.2f} ym={ym:.2f}"
            )

    # What about extracting with different settings?
    print("\n=== extract_words near area with extra_attrs ===")
    words = page.extract_words(
        keep_blank_chars=False, use_text_flow=False, extra_attrs=["fontname", "size"]
    )
    for w in words:
        ym = (w["top"] + w["bottom"]) / 2
        if 1155 < ym < 1180 and 970 < w["x0"] < 1145:
            print(
                f"  x0={w['x0']:.2f} x1={w['x1']:.2f} ym={ym:.2f} text={repr(w['text'])} font={w.get('fontname','')} size={w.get('size','')}"
            )

    # Look for ANY char between x=987 and x=991 (between '2' and 'AND')
    print("\n=== Any char between x=985-995 near y=1160-1180 ===")
    for c in chars:
        ym = (c["top"] + c["bottom"]) / 2
        if 1155 < ym < 1180 and 985 < c["x0"] < 995:
            print(
                f"  x0={c['x0']:.2f} x1={c['x1']:.2f} ym={ym:.2f} char={repr(c['text'])} font={c.get('fontname','')}"
            )

    # Look for ANY char between x=1131 and x=1145 (after '12.')
    print("\n=== Any char between x=1125-1145 near y=1155-1180 ===")
    for c in chars:
        ym = (c["top"] + c["bottom"]) / 2
        if 1155 < ym < 1180 and 1125 < c["x0"] < 1145:
            print(
                f"  x0={c['x0']:.2f} x1={c['x1']:.2f} ym={ym:.2f} char={repr(c['text'])} font={c.get('fontname','')}"
            )

    relevant.sort(key=lambda c: c["x0"])
    for c in relevant:
        ym = (c["top"] + c["bottom"]) / 2
        print(
            f"  x0={c['x0']:.2f} x1={c['x1']:.2f} ym={ym:.2f} size={c.get('size',0):.1f} font={c.get('fontname','?')} char={repr(c['text'])}"
        )
