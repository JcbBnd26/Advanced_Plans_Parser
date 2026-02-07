"""Find which page has SIDEWALK text."""

import pdfplumber

pdf_path = "input/IFC Operations Facilities McClain County - Drawings 25_0915.pdf"
with pdfplumber.open(pdf_path) as pdf:
    for pi in [0, 1, 2]:
        page = pdf.pages[pi]
        words = page.extract_words()
        sw = [w for w in words if "SIDEWALK" in w["text"]]
        print(
            f"Page index {pi} (page {pi+1}): {len(words)} words, SIDEWALK matches: {len(sw)}"
        )
        for w in sw:
            ym = (w["top"] + w["bottom"]) / 2
            print(f"  x0={w['x0']:.2f} ym={ym:.2f} text={repr(w['text'])}")
