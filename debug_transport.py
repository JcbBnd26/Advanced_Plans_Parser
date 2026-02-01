import pdfplumber

with pdfplumber.open(
    "input/IFC Operations Facilities McClain County - Drawings 25_0915.pdf"
) as pdf:
    page = pdf.pages[2]
    print(f"Page width: {page.width}")
    print(f"Page bbox: {page.bbox}")
    print()

    words = page.extract_words()
    for w in words:
        if "TRANSPORT" in w["text"].upper():
            print(f"Word: '{w['text']}'")
            print(f"  x0={w['x0']:.2f}, x1={w['x1']:.2f}")
            print(f"  top={w['top']:.2f}, bottom={w['bottom']:.2f}")
            print(f"  Width of word: {w['x1'] - w['x0']:.2f}")
            print(f"  Extends past page? {w['x1'] > page.width}")
            print()
