import sys

sys.path.insert(0, "src")
import pdfplumber

# Get page
with pdfplumber.open(
    "input/IFC Operations Facilities McClain County - Drawings 25_0915.pdf"
) as pdf:
    page = pdf.pages[2]
    words = page.extract_words()

# Look for TRANSPORTATION
for w in words:
    if "TRANSPORT" in w["text"].upper():
        print(
            f"Word: '{w['text']}' x0={w['x0']:.1f} x1={w['x1']:.1f} top={w['top']:.1f} bottom={w['bottom']:.1f}"
        )
