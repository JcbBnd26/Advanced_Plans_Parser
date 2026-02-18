import sys

sys.path.insert(0, "src")
import pdfplumber

from plancheck import GroupingConfig

cfg = GroupingConfig()
with pdfplumber.open(
    "input/IFC Operations Facilities McClain County - Drawings 25_0915.pdf"
) as pdf:
    page = pdf.pages[2]

    # Words near SIDEWALK/CURB header (y~1007, x~953)
    words = page.extract_words(
        x_tolerance=cfg.tocr_x_tolerance,
        y_tolerance=cfg.tocr_y_tolerance,
        extra_attrs=["fontname", "size"],
    )
    print("=== Words with default x_tolerance ===")
    for w in words:
        if 1000 < float(w["top"]) < 1025 and 940 < float(w["x0"]) < 1200:
            print(f"  '{w['text']}' x0={float(w['x0']):.1f} x1={float(w['x1']):.1f}")

    # With x_tolerance=1
    words2 = page.extract_words(x_tolerance=1, y_tolerance=cfg.tocr_y_tolerance)
    print("\n=== Words with x_tolerance=1 ===")
    for w in words2:
        if 1000 < float(w["top"]) < 1025 and 940 < float(w["x0"]) < 1200:
            print(f"  '{w['text']}' x0={float(w['x0']):.1f} x1={float(w['x1']):.1f}")

    # Raw characters
    print("\n=== Raw chars ===")
    for c in page.chars:
        if 1000 < float(c["top"]) < 1025 and 940 < float(c["x0"]) < 1200:
            print(
                f"  '{c['text']}' x0={float(c['x0']):.1f} x1={float(c['x1']):.1f} font={c.get('fontname','')}"
            )
