"""Profile TOCR stage on a real PDF — measures per-page and per-phase timings."""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

PDF_PATH = Path(
    r"C:\Users\jake\OneDrive\Desktop\IFC Operations Facilities McClain County - Drawings 25_0915.pdf"
)
PAGES_TO_PROFILE = [0, 1, 2]  # first 3 pages
RESOLUTION = 200


def profile_page(pdf_path: Path, page_num: int) -> dict:
    """Profile a single page through the TOCR pipeline."""
    from plancheck.config import GroupingConfig
    from plancheck.tocr.extract import build_extract_words_kwargs

    cfg = GroupingConfig()
    extract_kwargs = build_extract_words_kwargs(cfg, mode="full")

    timings = {}

    # Phase A: Open PDF + extract text + render image (ingest)
    t0 = time.perf_counter()
    import pdfplumber

    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[page_num]
        page_w = float(page.width)
        page_h = float(page.height)

        t1 = time.perf_counter()
        words = page.extract_words(**extract_kwargs)
        t2 = time.perf_counter()

        chars = list(page.chars)
        t3 = time.perf_counter()

        lines = list(page.lines)
        rects = list(page.rects)
        curves = list(page.curves)
        t4 = time.perf_counter()

        from PIL import Image

        bg_img = page.to_image(resolution=RESOLUTION).original.copy()
        if bg_img.mode != "RGB":
            bg_img = bg_img.convert("RGB")
        t5 = time.perf_counter()

    timings["pdf_open_page"] = (t1 - t0) * 1000
    timings["extract_words"] = (t2 - t1) * 1000
    timings["extract_chars"] = (t3 - t2) * 1000
    timings["extract_geometry"] = (t4 - t3) * 1000
    timings["render_image"] = (t5 - t4) * 1000
    timings["ingest_total"] = (t5 - t0) * 1000

    # Phase B: TOCR extraction (words → GlyphBoxes)
    from plancheck.tocr.extract import extract_tocr_from_words

    t6 = time.perf_counter()
    result = extract_tocr_from_words(words, page_num, page_w, page_h, cfg, mode="full")
    t7 = time.perf_counter()
    timings["tocr_extract"] = (t7 - t6) * 1000
    timings["tocr_token_count"] = len(result.tokens)

    # Phase C: Vector symbol recovery
    from plancheck.tocr.vector_symbols import recover_vector_symbols

    t8 = time.perf_counter()
    if cfg.tocr_vector_symbols_enabled:
        enriched, vs_diag = recover_vector_symbols(
            result.tokens, lines, curves, page_num, cfg
        )
    else:
        enriched = result.tokens
        vs_diag = {}
    t9 = time.perf_counter()
    timings["vector_symbols"] = (t9 - t8) * 1000
    timings["symbols_found"] = vs_diag.get("vector_symbols_found", 0)

    # Phase D: NMS prune + deskew
    from plancheck.tocr.preprocess import estimate_skew_degrees, nms_prune, rotate_boxes

    t10 = time.perf_counter()
    pruned = nms_prune(enriched, cfg.iou_prune)
    t11 = time.perf_counter()

    if cfg.enable_skew:
        skew = estimate_skew_degrees(pruned, cfg.max_skew_degrees)
        rotated = rotate_boxes(pruned, skew, page_w, page_h)
    else:
        skew = 0.0
        rotated = pruned
    t12 = time.perf_counter()

    timings["nms_prune"] = (t11 - t10) * 1000
    timings["deskew"] = (t12 - t11) * 1000
    timings["pruned_from"] = len(enriched)
    timings["pruned_to"] = len(pruned)
    timings["skew_degrees"] = skew

    timings["total"] = (t12 - t0) * 1000

    return timings


def main():
    print(f"PDF: {PDF_PATH.name}")
    print(f"Pages: {PAGES_TO_PROFILE}")
    print(f"Resolution: {RESOLUTION} DPI")
    print("=" * 70)

    # Get page count
    import pdfplumber

    with pdfplumber.open(PDF_PATH) as pdf:
        total_pages = len(pdf.pages)
    print(f"Total pages in PDF: {total_pages}")
    print()

    word_counts = {}
    for pg in PAGES_TO_PROFILE:
        print(f"--- Page {pg} ---")
        t = profile_page(PDF_PATH, pg)
        word_counts[pg] = t["tocr_token_count"]

        print(f"  Ingest total:     {t['ingest_total']:7.0f}ms")
        print(f"    PDF open+page:  {t['pdf_open_page']:7.0f}ms")
        print(f"    extract_words:  {t['extract_words']:7.0f}ms")
        print(f"    extract_chars:  {t['extract_chars']:7.0f}ms")
        print(f"    extract_geom:   {t['extract_geometry']:7.0f}ms")
        print(f"    render image:   {t['render_image']:7.0f}ms")
        print(
            f"  TOCR extract:     {t['tocr_extract']:7.0f}ms  ({t['tocr_token_count']} tokens)"
        )
        print(
            f"  Vector symbols:   {t['vector_symbols']:7.0f}ms  ({t['symbols_found']} found)"
        )
        print(
            f"  NMS prune:        {t['nms_prune']:7.0f}ms  ({t['pruned_from']}->{t['pruned_to']})"
        )
        print(
            f"  Deskew:           {t['deskew']:7.0f}ms  (skew={t['skew_degrees']:.3f} deg)"
        )
        print(f"  TOTAL:            {t['total']:7.0f}ms")
        print()

    # Now time a full document run (for comparison-sake)
    print("=" * 70)
    print(f"Estimated full-doc TOCR time ({total_pages} pages):")
    avg_total = sum(
        profile_page(PDF_PATH, pg)["total"] for pg in PAGES_TO_PROFILE
    ) / len(PAGES_TO_PROFILE)
    print(f"  Average per page: {avg_total:.0f}ms")
    print(f"  Estimated total:  {avg_total * total_pages / 1000:.1f}s")

    # ── Single-open comparison ───────────────────────────────────
    print()
    print("=" * 70)
    print("Single-open comparison (pages processed from one pdfplumber handle):")
    with pdfplumber.open(PDF_PATH) as pdf:
        from plancheck.config import GroupingConfig as GC
        from plancheck.ingest import build_page_context
        from plancheck.tocr.extract import build_extract_words_kwargs as bew

        cfg2 = GC()
        kw = bew(cfg2, mode="full")
        for pg in PAGES_TO_PROFILE:
            t0 = time.perf_counter()
            _ctx = build_page_context(
                PDF_PATH, pg,
                overlay_resolution=RESOLUTION,
                extract_words_kwargs=kw,
                _pdf=pdf,
            )
            t1 = time.perf_counter()
            print(f"  Page {pg}: {(t1 - t0) * 1000:.0f}ms (single-open ingest)")


if __name__ == "__main__":
    main()
