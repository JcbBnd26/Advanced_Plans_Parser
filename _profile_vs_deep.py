"""Deep-profile vector_symbols stage to find remaining hotspots."""
from __future__ import annotations

import cProfile
import pstats
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

PDF_PATH = Path(
    r"C:\Users\jake\OneDrive\Desktop\IFC Operations Facilities McClain County - Drawings 25_0915.pdf"
)
PAGE = 2  # heaviest page


def get_page_data():
    """Extract tokens, lines, curves from page."""
    import pdfplumber
    from plancheck.config import GroupingConfig
    from plancheck.tocr.extract import build_extract_words_kwargs, extract_tocr_from_words

    cfg = GroupingConfig()
    extract_kwargs = build_extract_words_kwargs(cfg, mode="full")

    with pdfplumber.open(PDF_PATH) as pdf:
        page = pdf.pages[PAGE]
        words = page.extract_words(**extract_kwargs)
        lines = list(page.lines)
        curves = list(page.curves)
        page_w = float(page.width)
        page_h = float(page.height)

    result = extract_tocr_from_words(words, PAGE, page_w, page_h, cfg, mode="full")
    return result.tokens, lines, curves, cfg


def main():
    tokens, lines, curves, cfg = get_page_data()
    print(f"Page {PAGE}: {len(tokens)} tokens, {len(lines)} lines, {len(curves)} curves")

    from plancheck.tocr.vector_symbols import recover_vector_symbols

    # Time individual classifier phases by profiling
    prof = cProfile.Profile()
    t0 = time.perf_counter()
    prof.enable()
    enriched, diag = recover_vector_symbols(tokens, lines, curves, PAGE, cfg)
    prof.disable()
    t1 = time.perf_counter()
    print(f"\nTotal wall time: {(t1 - t0) * 1000:.0f}ms")
    print(f"Symbols found: {diag.get('vector_symbols_found', 0)}")

    stats = pstats.Stats(prof)
    stats.sort_stats("cumulative")
    print("\n=== Top 30 by cumulative time ===")
    stats.print_stats(30)

    stats.sort_stats("tottime")
    print("\n=== Top 30 by total time ===")
    stats.print_stats(30)


if __name__ == "__main__":
    main()
