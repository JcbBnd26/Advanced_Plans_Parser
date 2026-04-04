"""Deep profiling of the two worst TOCR bottlenecks: NMS prune and vector_symbols.

Measures call counts and per-function time in the hot paths.
"""

from __future__ import annotations

import cProfile
import pstats
import sys
import time
from io import StringIO
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

PDF_PATH = Path(
    r"C:\Users\jake\OneDrive\Desktop"
    r"\IFC Operations Facilities McClain County - Drawings 25_0915.pdf"
)
PAGE = 2  # worst page from initial profiling


def get_test_data():
    """Extract words, lines, curves from the PDF page."""
    import pdfplumber

    from plancheck.config import GroupingConfig
    from plancheck.tocr.extract import (
        build_extract_words_kwargs,
        extract_tocr_from_words,
    )

    cfg = GroupingConfig()
    kwargs = build_extract_words_kwargs(cfg, mode="full")

    with pdfplumber.open(PDF_PATH) as pdf:
        page = pdf.pages[PAGE]
        words = page.extract_words(**kwargs)
        lines = list(page.lines)
        curves = list(page.curves)
        page_w = float(page.width)
        page_h = float(page.height)

    result = extract_tocr_from_words(words, PAGE, page_w, page_h, cfg, mode="full")
    return result.tokens, lines, curves, page_w, page_h, cfg


def profile_nms():
    """Profile NMS prune with cProfile."""
    tokens, lines, curves, pw, ph, cfg = get_test_data()
    print(f"=== NMS Prune Profile (n={len(tokens)} tokens) ===")

    from plancheck.tocr.preprocess import nms_prune

    # Time it
    t0 = time.perf_counter()
    result = nms_prune(tokens, cfg.iou_prune)
    t1 = time.perf_counter()
    print(f"Wall time: {(t1-t0)*1000:.0f}ms")
    print(f"Result: {len(tokens)} -> {len(result)} boxes")
    print()

    # cProfile
    pr = cProfile.Profile()
    pr.enable()
    nms_prune(tokens, cfg.iou_prune)
    pr.disable()

    s = StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    ps.print_stats(20)
    print(s.getvalue())

    # Count IoU comparisons
    n = len(tokens)
    max_comparisons = n * (n - 1) // 2
    print(f"Max possible IoU comparisons: {max_comparisons:,}")
    print()


def profile_vector_symbols():
    """Profile vector symbol recovery with cProfile."""
    tokens, lines, curves, pw, ph, cfg = get_test_data()
    print(
        f"=== Vector Symbol Profile (tokens={len(tokens)}, lines={len(lines)}, curves={len(curves)}) ==="
    )

    from plancheck.tocr.vector_symbols import recover_vector_symbols

    # Time it
    t0 = time.perf_counter()
    result, diag = recover_vector_symbols(tokens, lines, curves, PAGE, cfg)
    t1 = time.perf_counter()
    print(f"Wall time: {(t1-t0)*1000:.0f}ms")
    print(f"Diag: {diag}")
    print()

    # cProfile
    pr = cProfile.Profile()
    pr.enable()
    recover_vector_symbols(tokens, lines, curves, PAGE, cfg)
    pr.disable()

    s = StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    ps.print_stats(30)
    print(s.getvalue())


def profile_extract_words():
    """Profile pdfplumber extract_words."""
    import pdfplumber

    from plancheck.config import GroupingConfig
    from plancheck.tocr.extract import build_extract_words_kwargs

    cfg = GroupingConfig()
    kwargs = build_extract_words_kwargs(cfg, mode="full")

    print(f"=== extract_words Profile (page {PAGE}) ===")
    print(f"extract_words kwargs: {kwargs}")

    with pdfplumber.open(PDF_PATH) as pdf:
        page = pdf.pages[PAGE]
        print(f"Page size: {page.width:.0f} x {page.height:.0f}")

        # Count chars first
        chars = list(page.chars)
        print(f"Total chars on page: {len(chars)}")

        # Profile extract_words
        t0 = time.perf_counter()
        words = page.extract_words(**kwargs)
        t1 = time.perf_counter()
        print(f"extract_words time: {(t1-t0)*1000:.0f}ms ({len(words)} words)")

        # Try with minimal kwargs
        t2 = time.perf_counter()
        words_min = page.extract_words(keep_blank_chars=False)
        t3 = time.perf_counter()
        print(f"extract_words (minimal): {(t3-t2)*1000:.0f}ms ({len(words_min)} words)")

        # Extra attrs contribution
        t4 = time.perf_counter()
        words_no_extra = page.extract_words(
            keep_blank_chars=False,
            x_tolerance=kwargs.get("x_tolerance", 3),
            y_tolerance=kwargs.get("y_tolerance", 3),
        )
        t5 = time.perf_counter()
        print(
            f"extract_words (no extra_attrs): {(t5-t4)*1000:.0f}ms ({len(words_no_extra)} words)"
        )

        # Check if extra_attrs is the culprit
        t6 = time.perf_counter()
        words_extra = page.extract_words(
            keep_blank_chars=False,
            x_tolerance=kwargs.get("x_tolerance", 3),
            y_tolerance=kwargs.get("y_tolerance", 3),
            extra_attrs=["fontname", "size", "upright"],
        )
        t7 = time.perf_counter()
        print(
            f"extract_words (with extra_attrs): {(t7-t6)*1000:.0f}ms ({len(words_extra)} words)"
        )
        print()


if __name__ == "__main__":
    profile_extract_words()
    profile_nms()
    profile_vector_symbols()
