"""Test Surya OCR accuracy with different preprocessing configurations.

Compares three preprocessing modes:
1. grayscale=True (current default) — CLAHE + binarization, outputs mode "L"
2. grayscale=False — CLAHE only, preserves color
3. disabled — raw color image, no preprocessing

Outputs token counts, average confidence, and symbol detection rates.

Usage:
    python -m scripts.diagnostics.test_surya_preprocessing input/some_plan.pdf
    python -m scripts.diagnostics.test_surya_preprocessing input/*.pdf --pages 0-5
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np
import pdfplumber
from PIL import Image

# Insert src/ into path for local dev
src_path = Path(__file__).resolve().parents[2] / "src"
if src_path.exists() and str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from plancheck.config import GroupingConfig
from plancheck.vocr.backends import clear_backend_cache, get_ocr_backend
from plancheck.vocrpp.preprocess import OcrPreprocessConfig, preprocess_image_for_ocr

# Symbols we care about for plan checking
SYMBOL_CHARS = set("%/°±Ø×'\"#@⌀∅≤≥≈")


@dataclass
class PreprocessingResult:
    """Results from a single preprocessing configuration."""

    mode: str
    token_count: int
    avg_confidence: float
    symbol_count: int
    symbols_found: List[str]
    high_conf_tokens: int  # confidence >= 0.8


@dataclass
class PageComparison:
    """Comparison results for a single page."""

    pdf_path: str
    page_num: int
    grayscale_on: PreprocessingResult
    grayscale_off: PreprocessingResult
    preprocess_off: PreprocessingResult


def render_page(pdf_path: Path, page_num: int, resolution: int = 150) -> Image.Image:
    """Render a PDF page to PIL image."""
    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[page_num]
        return page.to_image(resolution=resolution).original.copy()


def run_ocr_with_config(
    page_image: Image.Image,
    cfg: GroupingConfig,
    enable_preprocess: bool,
    grayscale: bool,
) -> PreprocessingResult:
    """Run OCR with specific preprocessing settings and return metrics."""

    # Apply preprocessing if enabled
    if enable_preprocess:
        preproc_cfg = OcrPreprocessConfig(
            enable_clahe=True,
            grayscale=grayscale,
            binarize=grayscale,  # Only binarize with grayscale
        )
        processed_img = preprocess_image_for_ocr(page_image, preproc_cfg)
    else:
        processed_img = page_image

    # Ensure RGB for OCR
    if processed_img.mode != "RGB":
        processed_img = processed_img.convert("RGB")

    # Run OCR
    backend = get_ocr_backend(cfg)
    img_array = np.array(processed_img)
    text_boxes = backend.predict(img_array)

    # Compute metrics
    token_count = len(text_boxes)
    confidences = [box.confidence for box in text_boxes]
    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
    high_conf = sum(1 for c in confidences if c >= 0.8)

    # Find symbols
    symbols_found = []
    for box in text_boxes:
        for ch in box.text:
            if ch in SYMBOL_CHARS and ch not in symbols_found:
                symbols_found.append(ch)

    mode_name = (
        "preprocess_off"
        if not enable_preprocess
        else ("grayscale_on" if grayscale else "grayscale_off")
    )

    return PreprocessingResult(
        mode=mode_name,
        token_count=token_count,
        avg_confidence=round(avg_conf, 4),
        symbol_count=len(symbols_found),
        symbols_found=symbols_found,
        high_conf_tokens=high_conf,
    )


def compare_page(
    pdf_path: Path,
    page_num: int,
    cfg: GroupingConfig,
    resolution: int = 150,
) -> PageComparison:
    """Compare all three preprocessing modes on a single page."""

    page_image = render_page(pdf_path, page_num, resolution)

    print(f"  Page {page_num}: ", end="", flush=True)

    # Mode 1: grayscale=True (current default)
    result_gray = run_ocr_with_config(
        page_image, cfg, enable_preprocess=True, grayscale=True
    )
    print(f"gray={result_gray.token_count} ", end="", flush=True)

    # Mode 2: grayscale=False (color + CLAHE)
    result_color = run_ocr_with_config(
        page_image, cfg, enable_preprocess=True, grayscale=False
    )
    print(f"color={result_color.token_count} ", end="", flush=True)

    # Mode 3: no preprocessing (raw)
    result_raw = run_ocr_with_config(
        page_image, cfg, enable_preprocess=False, grayscale=False
    )
    print(f"raw={result_raw.token_count}")

    return PageComparison(
        pdf_path=str(pdf_path),
        page_num=page_num,
        grayscale_on=result_gray,
        grayscale_off=result_color,
        preprocess_off=result_raw,
    )


def aggregate_results(comparisons: List[PageComparison]) -> dict:
    """Compute aggregate statistics across all pages."""

    def aggregate_mode(results: List[PreprocessingResult]) -> dict:
        if not results:
            return {}
        return {
            "total_tokens": sum(r.token_count for r in results),
            "avg_tokens_per_page": round(
                sum(r.token_count for r in results) / len(results), 1
            ),
            "avg_confidence": round(
                sum(r.avg_confidence for r in results) / len(results), 4
            ),
            "total_high_conf": sum(r.high_conf_tokens for r in results),
            "unique_symbols": list(set(s for r in results for s in r.symbols_found)),
            "total_symbol_count": sum(r.symbol_count for r in results),
        }

    return {
        "pages_tested": len(comparisons),
        "grayscale_on": aggregate_mode([c.grayscale_on for c in comparisons]),
        "grayscale_off": aggregate_mode([c.grayscale_off for c in comparisons]),
        "preprocess_off": aggregate_mode([c.preprocess_off for c in comparisons]),
    }


def print_summary(agg: dict) -> None:
    """Print a human-readable summary."""
    print("\n" + "=" * 60)
    print("PREPROCESSING COMPARISON SUMMARY")
    print("=" * 60)
    print(f"Pages tested: {agg['pages_tested']}\n")

    headers = ["Metric", "Grayscale ON", "Grayscale OFF", "No Preprocess"]
    rows = [
        (
            "Avg tokens/page",
            agg["grayscale_on"]["avg_tokens_per_page"],
            agg["grayscale_off"]["avg_tokens_per_page"],
            agg["preprocess_off"]["avg_tokens_per_page"],
        ),
        (
            "Avg confidence",
            agg["grayscale_on"]["avg_confidence"],
            agg["grayscale_off"]["avg_confidence"],
            agg["preprocess_off"]["avg_confidence"],
        ),
        (
            "High-conf tokens",
            agg["grayscale_on"]["total_high_conf"],
            agg["grayscale_off"]["total_high_conf"],
            agg["preprocess_off"]["total_high_conf"],
        ),
        (
            "Symbols found",
            len(agg["grayscale_on"]["unique_symbols"]),
            len(agg["grayscale_off"]["unique_symbols"]),
            len(agg["preprocess_off"]["unique_symbols"]),
        ),
    ]

    # Print table
    col_width = 18
    print(
        f"{'Metric':<20} {'Gray ON':>{col_width}} {'Gray OFF':>{col_width}} {'Raw':>{col_width}}"
    )
    print("-" * 74)
    for row in rows:
        print(
            f"{row[0]:<20} {str(row[1]):>{col_width}} {str(row[2]):>{col_width}} {str(row[3]):>{col_width}}"
        )

    print("\nUnique symbols detected:")
    print(f"  Grayscale ON:  {agg['grayscale_on']['unique_symbols']}")
    print(f"  Grayscale OFF: {agg['grayscale_off']['unique_symbols']}")
    print(f"  No preprocess: {agg['preprocess_off']['unique_symbols']}")

    # Recommendation
    print("\n" + "-" * 60)
    gray_on = agg["grayscale_on"]
    gray_off = agg["grayscale_off"]
    raw = agg["preprocess_off"]

    best_conf = max(
        (gray_on["avg_confidence"], "grayscale_on"),
        (gray_off["avg_confidence"], "grayscale_off"),
        (raw["avg_confidence"], "preprocess_off"),
    )
    best_tokens = max(
        (gray_on["avg_tokens_per_page"], "grayscale_on"),
        (gray_off["avg_tokens_per_page"], "grayscale_off"),
        (raw["avg_tokens_per_page"], "preprocess_off"),
    )

    print(f"Best avg confidence: {best_conf[1]} ({best_conf[0]:.4f})")
    print(f"Most tokens/page:    {best_tokens[1]} ({best_tokens[0]:.1f})")


def parse_page_range(page_arg: Optional[str], max_pages: int) -> tuple[int, int]:
    """Parse page range argument like '0-5' or '3'."""
    if not page_arg:
        return 0, min(5, max_pages)  # Default: first 5 pages

    if "-" in page_arg:
        start, end = page_arg.split("-", 1)
        return int(start), min(int(end) + 1, max_pages)
    else:
        page = int(page_arg)
        return page, page + 1


def main():
    parser = argparse.ArgumentParser(
        description="Compare Surya OCR accuracy across preprocessing modes"
    )
    parser.add_argument(
        "pdf_paths",
        nargs="+",
        type=Path,
        help="PDF file(s) to test",
    )
    parser.add_argument(
        "--pages",
        type=str,
        default=None,
        help="Page range to test, e.g., '0-5' or '3' (default: first 5 pages)",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=150,
        help="Render resolution in DPI (default: 150)",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "gpu"],
        default="cpu",
        help="Device to run Surya on (default: cpu)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON file for detailed results",
    )

    args = parser.parse_args()

    # Validate PDFs exist
    for pdf_path in args.pdf_paths:
        if not pdf_path.exists():
            print(f"ERROR: PDF not found: {pdf_path}", file=sys.stderr)
            sys.exit(1)

    # Create config
    cfg = GroupingConfig(
        vocr_backend="surya",
        vocr_device=args.device,
    )

    print(f"Surya OCR preprocessing comparison")
    print(f"Device: {args.device}")
    print(f"Resolution: {args.resolution} DPI")
    print("-" * 60)

    all_comparisons: List[PageComparison] = []

    for pdf_path in args.pdf_paths:
        print(f"\nProcessing: {pdf_path.name}")

        # Get page count
        with pdfplumber.open(pdf_path) as pdf:
            max_pages = len(pdf.pages)

        start_page, end_page = parse_page_range(args.pages, max_pages)
        print(f"  Testing pages {start_page}-{end_page - 1}")

        # Clear backend cache to ensure fresh state
        clear_backend_cache()

        for page_num in range(start_page, end_page):
            try:
                comparison = compare_page(pdf_path, page_num, cfg, args.resolution)
                all_comparisons.append(comparison)
            except Exception as e:
                print(f"  ERROR on page {page_num}: {e}")

    if not all_comparisons:
        print("No pages processed successfully.")
        sys.exit(1)

    # Aggregate and display results
    aggregate = aggregate_results(all_comparisons)
    print_summary(aggregate)

    # Save detailed results if requested
    if args.output:
        output_data = {
            "timestamp": datetime.now().isoformat(),
            "config": {
                "device": args.device,
                "resolution": args.resolution,
            },
            "aggregate": aggregate,
            "pages": [
                {
                    "pdf": c.pdf_path,
                    "page": c.page_num,
                    "grayscale_on": asdict(c.grayscale_on),
                    "grayscale_off": asdict(c.grayscale_off),
                    "preprocess_off": asdict(c.preprocess_off),
                }
                for c in all_comparisons
            ],
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nDetailed results saved to: {args.output}")


if __name__ == "__main__":
    main()
