"""
Batch PDF processing entry point for GUI/CLI use.
Usage:
    python scripts/run_from_args.py --pdfs file1.pdf file2.pdf --mode all|single|range --single 3 --start 1 --end 5 --resolution 200 --colors-file path/to/colors.json --ocr-preprocess
"""

import argparse
import json
import os
from pathlib import Path

from run_pdf_batch import cleanup_old_runs, run_pdf

from plancheck.config import GroupingConfig


def hex_to_rgba(hex_color: str, alpha: int = 200) -> tuple:
    """Convert hex color (#RRGGBB) to RGBA tuple."""
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return (r, g, b, alpha)


parser = argparse.ArgumentParser(
    description="Batch PDF processing for Advanced Plans Parser"
)
parser.add_argument(
    "--pdfs", nargs="+", type=Path, required=True, help="PDF file(s) to process"
)
parser.add_argument(
    "--mode",
    choices=["all", "single", "range"],
    default="all",
    help="Page selection mode",
)
parser.add_argument("--single", type=int, help="Single page number (1-indexed)")
parser.add_argument("--start", type=int, help="Start page (1-indexed)")
parser.add_argument("--end", type=int, help="End page (inclusive, 1-indexed)")
parser.add_argument("--resolution", type=int, default=200, help="Overlay render DPI")
parser.add_argument(
    "--run-root", type=Path, default=Path("runs"), help="Root directory for runs"
)
parser.add_argument("--keep-runs", type=int, default=50, help="Number of runs to keep")
parser.add_argument(
    "--colors-file",
    type=Path,
    default=None,
    help="Path to JSON file with color overrides",
)
parser.add_argument(
    "--no-tocr",
    action="store_true",
    help="Disable text-layer OCR (pdfplumber word extraction)",
)
parser.add_argument(
    "--vocr",
    action="store_true",
    help="Run PaddleOCR full-page visual token extraction",
)
parser.add_argument(
    "--ocr-preprocess",
    action="store_true",
    help="Preprocess OCR image (grayscale + CLAHE contrast) before PaddleOCR",
)
parser.add_argument(
    "--ocr-full-reconcile",
    action="store_true",
    help="Enable OCR reconciliation (inject missing symbols from VOCR into text layer)",
)
parser.add_argument(
    "--ocr-debug",
    action="store_true",
    help="Force OCR reconcile debug overlay",
)
parser.add_argument(
    "--ocr-resolution",
    type=int,
    default=300,
    help="DPI for OCR page render (default 300)",
)
args = parser.parse_args()

# Build GroupingConfig from OCR-related flags.
# --ocr-full-reconcile implies --vocr (reconcile needs VOCR tokens).
# --ocr-preprocess implies --vocr (preprocessing has no consumer without VOCR).
_vocr = args.vocr or args.ocr_full_reconcile or args.ocr_preprocess
ocr_cfg = GroupingConfig(
    enable_tocr=not args.no_tocr,
    enable_vocr=_vocr,
    enable_ocr_reconcile=args.ocr_full_reconcile,
    enable_ocr_preprocess=args.ocr_preprocess,
    ocr_reconcile_debug=args.ocr_debug,
    ocr_reconcile_resolution=args.ocr_resolution,
)

# Parse color overrides from file
color_overrides = None
if args.colors_file and args.colors_file.exists():
    try:
        raw_colors = json.loads(args.colors_file.read_text())
        color_overrides = {}
        for key, val in raw_colors.items():
            if isinstance(val, str):
                # Hex string like "#FF0000"
                color_overrides[key] = hex_to_rgba(val)
            elif isinstance(val, (list, tuple)) and len(val) >= 3:
                # RGBA tuple like [255, 0, 0, 200]
                color_overrides[key] = tuple(val)
        print(f"Loaded {len(color_overrides)} color overrides from {args.colors_file}")
    except (json.JSONDecodeError, Exception) as e:
        print(f"Warning: Could not parse colors file: {e}")

for pdf_path in args.pdfs:
    if not pdf_path.exists():
        print(f"File not found: {pdf_path}")
        continue
    if args.mode == "all":
        start, end = 0, None
    elif args.mode == "single":
        if not args.single:
            print("--single required for single mode")
            continue
        start, end = args.single - 1, args.single
    elif args.mode == "range":
        start = (args.start - 1) if args.start else 0
        end = args.end if args.end else None
    else:
        start, end = 0, None
    run_prefix = pdf_path.stem.replace(" ", "_")[:20]
    print(
        f"Processing {pdf_path} (start={start}, end={end}, resolution={args.resolution})"
    )
    run_dir = run_pdf(
        pdf=pdf_path,
        start=start,
        end=end,
        resolution=args.resolution,
        run_root=args.run_root,
        run_prefix=run_prefix,
        color_overrides=color_overrides,
        cfg=ocr_cfg,
    )

cleanup_old_runs(args.run_root, keep=args.keep_runs)
os.startfile(args.run_root)
