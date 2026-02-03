"""
Batch PDF processing entry point for GUI/CLI use.
Usage:
    python scripts/run_from_args.py --pdfs file1.pdf file2.pdf --mode all|single|range --single 3 --start 1 --end 5 --resolution 200
"""

import argparse
import os
from pathlib import Path

from run_pdf_batch import cleanup_old_runs, run_pdf

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
args = parser.parse_args()

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
    run_pdf(
        pdf=pdf_path,
        start=start,
        end=end,
        resolution=args.resolution,
        run_root=args.run_root,
        run_prefix=run_prefix,
    )
cleanup_old_runs(args.run_root, keep=args.keep_runs)
os.startfile(args.run_root)
