#!/usr/bin/env python
"""Generate a text-only recreation PDF from an existing pipeline run.

Usage
-----
    python scripts/runners/run_sheet_recreation.py --run-dir runs/run_20260219_143000
    python scripts/runners/run_sheet_recreation.py --run-dir runs/run_20260219_143000 --pages 1,3,5
    python scripts/runners/run_sheet_recreation.py --run-dir runs/run_20260219_143000 --color-mode origin
    python scripts/runners/run_sheet_recreation.py --run-dir runs/run_20260219_143000 --layers
    python scripts/runners/run_sheet_recreation.py --run-dir runs/run_20260219_143000 --source-pdf plans.pdf
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure src/ is on the import path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from plancheck.export.sheet_recreation import ORIGIN_COLORS, recreate_sheet


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Recreate plan sheets as text-only PDFs from pipeline artifacts.",
    )
    p.add_argument(
        "--run-dir",
        required=True,
        type=Path,
        help="Path to an existing pipeline run directory (must contain artifacts/).",
    )
    p.add_argument(
        "--pages",
        default=None,
        help="Comma-separated 1-based page numbers (default: all).",
    )

    style_group = p.add_mutually_exclusive_group()
    style_group.add_argument(
        "--color-mode",
        choices=["plain", "origin"],
        default="plain",
        help=(
            "'plain' = black text on white (default). "
            "'origin' = colour-coded by token origin (TOCR/VOCR/reconcile)."
        ),
    )
    style_group.add_argument(
        "--layers",
        action="store_true",
        default=False,
        help=(
            "Organise content into togglable PDF layers (TOCR, VOCR, reconcile, "
            "block structure, labels). Mutually exclusive with --color-mode."
        ),
    )

    p.add_argument(
        "--no-blocks",
        action="store_true",
        default=False,
        help="Disable block boundary rectangles and semantic labels.",
    )
    p.add_argument(
        "--source-pdf",
        default=None,
        type=Path,
        help="Path to the original PDF for faint watermark background.",
    )
    p.add_argument(
        "--watermark-opacity",
        type=float,
        default=0.15,
        help="Opacity for watermark background (0.05–0.5, default 0.15).",
    )
    p.add_argument(
        "--out",
        default=None,
        type=Path,
        help="Explicit output PDF path (default: {run-dir}/exports/{stem}_recreation.pdf).",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    pages = None
    if args.pages:
        pages = [int(p.strip()) for p in args.pages.split(",")]

    # Colour mode: --layers supersedes --color-mode
    color_map = None
    if not args.layers and args.color_mode == "origin":
        color_map = ORIGIN_COLORS

    # Clamp watermark opacity
    wm_opacity = max(0.05, min(0.5, args.watermark_opacity))

    out = recreate_sheet(
        run_dir=args.run_dir,
        out_path=args.out,
        pages=pages,
        color_map=color_map,
        draw_blocks=not args.no_blocks,
        use_layers=args.layers,
        source_pdf=args.source_pdf,
        watermark_opacity=wm_opacity,
    )
    print(f"Sheet recreation saved -> {out}")


if __name__ == "__main__":
    main()
