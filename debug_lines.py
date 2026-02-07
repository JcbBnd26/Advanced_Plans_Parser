"""Debug script to test the new row-truth pipeline (build_lines).

This script:
1. Loads tokens from a PDF page
2. Runs build_lines() to create canonical line groupings
3. Computes median_space_gap from actual token spacing
4. Splits lines into spans
5. Detects column boundaries and assigns col_ids
6. Renders an overlay showing lines and spans

Usage:
    python debug_lines.py "path/to/input.pdf" --page 2
"""

import argparse
import json
import sys
from pathlib import Path

# Ensure src is in path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import pdfplumber
from PIL import Image

from plancheck import GlyphBox, GroupingConfig
from plancheck.grouping import (
    assign_column_ids,
    build_lines,
    compute_median_space_gap,
    detect_column_boundaries,
    split_line_spans,
)
from plancheck.overlay import draw_lines_overlay


def load_page_tokens(
    pdf_path: Path, page_num: int
) -> tuple[list[GlyphBox], float, float]:
    """Extract tokens from a PDF page."""
    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[page_num]
        page_w, page_h = float(page.width), float(page.height)
        words = page.extract_words()

        boxes: list[GlyphBox] = []
        for w in words:
            x0 = max(0.0, min(page_w, float(w.get("x0", 0.0))))
            x1 = max(0.0, min(page_w, float(w.get("x1", 0.0))))
            y0 = max(0.0, min(page_h, float(w.get("top", 0.0))))
            y1 = max(0.0, min(page_h, float(w.get("bottom", 0.0))))
            text = w.get("text", "")
            if x1 <= x0 or y1 <= y0:
                continue
            boxes.append(
                GlyphBox(
                    page=page_num, x0=x0, y0=y0, x1=x1, y1=y1, text=text, origin="text"
                )
            )
    return boxes, page_w, page_h


def render_page_image(
    pdf_path: Path, page_num: int, resolution: int = 200
) -> Image.Image:
    """Render a page to a PIL image."""
    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[page_num]
        img_page = page.to_image(resolution=resolution)
        return img_page.original.copy()


def main():
    parser = argparse.ArgumentParser(description="Debug the row-truth pipeline")
    parser.add_argument("pdf_path", type=str, help="Path to the PDF file")
    parser.add_argument("--page", type=int, default=0, help="Page number (0-indexed)")
    parser.add_argument(
        "--output",
        type=str,
        default="debug_lines_overlay.png",
        help="Output overlay path",
    )
    parser.add_argument("--no-spans", action="store_true", help="Don't draw spans")
    parser.add_argument("--json", type=str, help="Output JSON with line/span data")
    args = parser.parse_args()

    pdf_path = Path(args.pdf_path)
    if not pdf_path.exists():
        print(f"Error: PDF not found: {pdf_path}")
        return 1

    print(f"Loading page {args.page} from {pdf_path}...")
    tokens, page_w, page_h = load_page_tokens(pdf_path, args.page)
    print(f"  Loaded {len(tokens)} tokens, page size: {page_w:.1f} x {page_h:.1f}")

    settings = GroupingConfig()

    # Step 1: Build lines
    print("\nBuilding lines...")
    lines = build_lines(tokens, settings)
    print(f"  Created {len(lines)} lines")

    # Step 2: Compute median space gap
    median_space_gap = compute_median_space_gap(lines, tokens)
    print(f"  Median space gap: {median_space_gap:.2f} pts")

    # Step 3: Split into spans
    print("\nSplitting lines into spans...")
    for line in lines:
        split_line_spans(line, tokens, median_space_gap, settings.span_gap_mult)
    total_spans = sum(len(line.spans) for line in lines)
    print(f"  Total spans: {total_spans}")

    # Step 4: Detect column boundaries
    print("\nDetecting column boundaries (content band only)...")
    col_boundaries = detect_column_boundaries(tokens, page_h, settings)
    print(f"  Column boundaries at x = {[f'{b:.1f}' for b in col_boundaries]}")
    print(f"  → {len(col_boundaries) + 1} column(s)")

    # Step 5: Assign column IDs
    assign_column_ids(lines, tokens, col_boundaries)

    # Print some sample lines
    print("\n--- Sample Lines ---")
    for line in lines[:10]:
        bbox = line.bbox(tokens)
        text = line.text(tokens)
        span_info = ", ".join(
            f"[col={s.col_id}:{len(s.token_indices)}tok]" for s in line.spans
        )
        print(
            f"L{line.line_id}: y={bbox[1]:.1f} | {len(line.spans)} spans ({span_info}) | {text[:60]}..."
        )

    if len(lines) > 10:
        print(f"  ... and {len(lines) - 10} more lines")

    # Render overlay
    print(f"\nRendering overlay to {args.output}...")
    bg_image = render_page_image(pdf_path, args.page, resolution=200)
    scale = 200 / 72.0  # PDF points to image pixels at 200 DPI

    draw_lines_overlay(
        page_width=page_w,
        page_height=page_h,
        lines=lines,
        tokens=tokens,
        out_path=Path(args.output),
        scale=scale,
        background=bg_image,
        draw_spans=not args.no_spans,
    )
    print(f"  Done! Open {args.output} to view.")

    # Optionally save JSON
    if args.json:
        print(f"\nSaving JSON to {args.json}...")
        data = {
            "page": args.page,
            "page_width": page_w,
            "page_height": page_h,
            "num_tokens": len(tokens),
            "median_space_gap": median_space_gap,
            "column_boundaries": col_boundaries,
            "lines": [
                {
                    "line_id": line.line_id,
                    "baseline_y": line.baseline_y,
                    "bbox": list(line.bbox(tokens)),
                    "text": line.text(tokens),
                    "num_tokens": len(line.token_indices),
                    "spans": [
                        {
                            "col_id": span.col_id,
                            "num_tokens": len(span.token_indices),
                            "bbox": list(span.bbox(tokens)),
                            "text": span.text(tokens),
                        }
                        for span in line.spans
                    ],
                }
                for line in lines
            ],
        }
        Path(args.json).write_text(json.dumps(data, indent=2))
        print(f"  Done!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
