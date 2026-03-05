"""Extract a single PDF page through the full grouping pipeline and write JSON.

Usage:
    python scripts/extract_page.py <pdf> --page <N> [--out <path>]

Page is zero-based.  Output defaults to the most recent run's overlays folder:
    runs/<latest>/overlays/page_N_extraction.json

The JSON file contains:
  - All extracted tokens (GlyphBox list)
  - All blocks with labels, flags, bbox, and line/token indices
  - All notes columns with header/block references

Overlay scripts read this JSON instead of re-parsing the PDF.
"""

import argparse
import json
from pathlib import Path

from plancheck import GroupingConfig
from plancheck.export.page_data import serialize_page
from plancheck.pipeline import run_pipeline

from .run_utils import latest_overlays_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract page pipeline to JSON",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--page", type=int, default=0, help="Zero-based page index")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output JSON path (default: overlays/page_N_extraction.json in latest run)",
    )
    args = parser.parse_args()

    # Minimal config: TOCR + grouping only
    cfg = GroupingConfig()

    # ── Run pipeline ─────────────────────────────────────────────────────
    result = run_pipeline(args.pdf, args.page, cfg)

    print(
        f"Page {args.page}: {len(result.tokens)} tokens, "
        f"{len(result.blocks)} blocks, "
        f"{len(result.notes_columns)} notes column(s)"
    )

    # ── Serialize ────────────────────────────────────────────────────────
    data = serialize_page(
        page=args.page,
        page_width=result.page_width,
        page_height=result.page_height,
        tokens=result.tokens,
        blocks=result.blocks,
        notes_columns=result.notes_columns,
    )

    out_path = args.out
    if out_path is None:
        out_path = latest_overlays_dir() / f"page_{args.page}_extraction.json"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"Extraction saved: {out_path}")


if __name__ == "__main__":
    main()
