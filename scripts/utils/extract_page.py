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

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

import argparse
import json

import pdfplumber

from plancheck import GroupingConfig, build_clusters_v2, nms_prune
from plancheck.export.page_data import serialize_page
from plancheck.grouping import group_notes_columns, link_continued_columns
from plancheck.tocr.extract import extract_tocr_from_page


def _latest_overlays_dir() -> Path:
    runs_dir = Path("runs")
    if runs_dir.is_dir():
        run_dirs = sorted(runs_dir.iterdir(), reverse=True)
        if run_dirs:
            d = run_dirs[0] / "overlays"
            d.mkdir(parents=True, exist_ok=True)
            return d
    return Path(".")


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

    cfg = GroupingConfig()

    # ── Extract tokens from PDF ────────────────────────────────────────────
    with pdfplumber.open(args.pdf) as pdf:
        page = pdf.pages[args.page]
        result = extract_tocr_from_page(page, args.page, cfg, mode="minimal")
        tokens = result.tokens
        page_w = result.page_width
        page_h = result.page_height

    # ── Run grouping pipeline ─────────────────────────────────────────────
    tokens = nms_prune(tokens, cfg.iou_prune)
    blocks = build_clusters_v2(tokens, page_h, cfg)
    notes_columns = group_notes_columns(blocks, cfg=cfg)
    link_continued_columns(notes_columns, blocks=blocks, cfg=cfg)

    print(
        f"Page {args.page}: {len(tokens)} tokens, "
        f"{len(blocks)} blocks, "
        f"{len(notes_columns)} notes column(s)"
    )

    # ── Serialize ────────────────────────────────────────────────────────
    data = serialize_page(
        page=args.page,
        page_width=page_w,
        page_height=page_h,
        tokens=tokens,
        blocks=blocks,
        notes_columns=notes_columns,
    )

    out_path = args.out
    if out_path is None:
        out_path = _latest_overlays_dir() / f"page_{args.page}_extraction.json"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"Extraction saved: {out_path}")


if __name__ == "__main__":
    main()
