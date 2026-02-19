"""Generate an overlay with notes columns boxed in purple.

Usage:
    # Run pipeline + write JSON + draw overlay (standard):
    python scripts/overlay_notes_purple.py <pdf> --page <N> [--resolution <DPI>]

    # Draw overlay from existing JSON (skip PDF re-parse):
    python scripts/overlay_notes_purple.py --json <path/to/page_N_extraction.json> \\
        --pdf <pdf> [--resolution <DPI>]

Page is zero-based.  Writes to the most recent run's overlays/ folder.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

import argparse
import json

import pdfplumber

from plancheck import (
    GlyphBox,
    GroupingConfig,
    build_clusters_v2,
    draw_overlay,
    nms_prune,
)
from plancheck.export.page_data import deserialize_page, serialize_page
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
    parser = argparse.ArgumentParser(description="Purple notes-column overlay")
    parser.add_argument("pdf", type=Path, nargs="?", default=None)
    parser.add_argument("--page", type=int, default=0, help="Zero-based page index")
    parser.add_argument("--resolution", type=int, default=200)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output PNG path (default: overlays/ in latest run folder)",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        metavar="EXTRACTION_JSON",
        help="Read pipeline output from this JSON instead of re-running the pipeline",
    )
    args = parser.parse_args()

    cfg = GroupingConfig()

    #  Determine output path
    out_path = args.out
    if out_path is None:
        page_for_name = args.page if args.json is None else None
        out_path = _latest_overlays_dir() / f"page_{args.page}_notes_purple.png"

    #  Load or compute pipeline data
    if args.json is not None:
        raw = json.loads(args.json.read_text(encoding="utf-8"))
        tokens, blocks, notes_columns, page_w, page_h = deserialize_page(raw)
        page_idx = raw["page"]
        # Fix up out_path page index if not overridden
        if args.out is None:
            out_path = _latest_overlays_dir() / f"page_{page_idx}_notes_purple.png"
        pdf_path = args.pdf
    else:
        if args.pdf is None:
            parser.error("A PDF path is required unless --json is provided")
        pdf_path = args.pdf
        page_idx = args.page

        with pdfplumber.open(pdf_path) as pdf:
            page = pdf.pages[page_idx]
            result = extract_tocr_from_page(page, page_idx, cfg, mode="minimal")
            tokens = result.tokens
            page_w = result.page_width
            page_h = result.page_height

        tokens = nms_prune(tokens, cfg.iou_prune)
        blocks = build_clusters_v2(tokens, page_h, cfg)
        notes_columns = group_notes_columns(blocks, cfg=cfg)
        link_continued_columns(notes_columns, blocks=blocks, cfg=cfg)

        json_out = out_path.parent / f"page_{page_idx}_extraction.json"
        data = serialize_page(
            page=page_idx,
            page_width=page_w,
            page_height=page_h,
            tokens=tokens,
            blocks=blocks,
            notes_columns=notes_columns,
        )
        json_out.write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(f"Extraction JSON: {json_out}")

    boxes = tokens  # alias for draw_overlay call below

    # ── Flag suspect header words for VOCR inspection ───────────────
    from plancheck.grouping import flag_suspect_header_words

    suspects = flag_suspect_header_words(blocks)

    print(
        f"Page {page_idx}: {len(blocks)} blocks, {len(notes_columns)} notes column(s)"
    )

    # Compute smart labels for console output (same logic as overlay)
    from plancheck.export.overlay import _header_to_prefix

    prefix_counters: dict[str, int] = {}
    group_prefix: dict[str, str] = {}
    group_sub_counter: dict[str, int] = {}
    gn_labels: list[str] = []
    for col in notes_columns:
        grp = col.column_group_id
        if col.header is not None:
            hdr_text = col.header_text()
            prefix = _header_to_prefix(hdr_text) if hdr_text else "NC"
            count = prefix_counters.get(prefix, 0) + 1
            prefix_counters[prefix] = count
            label = f"{prefix}{count}"
            if grp is not None:
                group_prefix[grp] = label
                group_sub_counter[grp] = 2
            gn_labels.append(label)
        elif grp is not None and grp in group_prefix:
            parent_label = group_prefix[grp]
            sub = group_sub_counter.get(grp, 2)
            group_sub_counter[grp] = sub + 1
            gn_labels.append(f"{parent_label}.{sub}")
        else:
            prefix = "NC"
            count = prefix_counters.get(prefix, 0) + 1
            prefix_counters[prefix] = count
            gn_labels.append(f"NC{count}")

    for i, col in enumerate(notes_columns):
        hdr_boxes = col.header.get_all_boxes() if col.header else []
        hdr = " ".join(b.text for b in hdr_boxes) if hdr_boxes else "(no header)"
        cont = f"  continues='{col.continues_from}'" if col.continues_from else ""
        print(
            f"  {gn_labels[i]}: header={hdr!r}  blocks={len(col.notes_blocks)}  bbox={col.bbox()}{cont}"
        )

    # ── Report & persist suspect header words ───────────────────────
    if suspects:
        print(f"\n  VOCR SUSPECTS ({len(suspects)}):")
        for sr in suspects:
            print(
                f"    [{sr.reason}] '{sr.word_text}' "
                f"bbox=({sr.x0:.1f},{sr.y0:.1f},{sr.x1:.1f},{sr.y1:.1f})  "
                f"header='{sr.context}'"
            )

    # ── Render background image ─────────────────────────────────────
    if pdf_path is None:
        parser.error("A PDF path is required to render the background image")

    with pdfplumber.open(pdf_path) as pdf:
        bg_img = (
            pdf.pages[page_idx].to_image(resolution=args.resolution).original.copy()
        )

    # ── Build rows list (needed by draw_overlay) ────────────────────
    rows = [row for blk in blocks for row in (blk.rows or [])]

    # ── Draw overlay — purple notes columns only ────────────────────
    scale = args.resolution / 72.0
    PURPLE = (128, 0, 128, 220)

    draw_overlay(
        page_width=page_w,
        page_height=page_h,
        boxes=boxes,
        rows=rows,
        blocks=blocks,
        out_path=out_path,
        scale=scale,
        background=bg_img,
        notes_columns=notes_columns,
        color_overrides={"notes_columns": PURPLE},
        cfg=cfg,
    )
    print(f"Overlay saved: {out_path}")

    # ── Persist suspect regions as JSON for VOCR rectifier ──────────
    if suspects:
        suspect_path = out_path.parent / f"page_{page_idx}_vocr_suspects.json"
        payload = {
            "page": page_idx,
            "pdf": str(pdf_path),
            "page_width": page_w,
            "page_height": page_h,
            "suspects": [sr.to_dict() for sr in suspects],
        }
        suspect_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"VOCR suspect regions: {suspect_path}")


if __name__ == "__main__":
    main()
