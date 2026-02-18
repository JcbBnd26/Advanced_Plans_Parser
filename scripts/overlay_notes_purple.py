"""Generate an overlay with notes columns boxed in purple.

Usage:
    python scripts/overlay_notes_purple.py <pdf> --page <N> [--resolution <DPI>]

Page is zero-based.  Writes to the most recent run's overlays/ folder
(or creates a new one if --new-run is set).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import argparse

import pdfplumber

from plancheck import (
    GlyphBox,
    GroupingConfig,
    build_clusters_v2,
    draw_overlay,
    nms_prune,
)
from plancheck.grouping import group_notes_columns, link_continued_columns


def main() -> None:
    parser = argparse.ArgumentParser(description="Purple notes-column overlay")
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--page", type=int, default=0, help="Zero-based page index")
    parser.add_argument("--resolution", type=int, default=200)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output PNG path (default: overlays/ in latest run folder)",
    )
    args = parser.parse_args()

    cfg = GroupingConfig()

    # ── Extract text boxes ──────────────────────────────────────────
    with pdfplumber.open(args.pdf) as pdf:
        page = pdf.pages[args.page]
        page_w, page_h = float(page.width), float(page.height)
        words = page.extract_words(
            x_tolerance=cfg.tocr_x_tolerance,
            y_tolerance=cfg.tocr_y_tolerance,
            extra_attrs=["fontname", "size"] if cfg.tocr_extra_attrs else None,
        )
        boxes = []
        for w in words:
            x0 = max(0.0, min(page_w, float(w.get("x0", 0))))
            x1 = max(0.0, min(page_w, float(w.get("x1", 0))))
            y0 = max(0.0, min(page_h, float(w.get("top", 0))))
            y1 = max(0.0, min(page_h, float(w.get("bottom", 0))))
            text = w.get("text", "")
            if x1 <= x0 or y1 <= y0:
                continue
            boxes.append(
                GlyphBox(
                    page=args.page,
                    x0=x0,
                    y0=y0,
                    x1=x1,
                    y1=y1,
                    text=text,
                    origin="text",
                    fontname=w.get("fontname", ""),
                    font_size=float(w["size"]) if "size" in w else 0.0,
                )
            )

    # ── Render background image ─────────────────────────────────────
    with pdfplumber.open(args.pdf) as pdf:
        bg_img = (
            pdf.pages[args.page].to_image(resolution=args.resolution).original.copy()
        )

    # ── Grouping pipeline ───────────────────────────────────────────
    boxes = nms_prune(boxes, cfg.iou_prune)
    blocks = build_clusters_v2(boxes, page_h, cfg)
    notes_columns = group_notes_columns(blocks, cfg=cfg)
    link_continued_columns(notes_columns, blocks=blocks, cfg=cfg)

    # ── Flag suspect header words for VOCR inspection ───────────────
    import json

    from plancheck.grouping import flag_suspect_header_words

    suspects = flag_suspect_header_words(blocks)

    print(
        f"Page {args.page}: {len(blocks)} blocks, {len(notes_columns)} notes column(s)"
    )

    # Compute smart labels for console output (same logic as overlay)
    from plancheck.overlay import _header_to_prefix

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

    # ── Build rows list (needed by draw_overlay) ────────────────────
    rows = [row for blk in blocks for row in (blk.rows or [])]

    # ── Draw overlay — purple notes columns only ────────────────────
    scale = args.resolution / 72.0
    PURPLE = (128, 0, 128, 220)

    out_path = args.out
    if out_path is None:
        # Put in latest run's overlays folder
        runs_dir = Path("runs")
        run_dirs = sorted(runs_dir.iterdir(), reverse=True)
        latest = run_dirs[0] if run_dirs else None
        if latest and (latest / "overlays").is_dir():
            out_path = latest / "overlays" / f"page_{args.page}_notes_purple.png"
        else:
            out_path = Path(f"page_{args.page}_notes_purple.png")

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
        suspect_path = out_path.parent / f"page_{args.page}_vocr_suspects.json"
        payload = {
            "page": args.page,
            "pdf": str(args.pdf),
            "page_width": page_w,
            "page_height": page_h,
            "suspects": [sr.to_dict() for sr in suspects],
        }
        suspect_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"VOCR suspect regions: {suspect_path}")


if __name__ == "__main__":
    main()
