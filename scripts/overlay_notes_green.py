"""Generate an overlay with ONLY notes blocks boxed in green.

Usage:
    python scripts/overlay_notes_green.py <pdf> --page <N> [--resolution <DPI>]

Page is zero-based.  Writes to the most recent run's overlays/ folder.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import argparse

import pdfplumber
from PIL import ImageDraw, ImageFont

from plancheck import GlyphBox, GroupingConfig, build_clusters_v2, nms_prune


def _scale(x: float, y: float, s: float):
    return int(x * s), int(y * s)


def main() -> None:
    parser = argparse.ArgumentParser(description="Green notes-block overlay")
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--page", type=int, default=0, help="Zero-based page index")
    parser.add_argument("--resolution", type=int, default=200)
    parser.add_argument("--out", type=Path, default=None, help="Output PNG path")
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

    # ── Identify notes blocks ───────────────────────────────────────
    notes_blocks = [
        (i, blk) for i, blk in enumerate(blocks) if getattr(blk, "is_notes", False)
    ]

    # ── Build display labels ────────────────────────────────────────
    # Walk headers in order; each header starts a section. Notes under
    # that header inherit the header prefix + sequential note number.
    from plancheck.overlay import _header_to_prefix

    prefix_counters: dict[str, int] = {}
    current_prefix = "N"
    note_seq = 0
    display_labels: dict[int, str] = {}

    for i, blk in enumerate(blocks):
        label = getattr(blk, "label", None)
        if label == "note_column_header":
            # New header section — derive prefix
            if blk.rows and blk.rows[0].boxes:
                hdr_text = " ".join(b.text for b in blk.rows[0].boxes)
            else:
                all_boxes = blk.get_all_boxes()
                hdr_text = " ".join(b.text for b in all_boxes)
            prefix = _header_to_prefix(hdr_text) if hdr_text else "N"
            count = prefix_counters.get(prefix, 0) + 1
            prefix_counters[prefix] = count
            current_prefix = f"{prefix}{count}"
            note_seq = 0
        elif getattr(blk, "is_notes", False):
            note_seq += 1
            display_labels[i] = f"{current_prefix}-n{note_seq}"

    print(f"Page {args.page}: {len(blocks)} blocks, {len(notes_blocks)} notes block(s)")

    # ── Draw green rectangles on the background ─────────────────────
    scale = args.resolution / 72.0
    img = bg_img.convert("RGBA")
    img_w, img_h = int(page_w * scale), int(page_h * scale)
    if img.size != (img_w, img_h):
        img = img.resize((img_w, img_h))

    draw = ImageDraw.Draw(img, "RGBA")
    GREEN = (30, 160, 30, 230)
    LABEL_BG = (30, 130, 30, 180)
    LABEL_FG = (255, 255, 255, 255)
    LINE_W = 2

    try:
        font = ImageFont.truetype("arial.ttf", max(10, int(10 * scale / 2.78)))
    except OSError:
        font = ImageFont.load_default()

    for idx, blk in notes_blocks:
        x0, y0, x1, y1 = blk.bbox()
        sx0, sy0 = _scale(x0, y0, scale)
        sx1, sy1 = _scale(x1, y1, scale)
        draw.rectangle([(sx0, sy0), (sx1, sy1)], outline=GREEN, width=LINE_W)

        # First-row text preview
        if blk.rows and blk.rows[0].boxes:
            preview = " ".join(
                b.text for b in sorted(blk.rows[0].boxes, key=lambda b: b.x0)
            )
        else:
            all_boxes = blk.get_all_boxes()
            preview = " ".join(b.text for b in all_boxes)
        if len(preview) > 40:
            preview = preview[:37] + "..."

        dl = display_labels.get(idx, f"n{idx}")
        label = f"{dl}: {preview}"

        bbox_text = draw.textbbox((0, 0), label, font=font)
        tw = bbox_text[2] - bbox_text[0]
        th = bbox_text[3] - bbox_text[1]
        pad = 2
        lx = sx0
        ly = max(0, sy0 - th - pad * 2 - 2)
        draw.rectangle(
            [(lx, ly), (lx + tw + pad * 2, ly + th + pad * 2)],
            fill=LABEL_BG,
        )
        draw.text((lx + pad, ly + pad), label, fill=LABEL_FG, font=font)

    # ── Save ────────────────────────────────────────────────────────
    out_path = args.out
    if out_path is None:
        runs_dir = Path("runs")
        run_dirs = sorted(runs_dir.iterdir(), reverse=True)
        latest = run_dirs[0] if run_dirs else None
        if latest and (latest / "overlays").is_dir():
            out_path = latest / "overlays" / f"page_{args.page}_notes_green.png"
        else:
            out_path = Path(f"page_{args.page}_notes_green.png")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(out_path))
    print(f"Overlay saved: {out_path}")

    # ── Console summary ─────────────────────────────────────────────
    for idx, blk in notes_blocks:
        all_boxes = blk.get_all_boxes()
        text = " ".join(b.text for b in all_boxes)
        if len(text) > 80:
            text = text[:77] + "..."
        bx = blk.bbox()
        dl = display_labels.get(idx, f"n{idx}")
        print(
            f"  {dl}: bbox=({bx[0]:.1f},{bx[1]:.1f},{bx[2]:.1f},{bx[3]:.1f})  {text!r}"
        )


if __name__ == "__main__":
    main()
