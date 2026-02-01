import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

import pdfplumber

from plancheck import (
    BlockCluster,
    GlyphBox,
    GroupingConfig,
    draw_overlay,
    estimate_skew_degrees,
    group_blocks,
    group_rows,
    mark_tables,
    nms_prune,
    rotate_boxes,
)


def make_run_dir(name: str | None = None) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"run_{stamp}" if not name else f"run_{stamp}_{name}"
    run_dir = Path("runs") / run_name
    for sub in ["inputs", "artifacts", "overlays", "exports", "logs"]:
        (run_dir / sub).mkdir(parents=True, exist_ok=True)
    return run_dir


def page_boxes(pdf_path: Path, page_num: int) -> tuple[list[GlyphBox], float, float]:
    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[page_num]
        page_w, page_h = float(page.width), float(page.height)
        words = page.extract_words()
        boxes: list[GlyphBox] = []
        for w in words:
            # Clip coordinates to page bounds (PDF content can extend past page edge)
            x0 = max(0.0, min(page_w, float(w.get("x0", 0.0))))
            x1 = max(0.0, min(page_w, float(w.get("x1", 0.0))))
            y0 = max(0.0, min(page_h, float(w.get("top", 0.0))))
            y1 = max(0.0, min(page_h, float(w.get("bottom", 0.0))))
            text = w.get("text", "")
            # Skip degenerate boxes (fully clipped)
            if x1 <= x0 or y1 <= y0:
                continue
            boxes.append(
                GlyphBox(
                    page=page_num, x0=x0, y0=y0, x1=x1, y1=y1, text=text, origin="text"
                )
            )
    return boxes, page_w, page_h


def render_page_image(pdf_path: Path, page_num: int, resolution: int = 200):
    """Render a page to a PIL image at given resolution (DPI)."""
    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[page_num]
        img_page = page.to_image(resolution=resolution)
        return img_page.original.copy()


def save_boxes_json(boxes: list[GlyphBox], out_path: Path) -> None:
    serializable = [
        {
            "page": b.page,
            "x0": b.x0,
            "y0": b.y0,
            "x1": b.x1,
            "y1": b.y1,
            "text": b.text,
            "origin": b.origin,
        }
        for b in boxes
    ]
    out_path.write_text(json.dumps(serializable, indent=2))


def summarize(blocks: list[BlockCluster]) -> None:
    print(f"Blocks: {len(blocks)}")
    table_count = sum(1 for b in blocks if b.is_table)
    print(f"Marked tables: {table_count}")
    for i, blk in enumerate(blocks, start=1):
        x0, y0, x1, y1 = blk.bbox()
        print(
            f"Block {i}: rows={len(blk.rows)} table={blk.is_table} bbox=({x0:.1f},{y0:.1f},{x1:.1f},{y1:.1f})"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract a PDF page to boxes JSON, run grouping, and save overlay"
    )
    parser.add_argument("pdf", type=Path, help="Path to PDF")
    parser.add_argument("--page", type=int, default=0, help="Zero-based page index")
    parser.add_argument(
        "--run-name", type=str, default=None, help="Optional suffix for the run folder"
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=200,
        help="Render resolution for overlays (DPI)",
    )
    args = parser.parse_args()

    run_dir = make_run_dir(args.run_name)

    # Copy PDF into run inputs for provenance.
    copied_pdf = run_dir / "inputs" / args.pdf.name
    shutil.copy2(args.pdf, copied_pdf)

    pdf_stem = args.pdf.stem.replace(" ", "_")

    boxes, page_w, page_h = page_boxes(args.pdf, args.page)
    bg_img = render_page_image(args.pdf, args.page, resolution=args.resolution)

    cfg = GroupingConfig()
    boxes = nms_prune(boxes, cfg.iou_prune)
    if cfg.enable_skew:
        skew = estimate_skew_degrees(boxes, cfg.max_skew_degrees)
        boxes = rotate_boxes(boxes, -skew, page_w, page_h)

    rows = group_rows(boxes, cfg)
    blocks = group_blocks(rows, cfg)
    mark_tables(blocks, cfg)

    # Save JSON of boxes used.
    boxes_path = run_dir / "artifacts" / f"{pdf_stem}_page_{args.page}_boxes.json"
    save_boxes_json(boxes, boxes_path)

    # Save overlay.
    overlay_path = run_dir / "overlays" / f"{pdf_stem}_page_{args.page}_overlay.png"
    scale = args.resolution / 72.0  # PDF user units are 1/72 inch.
    draw_overlay(
        page_width=page_w,
        page_height=page_h,
        boxes=boxes,
        rows=rows,
        blocks=blocks,
        out_path=overlay_path,
        scale=scale,
        background=bg_img,
    )

    # Manifest
    manifest = {
        "run_id": run_dir.name,
        "created_at": datetime.now().isoformat(),
        "pdf_original": str(args.pdf),
        "pdf_copied": str(copied_pdf),
        "page": args.page,
        "page_width": page_w,
        "page_height": page_h,
        "render_resolution_dpi": args.resolution,
        "overlay_scale": scale,
        "settings": vars(cfg),
        "skew_degrees": skew if cfg.enable_skew else 0.0,
        "counts": {
            "boxes": len(boxes),
            "rows": len(rows),
            "blocks": len(blocks),
            "tables": sum(1 for b in blocks if b.is_table),
        },
        "artifacts": {
            "boxes_json": str(boxes_path),
            "overlay_png": str(overlay_path),
        },
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"Run folder: {run_dir}")
    print(f"Boxes JSON: {boxes_path}")
    print(f"Overlay PNG: {overlay_path}")
    summarize(blocks)


if __name__ == "__main__":
    main()
