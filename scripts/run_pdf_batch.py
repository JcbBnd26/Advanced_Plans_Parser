import re


def detect_headers_from_words(boxes: list[GlyphBox]) -> list[GlyphBox]:
    """Tag header candidates directly from word boxes (before grouping)."""
    # Improved: group words into lines by y0 and horizontal proximity
    from collections import defaultdict

    header_re = re.compile(r"^[A-Z0-9\s\-\(\)\'\.]+: *$", re.ASCII)
    # Step 1: group by y0 (row)
    y_tol = 2.0  # tolerance for y alignment in points
    rows = defaultdict(list)
    for b in boxes:
        found = False
        for y in rows:
            if abs(b.y0 - y) < y_tol:
                rows[y].append(b)
                found = True
                break
        if not found:
            rows[b.y0].append(b)
    header_boxes = []
    for row in rows.values():
        # Step 2: sort by x0 and group horizontally close words into lines
        row = sorted(row, key=lambda b: b.x0)
        line = []
        lines = []
        x_gap_tol = 20.0  # max gap between words in a line (points)
        for b in row:
            if not line:
                line.append(b)
            else:
                prev = line[-1]
                if b.x0 - prev.x1 < x_gap_tol:
                    line.append(b)
                else:
                    lines.append(line)
                    line = [b]
        if line:
            lines.append(line)
        # Step 3: apply header regex to each line
        for line_words in lines:
            line_text = " ".join(b.text for b in line_words if b.text).strip()
            line_text_norm = re.sub(r"\s+", " ", line_text).upper()
            if header_re.match(line_text_norm):
                for b in line_words:
                    b.origin = "header_candidate"
                header_boxes.extend(line_words)
    return header_boxes


import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
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
    mark_notes,
    mark_tables,
    nms_prune,
    rotate_boxes,
)


def make_run_dir(base: Path, name: str) -> Path:
    run_dir = base / name
    for sub in ["inputs", "artifacts", "overlays", "exports", "logs"]:
        (run_dir / sub).mkdir(parents=True, exist_ok=True)
    return run_dir


def page_boxes(pdf_path: Path, page_num: int) -> tuple[list[GlyphBox], float, float]:
    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[page_num]
        words = page.extract_words()
        boxes: list[GlyphBox] = []
        for w in words:
            x0 = float(w.get("x0", 0.0))
            x1 = float(w.get("x1", 0.0))
            y0 = float(w.get("top", 0.0))
            y1 = float(w.get("bottom", 0.0))
            text = w.get("text", "")
            boxes.append(
                GlyphBox(
                    page=page_num, x0=x0, y0=y0, x1=x1, y1=y1, text=text, origin="text"
                )
            )
        w, h = float(page.width), float(page.height)
    return boxes, w, h


def render_page_image(pdf_path: Path, page_num: int, resolution: int = 200):
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


def summarize(blocks: list[BlockCluster]) -> str:
    lines = [f"Blocks: {len(blocks)}"]
    table_count = sum(1 for b in blocks if b.is_table)
    lines.append(f"Marked tables: {table_count}")
    for i, blk in enumerate(blocks, start=1):
        x0, y0, x1, y1 = blk.bbox()
        lines.append(
            f"Block {i}: rows={len(blk.rows)} table={blk.is_table} bbox=({x0:.1f},{y0:.1f},{x1:.1f},{y1:.1f})"
        )
    return "\n".join(lines)


def process_page(
    pdf: Path, page_num: int, run_root: Path, run_prefix: str, resolution: int
) -> None:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"run_{stamp}_{run_prefix}_p{page_num}"
    run_dir = make_run_dir(run_root, run_name)

    copied_pdf = run_dir / "inputs" / pdf.name
    shutil.copy2(pdf, copied_pdf)

    pdf_stem = pdf.stem.replace(" ", "_")

    boxes, page_w, page_h = page_boxes(pdf, page_num)
    bg_img = render_page_image(pdf, page_num, resolution=resolution)

    cfg = GroupingConfig()
    boxes = nms_prune(boxes, cfg.iou_prune)
    if cfg.enable_skew:
        skew = estimate_skew_degrees(boxes, cfg.max_skew_degrees)
        boxes = rotate_boxes(boxes, -skew, page_w, page_h)
    else:
        skew = 0.0

    rows = group_rows(boxes, cfg)
    blocks = group_blocks(rows, cfg)
    mark_tables(blocks, cfg)
    # Block-level header detection and debug output
    from plancheck.grouping import mark_headers

    debug_path = str(run_dir / "artifacts" / "debug_headers.txt")
    mark_headers(blocks, debug_path=debug_path)
    # Notes detection, will skip header blocks
    mark_notes(blocks, debug_path=debug_path)

    boxes_path = run_dir / "artifacts" / f"{pdf_stem}_page_{page_num}_boxes.json"
    save_boxes_json(boxes, boxes_path)

    # Save block-level clusters with tags
    blocks_path = run_dir / "artifacts" / f"{pdf_stem}_page_{page_num}_blocks.json"

    def serialize_block(blk):
        x0, y0, x1, y1 = blk.bbox()
        return {
            "page": blk.page,
            "bbox": [x0, y0, x1, y1],
            "rows": [
                {
                    "bbox": list(row.bbox()),
                    "texts": [b.text for b in row.boxes],
                }
                for row in blk.rows
            ],
            "label": blk.label,
            "is_table": blk.is_table,
            "is_notes": blk.is_notes,
        }

    blocks_serialized = [serialize_block(blk) for blk in blocks]
    blocks_path.write_text(json.dumps(blocks_serialized, indent=2))

    overlay_path = run_dir / "overlays" / f"{pdf_stem}_page_{page_num}_overlay.png"
    scale = resolution / 72.0
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

    manifest = {
        "run_id": run_dir.name,
        "created_at": datetime.now().isoformat(),
        "pdf_original": str(pdf),
        "pdf_copied": str(copied_pdf),
        "page": page_num,
        "page_width": page_w,
        "page_height": page_h,
        "render_resolution_dpi": resolution,
        "overlay_scale": scale,
        "settings": vars(cfg),
        "skew_degrees": skew,
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

    print(f"page {page_num}: {run_dir}", flush=True)
    print(summarize(blocks), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch run pages of a PDF")
    parser.add_argument("pdf", type=Path, help="Path to PDF")
    parser.add_argument("--start", type=int, default=0, help="Start page (inclusive)")
    parser.add_argument(
        "--end", type=int, default=None, help="End page (exclusive); default = all"
    )
    parser.add_argument(
        "--resolution", type=int, default=200, help="Overlay render DPI"
    )
    parser.add_argument(
        "--run-prefix",
        type=str,
        default="batch",
        help="Suffix used in run folder names",
    )
    parser.add_argument(
        "--run-root", type=Path, default=Path("runs"), help="Root directory for runs"
    )
    args = parser.parse_args()

    with pdfplumber.open(args.pdf) as pdf:
        total_pages = len(pdf.pages)
    end_page = args.end if args.end is not None else total_pages

    for page_num in range(args.start, end_page):
        try:
            process_page(
                args.pdf, page_num, args.run_root, args.run_prefix, args.resolution
            )
        except Exception as exc:  # pragma: no cover
            print(f"page {page_num}: ERROR {exc}", flush=True)


if __name__ == "__main__":
    main()
