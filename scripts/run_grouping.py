import argparse
import json
from pathlib import Path

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


def load_boxes(json_path: Path) -> list[GlyphBox]:
    data = json.loads(json_path.read_text())
    boxes: list[GlyphBox] = []
    for item in data:
        boxes.append(
            GlyphBox(
                page=int(item.get("page", 0)),
                x0=float(item["x0"]),
                y0=float(item["y0"]),
                x1=float(item["x1"]),
                y1=float(item["y1"]),
                text=item.get("text", ""),
                origin=item.get("origin", "text"),
            )
        )
    return boxes


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
        description="Run geometry-first grouping on JSON boxes"
    )
    parser.add_argument("json", type=Path, help="Path to JSON list of boxes")
    parser.add_argument(
        "--page-width", type=float, default=1000.0, help="Page width for skew rotation"
    )
    parser.add_argument(
        "--page-height",
        type=float,
        default=1000.0,
        help="Page height for skew rotation",
    )
    parser.add_argument(
        "--overlay", type=Path, help="Optional path to save overlay PNG"
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Overlay scale factor (e.g., 0.5 for downscale)",
    )
    args = parser.parse_args()

    boxes = load_boxes(args.json)
    cfg = GroupingConfig()

    boxes = nms_prune(boxes, cfg.iou_prune)
    if cfg.enable_skew:
        skew = estimate_skew_degrees(boxes, cfg.max_skew_degrees)
        boxes = rotate_boxes(boxes, -skew, args.page_width, args.page_height)

    rows = group_rows(boxes, cfg)
    blocks = group_blocks(rows, cfg)
    mark_tables(blocks, cfg)

    summarize(blocks)

    if args.overlay:
        draw_overlay(
            page_width=args.page_width,
            page_height=args.page_height,
            boxes=boxes,
            rows=rows,
            blocks=blocks,
            out_path=args.overlay,
            scale=args.scale,
        )
        print(f"Overlay saved to {args.overlay}")


if __name__ == "__main__":
    main()
