import sys

sys.path.insert(0, "src")
import pdfplumber

from plancheck import GlyphBox, GroupingConfig, build_clusters_v2, nms_prune
from plancheck.grouping import (
    group_notes_columns,
    link_continued_columns,
    mark_headers,
    mark_notes,
)

cfg = GroupingConfig()
pdf_path = "input/IFC Operations Facilities McClain County - Drawings 25_0915.pdf"
page_num = 2

with pdfplumber.open(pdf_path) as pdf:
    page = pdf.pages[page_num]
    pw, ph = float(page.width), float(page.height)
    words = page.extract_words(
        x_tolerance=cfg.tocr_x_tolerance,
        y_tolerance=cfg.tocr_y_tolerance,
        extra_attrs=["fontname", "size"],
    )
    boxes = []
    for w in words:
        x0 = max(0, min(pw, float(w.get("x0", 0))))
        x1 = max(0, min(pw, float(w.get("x1", 0))))
        y0 = max(0, min(ph, float(w.get("top", 0))))
        y1 = max(0, min(ph, float(w.get("bottom", 0))))
        if x1 <= x0 or y1 <= y0:
            continue
        boxes.append(
            GlyphBox(
                page=page_num,
                x0=x0,
                y0=y0,
                x1=x1,
                y1=y1,
                text=w.get("text", ""),
                origin="text",
                fontname=w.get("fontname", ""),
                font_size=float(w.get("size", 0)),
            )
        )

boxes = nms_prune(boxes, cfg.iou_prune)
blocks = build_clusters_v2(boxes, ph, cfg)

from statistics import median

# Check _split_wide_blocks behavior
from plancheck.grouping import (
    _split_wide_blocks,
    build_lines,
    compute_median_space_gap,
    group_blocks_from_lines,
    split_line_spans,
    split_wide_lines,
)

# Recompute to check
lines2 = build_lines(boxes, cfg)
msg2 = compute_median_space_gap(lines2, boxes, cfg)
for ln in lines2:
    split_line_spans(ln, boxes, msg2, cfg.span_gap_mult)
lines2 = split_wide_lines(lines2, boxes)
blocks2 = group_blocks_from_lines(lines2, boxes, cfg, msg2)

# Check widths
widths = sorted([b.bbox()[2] - b.bbox()[0] for b in blocks2])
med_w = widths[len(widths) // 2]
width_thresh = med_w * 1.6
gap_thresh = max(msg2 * 6.0, 20.0)
print(f"=== _split_wide_blocks debug ===")
print(f"  median_space_gap={msg2:.1f}, gap_thresh={gap_thresh:.1f}")
print(f"  med_w={med_w:.1f}, width_thresh={width_thresh:.1f}")
print(f"  num blocks={len(blocks2)}")

# Check which blocks are wide
for i, blk in enumerate(blocks2):
    bb = blk.bbox()
    bw = bb[2] - bb[0]
    if bw > width_thresh:
        print(
            f"  WIDE Block[{i}]: width={bw:.1f} bbox=({bb[0]:.1f},{bb[1]:.1f},{bb[2]:.1f},{bb[3]:.1f}) lines={len(blk.lines)}"
        )
        # Check gaps in each line
        for li, ln in enumerate(blk.lines):
            if len(ln.token_indices) < 2:
                continue
            sorted_idx = sorted(ln.token_indices, key=lambda j: boxes[j].x0)
            max_gap = 0
            max_gap_pos = (0, 0)
            for j in range(len(sorted_idx) - 1):
                g = boxes[sorted_idx[j + 1]].x0 - boxes[sorted_idx[j]].x1
                if g > max_gap:
                    max_gap = g
                    max_gap_pos = (boxes[sorted_idx[j]].x1, boxes[sorted_idx[j + 1]].x0)
            if max_gap > 10:
                print(
                    f"    Line[{li}]: max_gap={max_gap:.1f} at x=({max_gap_pos[0]:.1f},{max_gap_pos[1]:.1f}) >= thresh={gap_thresh:.1f}? {max_gap >= gap_thresh}"
                )
mark_headers(blocks)
mark_notes(blocks)
cols = group_notes_columns(blocks, cfg=cfg)
link_continued_columns(cols, blocks=blocks, cfg=cfg)

## Dump all blocks in visual column 3 (x0 ~953) to see what's there
print("=== All blocks with x0 near 953 (visual column 3) ===")
for i, blk in enumerate(blocks):
    bb = blk.bbox()
    if 900 < bb[0] < 1000:
        all_boxes = blk.get_all_boxes()
        texts = " ".join(b.text for b in all_boxes[:12])
        flags = []
        if getattr(blk, "is_header", False):
            flags.append("HEADER")
        if getattr(blk, "is_notes", False):
            flags.append("NOTES")
        if blk.is_table:
            flags.append("TABLE")
        lbl = getattr(blk, "label", None)
        print(
            f"  Block[{i}]: bbox=({bb[0]:.1f}, {bb[1]:.1f}, {bb[2]:.1f}, {bb[3]:.1f})  "
            f"flags={flags}  label={lbl}  text={texts[:100]}"
        )

## 1. Check if CIPC words made it into GlyphBoxes
print("\n=== CIPC GlyphBoxes (y 180-205) ===")
cipc_glyphs = [b for b in boxes if 180 < b.y0 < 205 and 900 < b.x0 < 1250]
for g in cipc_glyphs:
    print(
        f"  '{g.text}' x0={g.x0:.1f} y0={g.y0:.1f} x1={g.x1:.1f} y1={g.y1:.1f} font={g.fontname} sz={g.font_size:.1f}"
    )

## 2. Check which block (if any) contains each CIPC glyph
print("\n=== Which block contains each CIPC glyph? ===")
for g in cipc_glyphs:
    found = False
    for i, blk in enumerate(blocks):
        if g in blk.get_all_boxes():
            print(f"  '{g.text}' -> Block[{i}]")
            found = True
            break
    if not found:
        print(f"  '{g.text}' -> NOT IN ANY BLOCK!")

## 3. Blocks around the CIPC area - show ALL boxes
print("\n=== Block[19] ALL boxes ===")
for g in blocks[19].get_all_boxes()[:30]:
    print(f"  '{g.text}' x0={g.x0:.1f} y0={g.y0:.1f}")

print(f"\n=== Block[14] ALL boxes ===")
for g in blocks[14].get_all_boxes()[:30]:
    print(f"  '{g.text}' x0={g.x0:.1f} y0={g.y0:.1f}")

## Block[12] - where CIPC header ended up
print(f"\n=== Block[12] details ===")
bb12 = blocks[12].bbox()
print(f"  bbox=({bb12[0]:.1f}, {bb12[1]:.1f}, {bb12[2]:.1f}, {bb12[3]:.1f})")
print(f"  is_header={getattr(blocks[12],'is_header',False)}")
print(f"  is_notes={getattr(blocks[12],'is_notes',False)}")
print(f"  label={getattr(blocks[12],'label',None)}")
print(f"  num_lines={len(blocks[12].lines)}")
print(f"  num_boxes={len(blocks[12].get_all_boxes())}")
for li, ln in enumerate(blocks[12].lines):
    lbb = ln.bbox(boxes)
    txt = ln.text(boxes)[:120]
    print(
        f"  Line[{li}]: x0={lbb[0]:.1f} y0={lbb[1]:.1f} x1={lbb[2]:.1f} y1={lbb[3]:.1f}  '{txt}'"
    )

print("\n=== NC3 (GN1.3) details ===")
nc3 = cols[2]
print(f"NC3 header: {nc3.header}")
print(f"NC3 blocks: {len(nc3.notes_blocks)}")
for i, blk in enumerate(nc3.notes_blocks):
    bb = blk.bbox()
    all_boxes = blk.get_all_boxes()
    texts = " ".join(b.text for b in all_boxes[:12])
    print(
        f"  Block {i}: bbox=({bb[0]:.1f}, {bb[1]:.1f}, {bb[2]:.1f}, {bb[3]:.1f})  text={texts[:100]}"
    )
