"""Diagnose L95 span splitting."""

import sys

sys.path.insert(0, "scripts")
from run_pdf_batch import page_boxes

from plancheck import nms_prune
from plancheck.config import GroupingConfig
from plancheck.grouping import build_clusters_v2, build_lines, compute_median_space_gap

boxes, pw, ph = page_boxes(
    "input/IFC Operations Facilities McClain County - Drawings 25_0915.pdf", 2
)
cfg = GroupingConfig()
boxes = nms_prune(boxes, cfg.iou_prune)
blocks = build_clusters_v2(boxes, ph, cfg)

# Get median_space_gap
lines_raw = build_lines(boxes, cfg)
msg = compute_median_space_gap(lines_raw, boxes)
thresh = msg * cfg.span_gap_mult
print(
    f"median_space_gap={msg:.4f}  span_gap_mult={cfg.span_gap_mult}  threshold={thresh:.4f}"
)
print()

all_lines = [ln for blk in blocks for ln in (blk.lines or [])]
for ln in all_lines:
    if ln.line_id == 95:
        print(f"=== Line {ln.line_id} (baseline_y={ln.baseline_y:.2f}) ===")
        si = sorted(ln.token_indices, key=lambda i: boxes[i].x0)
        for pos, idx in enumerate(si):
            t = boxes[idx]
            if pos < len(si) - 1:
                ni = si[pos + 1]
                gap = boxes[ni].x0 - t.x1
            else:
                gap = None
            gs = f"  gap={gap:.2f}" if gap is not None else ""
            split = " ** SPLIT **" if gap is not None and gap > thresh else ""
            print(
                f"  [{idx}] x0={t.x0:.2f} x1={t.x1:.2f} text={repr(t.text)}{gs}{split}"
            )
        print(f"  Spans ({len(ln.spans)}):")
        for si2, span in enumerate(ln.spans):
            stoks = sorted(span.token_indices, key=lambda i: boxes[i].x0)
            x0s = boxes[stoks[0]].x0 if stoks else 0
            x1s = boxes[stoks[-1]].x1 if stoks else 0
            print(
                f"    S{si2}: x0={x0s:.2f} x1={x1s:.2f} col_id={span.col_id} text={repr(span.text(boxes))}"
            )
        print()
