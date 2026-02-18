"""Debug: full notes pipeline analysis for page 10."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from plancheck.config import GroupingConfig
from plancheck.grouping import (
    build_clusters_v2,
    group_notes_columns,
    link_continued_columns,
)
from plancheck.models import GlyphBox

run_dir = Path("runs/run_20260216_212302_IFC_Operations_Facil")
boxes_file = list(run_dir.glob("artifacts/*page_10_boxes.json"))[0]
with open(boxes_file) as f:
    data = json.load(f)
tokens = [GlyphBox(**b) for b in data]

cfg = GroupingConfig()
blocks = build_clusters_v2(tokens, 1584.0, cfg)
print(f"Blocks: {len(blocks)}")

headers = [b for b in blocks if b.is_header]
notes = [b for b in blocks if b.is_notes]
print(f"Headers: {len(headers)}")
print(f"Notes blocks: {len(notes)}")

for h in headers:
    bb = h.bbox()
    texts = [
        b.text for r in h.rows for b in sorted(r.boxes, key=lambda b: b.x0) if b.text
    ]
    print(
        f"  Header: {' '.join(texts)[:60]}  bbox=({bb[0]:.0f},{bb[1]:.0f},{bb[2]:.0f},{bb[3]:.0f})"
    )

print(f"\nNotes blocks ({len(notes)}):")
for i, n in enumerate(notes):
    bb = n.bbox()
    texts = [
        b.text for r in n.rows for b in sorted(r.boxes, key=lambda b: b.x0) if b.text
    ]
    print(
        f"  [{i}] rows={len(n.rows)} bbox=({bb[0]:.0f},{bb[1]:.0f},{bb[2]:.0f},{bb[3]:.0f}) "
        f"text={' '.join(texts)[:120]}"
    )

cols = group_notes_columns(blocks, cfg=cfg)
print(f"\nNotes columns: {len(cols)}")
for col in cols:
    hdr = (
        "orphan"
        if not col.header
        else " ".join(
            [
                b.text
                for r in col.header.rows
                for b in sorted(r.boxes, key=lambda b: b.x0)
                if b.text
            ]
        )[:50]
    )
    print(f'  Column "{hdr}": {len(col.notes_blocks)} notes')
    for j, nb in enumerate(col.notes_blocks):
        bb = nb.bbox()
        texts = [
            bx.text
            for r in nb.rows
            for bx in sorted(r.boxes, key=lambda bx: bx.x0)
            if bx.text
        ]
        print(
            f"    [{j}] rows={len(nb.rows)} bbox=({bb[0]:.0f},{bb[1]:.0f},{bb[2]:.0f},{bb[3]:.0f}) "
            f"text={' '.join(texts)[:100]}"
        )

link_continued_columns(cols, blocks, cfg=cfg)
total_notes = sum(len(c.notes_blocks) for c in cols)
print(f"\nTotal notes after linking: {total_notes}")

# Show blocks near PAVEMENT LEGEND to understand why 0 notes
print("\n--- Blocks near PAVEMENT LEGEND (x0 > 2000, y > 850) ---")
for i, b in enumerate(blocks):
    bb = b.bbox()
    if bb[0] > 2000 and bb[1] > 850:
        texts = [
            bx.text
            for r in b.rows
            for bx in sorted(r.boxes, key=lambda bx: bx.x0)
            if bx.text
        ]
        print(
            f"  Block {i}: rows={len(b.rows)} hdr={b.is_header} note={b.is_notes} "
            f"bbox=({bb[0]:.0f},{bb[1]:.0f},{bb[2]:.0f},{bb[3]:.0f}) "
            f"text={' '.join(texts)[:100]}"
        )
