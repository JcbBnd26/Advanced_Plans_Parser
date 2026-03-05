"""Notes column grouping and continuation linking.

This module handles grouping header+notes blocks into NotesColumn objects
and detecting continuation relationships between columns.
"""

from __future__ import annotations

import io
import logging
import re
from contextlib import contextmanager
from typing import IO, List

from ..config import GroupingConfig
from ..models import BlockCluster, NotesColumn

log = logging.getLogger(__name__)

__all__ = [
    "group_notes_columns",
    "link_continued_columns",
]

# ── Pre-compiled patterns ──────────────────────────────────────────────
_NOTE_SIMPLE_RE = re.compile(r"^\d+\.")
_NOTE_BROAD_RE = re.compile(r"^(?:\d+\.|[A-Z]\.|[a-z]\.|\(\d+\)|\([A-Za-z]\))")
_NOTE_CAPTURE_RE = re.compile(r"^(\d+)\.")


# ── Shared utilities ───────────────────────────────────────────────────


@contextmanager
def _open_debug(path: str | None) -> IO[str]:  # type: ignore[type-arg]
    """Yield a writable text stream for debug output."""
    if path is None:
        yield io.StringIO()
    else:
        f = open(path, "a", encoding="utf-8")
        try:
            yield f
        finally:
            f.close()


def _block_first_row_text(blk: BlockCluster) -> str:
    """Get normalised text of the first row (ALL CAPS, collapsed whitespace)."""
    if not blk.rows:
        return ""
    first_row = blk.rows[0]
    texts = [b.text for b in sorted(first_row.boxes, key=lambda b: b.x0) if b.text]
    if not texts:
        return ""
    return re.sub(r"\s+", " ", " ".join(texts).strip()).upper()


# ── Notes column grouping ──────────────────────────────────────────────


def group_notes_columns(
    blocks: List[BlockCluster],
    x_tolerance: float = 30.0,
    y_gap_max: float = 50.0,
    debug_path: str = None,
    cfg: GroupingConfig | None = None,
) -> List[NotesColumn]:
    """Group header blocks with their associated notes blocks into NotesColumn objects.

    **Visual-column algorithm with reading-order continuation:**

    1. Collect all header and notes blocks.
    2. Cluster them by x0 proximity: sort by x0, split on gaps > x_tolerance.
       Each cluster represents a visual column on the page.
    3. Sort clusters left-to-right; within each cluster, sort blocks top-to-bottom.
    4. Walk blocks in reading order (left→right, top→bottom).  Headers open a
       new NotesColumn.  Orphan notes blocks (those with no header at the top
       of their visual column) are assigned to the *most recently opened*
       NotesColumn from a previous cluster, creating a sub-column linked via
       ``column_group_id``.  This handles "snake" columns where a single
       notes section wraps across multiple visual columns.

    Returns a list of :class:`NotesColumn` objects.
    """
    if cfg is not None:
        x_tolerance = cfg.grouping_notes_x_tolerance
        y_gap_max = cfg.grouping_notes_y_gap_max
    # Collect labeled blocks
    headers = [b for b in blocks if getattr(b, "is_header", False)]
    notes = [b for b in blocks if getattr(b, "is_notes", False)]
    labeled = set(id(b) for b in headers) | set(id(b) for b in notes)

    # Build list of all labeled blocks with their x0 positions
    labeled_blocks = [b for b in blocks if id(b) in labeled]
    if not labeled_blocks:
        log.debug("group_notes_columns: no labeled blocks found")
        return []

    log.debug(
        "group_notes_columns: %d headers, %d notes blocks",
        len(headers),
        len(notes),
    )

    # Sort by x0 for clustering
    labeled_blocks.sort(key=lambda b: b.bbox()[0])

    # Cluster by x0 proximity.
    # Running-mean mode: compare each block's x0 to the cluster's running
    # mean x0.  This is more robust than consecutive-pair comparison
    # because a single indented note won't fracture the cluster.
    use_running_mean = cfg.notes_column_running_mean if cfg else True

    x0_clusters: List[List[BlockCluster]] = [[labeled_blocks[0]]]
    x0_sums: List[float] = [labeled_blocks[0].bbox()[0]]  # running sum per cluster
    for blk in labeled_blocks[1:]:
        cur_x0 = blk.bbox()[0]
        if use_running_mean:
            cluster_mean_x0 = x0_sums[-1] / len(x0_clusters[-1])
            if abs(cur_x0 - cluster_mean_x0) > x_tolerance:
                x0_clusters.append([blk])
                x0_sums.append(cur_x0)
            else:
                x0_clusters[-1].append(blk)
                x0_sums[-1] += cur_x0
        else:
            prev_x0 = x0_clusters[-1][-1].bbox()[0]
            if abs(cur_x0 - prev_x0) > x_tolerance:
                x0_clusters.append([blk])
                x0_sums.append(cur_x0)
            else:
                x0_clusters[-1].append(blk)
                x0_sums[-1] += cur_x0

    with _open_debug(debug_path) as dbg:
        dbg.write(
            f"\n[DEBUG] group_notes_columns (visual-column + reading-order): "
            f"{len(headers)} headers, {len(notes)} notes blocks, "
            f"{len(x0_clusters)} visual-columns\n"
        )

        columns: List[NotesColumn] = []
        # Track the last *named* (has-a-header) column in reading order
        # so orphan visual columns can be linked as continuations.
        last_named_col: NotesColumn | None = None
        # Counter for sub-column indices per parent group
        group_sub_index: dict[str, int] = {}  # group_id -> next sub-index (2, 3, …)
        group_id_counter = 0

        for ci, cluster in enumerate(x0_clusters):
            # Sort cluster blocks top-to-bottom
            cluster.sort(key=lambda b: b.bbox()[1])

            # Does this visual column start with a header?
            first_is_header = cluster and getattr(cluster[0], "is_header", False)

            dbg.write(
                f"[DEBUG]   visual-column {ci}: {len(cluster)} blocks, "
                f"x0 range [{cluster[0].bbox()[0]:.1f} .. {cluster[-1].bbox()[0]:.1f}]"
                f"{' (starts with header)' if first_is_header else ''}\n"
            )

            active_col: NotesColumn | None = None

            for blk in cluster:
                bb = blk.bbox()

                if getattr(blk, "is_header", False):
                    header_text = _block_first_row_text(blk)
                    dbg.write(
                        f"[DEBUG]     Header '{header_text}' "
                        f"x0={bb[0]:.1f} y={bb[1]:.1f}\n"
                    )
                    active_col = NotesColumn(page=blk.page, header=blk, notes_blocks=[])
                    columns.append(active_col)
                    last_named_col = active_col

                elif getattr(blk, "is_notes", False):
                    if active_col is None:
                        # No header yet in this visual column.
                        # Link to last_named_col as a sub-column if available.
                        if last_named_col is not None:
                            # Create a sub-column linked to the parent
                            parent_hdr = (
                                _block_first_row_text(last_named_col.header)
                                if last_named_col.header
                                else "unnamed"
                            )
                            if last_named_col.column_group_id is None:
                                last_named_col.column_group_id = (
                                    f"notes_group_{group_id_counter}"
                                )
                                group_id_counter += 1
                                group_sub_index[last_named_col.column_group_id] = 2

                            grp = last_named_col.column_group_id
                            sub_idx = group_sub_index.get(grp, 2)
                            group_sub_index[grp] = sub_idx + 1

                            active_col = NotesColumn(
                                page=blk.page,
                                header=None,
                                notes_blocks=[],
                                column_group_id=grp,
                                continues_from=parent_hdr,
                            )
                            columns.append(active_col)
                            dbg.write(
                                f"[DEBUG]     Continuation sub-column "
                                f"(continues '{parent_hdr}', sub={sub_idx})\n"
                            )
                        else:
                            active_col = NotesColumn(
                                page=blk.page, header=None, notes_blocks=[]
                            )
                            columns.append(active_col)
                            dbg.write(
                                "[DEBUG]     Orphan column opened (no header yet)\n"
                            )

                    active_col.notes_blocks.append(blk)
                    note_text = _block_first_row_text(blk)
                    dbg.write(
                        f"[DEBUG]       +note x0={bb[0]:.1f} y={bb[1]:.1f} "
                        f"'{note_text[:60]}'\n"
                    )

        # Summary
        for col in columns:
            hdr = _block_first_row_text(col.header) if col.header else "(orphan)"
            group_info = f" group={col.column_group_id}" if col.column_group_id else ""
            cont_info = (
                f" continues='{col.continues_from}'" if col.continues_from else ""
            )
            dbg.write(
                f"[DEBUG] Column '{hdr}': {len(col.notes_blocks)} notes{group_info}{cont_info}\n"
            )

    return columns


# ── Internal helpers for continuation linking ──────────────────────────


def _get_last_block_text(col: NotesColumn) -> str:
    """Get the full text of the last notes block in a column."""
    if not col.notes_blocks:
        return ""
    last_block = col.notes_blocks[-1]
    texts = []
    for row in last_block.rows:
        row_texts = [b.text for b in sorted(row.boxes, key=lambda b: b.x0) if b.text]
        texts.extend(row_texts)
    return " ".join(texts).strip()


def _get_first_block_text(col: NotesColumn) -> str:
    """Get the full text of the first notes block in a column."""
    if not col.notes_blocks:
        return ""
    first_block = col.notes_blocks[0]
    texts = []
    for row in first_block.rows:
        row_texts = [b.text for b in sorted(row.boxes, key=lambda b: b.x0) if b.text]
        texts.extend(row_texts)
    return " ".join(texts).strip()


def _extract_note_numbers(col: NotesColumn) -> list[int]:
    """Extract all note numbers from a column's notes blocks."""
    note_re = _NOTE_CAPTURE_RE
    numbers = []
    for block in col.notes_blocks:
        if not block.rows:
            continue
        first_row = block.rows[0]
        texts = [b.text for b in sorted(first_row.boxes, key=lambda b: b.x0) if b.text]
        if texts:
            first_text = texts[0]
            match = note_re.match(first_text)
            if match:
                numbers.append(int(match.group(1)))
    return sorted(numbers)


def _text_ends_incomplete(text: str) -> bool:
    """Check if text ends without terminal punctuation (likely mid-sentence)."""
    if not text:
        return False
    text = text.strip()
    # Terminal punctuation that indicates complete sentence
    terminal = {".", "!", "?", ":", ";"}
    return text[-1] not in terminal


def _get_first_row_text(block: BlockCluster) -> str:
    """Get the text of the first row of a block."""
    if not block.rows:
        return ""
    first_row = block.rows[0]
    texts = [b.text for b in sorted(first_row.boxes, key=lambda b: b.x0) if b.text]
    return " ".join(texts)


def _text_starts_as_continuation(text: str) -> bool:
    """Check if text appears to be a sentence continuation (not a new note)."""
    if not text:
        return False
    text = text.strip().upper()
    note_re = _NOTE_SIMPLE_RE
    # If it starts with a note number, it's not a continuation
    if note_re.match(text):
        return False
    # If it starts with lowercase-like words or common continuation words
    # (Since all our text is uppercase, we check for sentence-continuation patterns)
    # A continuation typically won't start with note-like patterns
    return True


def _link_explicit_continuations(
    columns: List[NotesColumn],
    base_to_columns: dict,
    dbg,
) -> int:
    """Link columns sharing the same base header (e.g. SITE NOTES / SITE NOTES (CONT'D)).

    Returns the next available group_id_counter value.
    """
    group_id_counter = 0
    for base, cols in base_to_columns.items():
        if len(cols) < 2:
            continue

        cols.sort(key=lambda c: c.bbox()[1] if c.header else float("inf"))

        group_id = f"notes_group_{group_id_counter}"
        group_id_counter += 1

        dbg.write(f"[DEBUG] Found column group '{base}' with {len(cols)} columns:\n")

        parent_header_text = None
        for i, col in enumerate(cols):
            col.column_group_id = group_id
            header_text = col.header_text()

            if col.is_continuation():
                col.continues_from = parent_header_text
                dbg.write(
                    f"[DEBUG]   [{i}] '{header_text}' (continues from '{parent_header_text}')\n"
                )
            else:
                parent_header_text = header_text
                dbg.write(f"[DEBUG]   [{i}] '{header_text}' (primary)\n")

    linked_count = sum(1 for c in columns if c.column_group_id is not None)
    continuation_count = sum(1 for c in columns if c.continues_from is not None)
    dbg.write(
        f"[DEBUG] Explicit linking: {linked_count} columns in groups, {continuation_count} continuations\n"
    )
    return group_id_counter


def _link_snake_continuations(
    named_columns: List[NotesColumn],
    orphan_columns: List[NotesColumn],
    group_id_counter: int,
    dbg,
) -> int:
    """Link orphan columns that continue a named column's incomplete text (snake layout).

    Returns the updated group_id_counter.
    """
    dbg.write(f"\n[DEBUG] Checking for implicit snake-column continuations...\n")

    for named_col in named_columns:
        if not named_col.notes_blocks:
            continue

        named_bbox = named_col.bbox()
        named_x0, named_y0, named_x1, named_y1 = named_bbox
        last_text = _get_last_block_text(named_col)

        dbg.write(
            f"[DEBUG] Checking '{named_col.header_text()}' for snake continuation\n"
        )
        dbg.write(
            f"[DEBUG]   Last text: '{last_text[-80:] if len(last_text) > 80 else last_text}'\n"
        )

        if not _text_ends_incomplete(last_text):
            dbg.write(f"[DEBUG]   Last text ends complete, skipping\n")
            continue

        dbg.write(f"[DEBUG]   Last text ends incomplete, looking for continuation\n")

        best_orphan = None
        best_score = float("inf")

        for orphan in orphan_columns:
            if orphan.column_group_id is not None:
                continue

            orphan_bbox = orphan.bbox()
            orphan_x0, orphan_y0, orphan_x1, orphan_y1 = orphan_bbox

            x_gap = orphan_x0 - named_x1
            is_right_of = x_gap > -50
            is_above = orphan_y0 < named_y0

            dbg.write(
                f"[DEBUG]   Orphan at x0={orphan_x0:.1f}, y0={orphan_y0:.1f}: "
                f"x_gap={x_gap:.1f}, is_right={is_right_of}, is_above={is_above}\n"
            )

            if not is_right_of:
                continue

            first_text = _get_first_block_text(orphan)
            if not _text_starts_as_continuation(first_text):
                dbg.write(
                    f"[DEBUG]   Orphan first text doesn't look like continuation: '{first_text[:50]}...'\n"
                )
                continue

            score = x_gap + orphan_y0 / 10
            if score < best_score:
                best_score = score
                best_orphan = orphan

        if best_orphan is not None:
            if named_col.column_group_id is None:
                named_col.column_group_id = f"notes_group_{group_id_counter}"
                group_id_counter += 1

            best_orphan.column_group_id = named_col.column_group_id
            best_orphan.continues_from = named_col.header_text()

            dbg.write(
                f"[DEBUG]   LINKED orphan to '{named_col.header_text()}' "
                f"(group={named_col.column_group_id})\n"
            )

    return group_id_counter


def _link_by_note_numbering(
    named_columns: List[NotesColumn],
    orphan_columns: List[NotesColumn],
    group_id_counter: int,
    dbg,
) -> int:
    """Link orphan columns whose note numbers continue a named column's sequence.

    Returns the updated group_id_counter.
    """
    dbg.write(f"\n[DEBUG] Checking note numbering sequences...\n")

    named_with_notes = []
    for named_col in named_columns:
        if not named_col.notes_blocks:
            continue
        named_numbers = _extract_note_numbers(named_col)
        if not named_numbers:
            continue
        named_with_notes.append((named_col, max(named_numbers), named_numbers))
        dbg.write(
            f"[DEBUG] '{named_col.header_text()}' has notes: {named_numbers}, max={max(named_numbers)}\n"
        )

    for orphan in orphan_columns:
        if orphan.column_group_id is not None:
            continue

        orphan_numbers = _extract_note_numbers(orphan)
        if not orphan_numbers:
            continue

        min_orphan_note = min(orphan_numbers)
        orphan_bbox = orphan.bbox()
        orphan_x0 = orphan_bbox[0]

        dbg.write(
            f"[DEBUG]   Orphan has notes: {orphan_numbers}, min={min_orphan_note}\n"
        )

        best_match = None
        best_gap = float("inf")

        for named_col, max_named_note, named_numbers in named_with_notes:
            named_bbox = named_col.bbox()
            named_x1 = named_bbox[2]

            if min_orphan_note <= max_named_note:
                continue

            note_gap = min_orphan_note - max_named_note

            if note_gap > 3:
                continue

            x_gap = orphan_x0 - named_x1
            is_right_or_similar = x_gap > -100

            if not is_right_or_similar:
                continue

            dbg.write(
                f"[DEBUG]     Candidate '{named_col.header_text()}' max={max_named_note}, "
                f"note_gap={note_gap}, x_gap={x_gap:.1f}\n"
            )

            if note_gap < best_gap:
                best_gap = note_gap
                best_match = named_col

        if best_match is not None:
            if best_match.column_group_id is None:
                best_match.column_group_id = f"notes_group_{group_id_counter}"
                group_id_counter += 1

            orphan.column_group_id = best_match.column_group_id
            orphan.continues_from = best_match.header_text()

            dbg.write(
                f"[DEBUG]   LINKED by sequence: orphan notes {orphan_numbers} "
                f"continue '{best_match.header_text()}' (gap={best_gap})\n"
            )

    return group_id_counter


def _capture_leading_text(
    columns: List[NotesColumn],
    orphan_columns: List[NotesColumn],
    blocks: List[BlockCluster],
    x_tolerance: float,
    dbg,
) -> None:
    """Attach unassigned text blocks above orphan columns' first note as leading text."""
    dbg.write(f"\n[DEBUG] Checking for leading continuation text in linked orphans\n")

    assigned_block_ids = set()
    for col in columns:
        for blk in col.notes_blocks:
            assigned_block_ids.add(id(blk))
        if col.header is not None:
            assigned_block_ids.add(id(col.header))

    for orphan in orphan_columns:
        if orphan.continues_from is None:
            continue
        if not orphan.notes_blocks:
            continue

        first_notes_block = orphan.notes_blocks[0]
        fnb_x0, fnb_y0, fnb_x1, fnb_y1 = first_notes_block.bbox()
        orphan_bbox = orphan.bbox()
        orphan_x0 = orphan_bbox[0]

        dbg.write(
            f"[DEBUG]   Orphan continuing '{orphan.continues_from}': "
            f"first note at y={fnb_y0:.1f}, x={orphan_x0:.1f}\n"
        )

        leading_blocks = []
        for blk in blocks:
            if id(blk) in assigned_block_ids:
                continue
            if getattr(blk, "is_header", False):
                continue
            if blk.is_table:
                continue

            bx0, by0, bx1, by1 = blk.bbox()

            if by1 >= fnb_y0:
                continue

            if abs(bx0 - orphan_x0) > x_tolerance:
                continue

            if not blk.rows:
                continue

            dbg.write(
                f"[DEBUG]     Found leading text at y={by0:.1f}-{by1:.1f}: "
                f"'{_get_first_row_text(blk)[:50]}...'\n"
            )
            leading_blocks.append(blk)

        if leading_blocks:
            leading_blocks.sort(key=lambda b: b.bbox()[1])
            orphan.notes_blocks = leading_blocks + orphan.notes_blocks
            dbg.write(
                f"[DEBUG]     Added {len(leading_blocks)} leading text block(s) to orphan\n"
            )


# ── Continuation linking ───────────────────────────────────────────────


def link_continued_columns(
    columns: List[NotesColumn],
    blocks: List[BlockCluster] = None,
    x_tolerance: float = 50.0,
    debug_path: str = None,
    cfg: GroupingConfig | None = None,
) -> None:
    """
    Detect and link columns that are continuations of each other.

    Handles two cases:
    1. Explicit continuations: "SITE NOTES" and "SITE NOTES (CONT'D)" headers
    2. Implicit continuations: Snake columns where text wraps to next column
       without a continuation header

    If `blocks` is provided, also finds "leading continuation text" - text blocks
    that appear above the first numbered note in a continuation column and adds
    them to that column's notes_blocks.

    This modifies columns in-place, setting:
    - column_group_id: A shared identifier for linked columns
    - continues_from: The header text of the parent column (for continuations)
    """
    if cfg is not None:
        x_tolerance = cfg.grouping_link_x_tolerance

    # Build a map of base header text -> list of columns
    base_to_columns: dict = {}
    for col in columns:
        if col.header is None:
            continue
        base = col.base_header_text()
        if base:
            if base not in base_to_columns:
                base_to_columns[base] = []
            base_to_columns[base].append(col)

    with _open_debug(debug_path) as dbg:
        dbg.write(
            f"\n[DEBUG] link_continued_columns: checking {len(columns)} columns\n"
        )

        gid = _link_explicit_continuations(columns, base_to_columns, dbg)

        named_columns = [c for c in columns if c.header is not None]
        orphan_columns = [c for c in columns if c.header is None]

        gid = _link_snake_continuations(named_columns, orphan_columns, gid, dbg)
        gid = _link_by_note_numbering(named_columns, orphan_columns, gid, dbg)

        if blocks is not None:
            _capture_leading_text(columns, orphan_columns, blocks, x_tolerance, dbg)

        # Final summary
        linked_count = sum(1 for c in columns if c.column_group_id is not None)
        continuation_count = sum(1 for c in columns if c.continues_from is not None)
        dbg.write(
            f"[DEBUG] Final: {linked_count} columns in groups, {continuation_count} continuations\n"
        )
