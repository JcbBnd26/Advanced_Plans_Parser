"""Semantic labeling: tables, headers, notes, and suspect words.

This module handles block property detection:
- Table detection via regular row/column structure
- Header detection via caps, colon, bold, or large font
- Notes detection via numbered patterns
- Suspect word flagging for VOCR re-scanning
"""

from __future__ import annotations

import io
import logging
import re
from contextlib import contextmanager
from statistics import mean, pstdev
from typing import IO, List

from ..config import GroupingConfig
from ..models import BlockCluster

log = logging.getLogger(__name__)

__all__ = [
    "mark_tables",
    "mark_headers",
    "mark_notes",
    "flag_suspect_header_words",
]

# ── Pre-compiled patterns ──────────────────────────────────────────────
_NOTE_SIMPLE_RE = re.compile(r"^\d+\.")
_NOTE_BROAD_RE = re.compile(r"^(?:\d+\.|[A-Z]\.|[a-z]\.|\(\d+\)|\([A-Za-z]\))")
_HEADER_COLON_RE = re.compile(r"^[A-Z0-9\s\-\(\)\'\.\/]+: *$", re.ASCII)
_HEADER_CAPS_RE = re.compile(r"^[A-Z][A-Z0-9\s\-\(\)\'\.\/]{4,}$", re.ASCII)
_TITLE_BLOCK_RE = re.compile(
    r"OKLAHOMA DEPARTMENT|LEGEND AND |SHEET |^[A-Z]+-\d+$", re.ASCII
)


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


# ── Table detection ────────────────────────────────────────────────────


def mark_tables(blocks: List[BlockCluster], settings: GroupingConfig) -> None:
    """Flag blocks whose row structure resembles a table."""
    for blk in blocks:
        if len(blk.rows) < 2:
            blk.is_table = False
            continue
        col_counts: List[int] = []
        gaps_sets: List[List[float]] = []
        for row in blk.rows:
            xs = [b.x0 for b in row.boxes] + [b.x1 for b in row.boxes]
            xs_sorted = sorted(xs)
            col_counts.append(len(row.boxes))
            gaps = [xs_sorted[i + 1] - xs_sorted[i] for i in range(len(xs_sorted) - 1)]
            if gaps:
                gaps_sets.append(gaps)
        if not gaps_sets:
            blk.is_table = False
            continue
        flat_gaps = [g for gaps in gaps_sets for g in gaps if g > 0]
        if not flat_gaps:
            blk.is_table = False
            continue
        gap_mean = mean(flat_gaps)
        gap_cv = (pstdev(flat_gaps) / (gap_mean + 1e-6)) if len(flat_gaps) > 1 else 0.0
        col_mean = mean(col_counts) if col_counts else 0.0
        col_cv = (
            (pstdev(col_counts) / (col_mean + 1e-6)) if len(col_counts) > 1 else 0.0
        )
        blk.is_table = (
            gap_cv < settings.table_regular_tol and col_cv < settings.table_regular_tol
        )


# ── Header detection ───────────────────────────────────────────────────


def mark_headers(
    blocks: List[BlockCluster],
    debug_path: str = None,
    cfg: GroupingConfig | None = None,
) -> None:
    """Identify and label header blocks before notes grouping.

    Detection signals (any one is sufficient):
      1. ALL CAPS text ending with ":"  (classic plan header)
      2. ALL CAPS text + bold font      (bold section title)
      3. ALL CAPS text + font size > threshold × median  (large section title)

    Single-row blocks only.  Excluded phrases prevent false-positives.
    """
    # Classic: "GENERAL NOTES:", "EROSION CONTROL NOTES - GENERAL:", etc.
    header_colon_re = _HEADER_COLON_RE
    # Broader: ALL CAPS, ≥2 words, no numbered-note start
    header_caps_re = _HEADER_CAPS_RE
    excluded_phrases = {
        "BE USED ON THIS PROJECT:",
        "SHEET",
    }
    # Patterns that are title-block / sheet-label text, not section headers
    _title_block_re = _TITLE_BLOCK_RE

    # Tunables from config
    _large_mult = cfg.header_large_font_mult if cfg else 1.25
    _max_rows = cfg.header_max_rows if cfg else 3

    # Compute median font size across all boxes for relative-size check
    all_sizes = []
    for blk in blocks:
        for row in blk.rows:
            for b in row.boxes:
                sz = getattr(b, "font_size", 0.0)
                if sz > 0:
                    all_sizes.append(sz)
    median_size = sorted(all_sizes)[len(all_sizes) // 2] if all_sizes else 0.0

    with _open_debug(debug_path) as dbg:
        for i, blk in enumerate(blocks):
            # Preserve subheader labels from a prior split pass
            if getattr(blk, "label", None) == "note_column_subheader":
                continue
            blk.is_header = False
            if not blk.rows or len(blk.rows) > _max_rows:
                continue  # Headers are short — skip tall blocks

            text = _block_first_row_text(blk)
            if not text or text in excluded_phrases:
                continue
            if _title_block_re.search(text):
                continue  # Skip title-block / sheet-label text

            # --- Signal 1: ALL CAPS + colon ---
            has_colon = bool(header_colon_re.match(text))

            # --- Signal 2: bold font ---
            is_bold = any(
                getattr(b, "fontname", "").lower().find("bold") >= 0
                for b in blk.rows[0].boxes
            )

            # --- Signal 3: larger than median ---
            avg_size = 0.0
            sizes = [getattr(b, "font_size", 0.0) for b in blk.rows[0].boxes]
            sizes = [s for s in sizes if s > 0]
            if sizes:
                avg_size = sum(sizes) / len(sizes)
            is_large = median_size > 0 and avg_size > median_size * _large_mult

            # Must also be ALL CAPS (at least for signals 2 & 3)
            is_all_caps = bool(header_caps_re.match(text))

            # Accept: colon-pattern, OR (all caps + bold), OR (all caps + large)
            if has_colon or (is_all_caps and is_bold) or (is_all_caps and is_large):
                blk.is_header = True
                blk.label = "note_column_header"
                signals = []
                if has_colon:
                    signals.append("colon")
                if is_bold:
                    signals.append("bold")
                if is_large:
                    signals.append(f"large({avg_size:.1f}>{median_size:.1f})")
                dbg.write(
                    f"[DEBUG] Header block {i}: '{text}' [{', '.join(signals)}]\n"
                )

    # ── Post-pass: split header blocks with subtitle rows ───────────
    # If row-0 font size is noticeably larger than row-1+, the extra rows
    # are a subtitle / description — split them into a separate block.
    _font_drop_ratio = 0.85  # row-1 avg size < 85% of row-0 → split
    inserts: list[tuple[int, BlockCluster]] = []  # (insert_after_idx, new_block)
    for i, blk in enumerate(blocks):
        if not blk.is_header or len(blk.rows) < 2:
            continue
        r0_sizes = [getattr(b, "font_size", 0.0) for b in blk.rows[0].boxes]
        r0_sizes = [s for s in r0_sizes if s > 0]
        if not r0_sizes:
            continue
        r0_avg = sum(r0_sizes) / len(r0_sizes)
        r1_sizes = [
            getattr(b, "font_size", 0.0) for row in blk.rows[1:] for b in row.boxes
        ]
        r1_sizes = [s for s in r1_sizes if s > 0]
        if not r1_sizes:
            continue
        r1_avg = sum(r1_sizes) / len(r1_sizes)
        if r1_avg < r0_avg * _font_drop_ratio:
            # Split: keep only row 0 in the header block
            subtitle_rows = blk.rows[1:]
            blk.rows = blk.rows[:1]

            # Also split lines/_tokens if present (new pipeline)
            subtitle_lines: list = []
            shared_tokens = blk._tokens
            if blk.lines and shared_tokens and len(blk.lines) >= 2:
                # Row 0 corresponds to line 0; remaining lines → subtitle
                subtitle_lines = blk.lines[1:]
                blk.lines = blk.lines[:1]

            sub_block = BlockCluster(
                page=blk.page,
                rows=subtitle_rows,
                lines=subtitle_lines,
                _tokens=shared_tokens,
                label="note_column_subheader",
                parent_block_index=i,
            )
            inserts.append((i, sub_block))
            with _open_debug(debug_path) as dbg:
                sub_text = " ".join(b.text for row in subtitle_rows for b in row.boxes)
                dbg.write(
                    f"[DEBUG] Split header B{i}: subtitle row(s) "
                    f"(avg {r1_avg:.1f} < {r0_avg:.1f}*{_font_drop_ratio}) "
                    f"→ '{sub_text}'\n"
                )

    # Insert in reverse order so indices stay valid
    for idx, new_blk in reversed(inserts):
        blocks.insert(idx + 1, new_blk)


# ── Notes detection ────────────────────────────────────────────────────


def mark_notes(blocks: List[BlockCluster], debug_path: str = None) -> None:
    """Label notes blocks, skipping any block already labeled as header.

    Patterns recognised:
      - ``1.``, ``12.``  — numbered notes
      - ``A.``, ``B.``   — lettered notes
      - ``(1)``, ``(A)`` — parenthesised numbered/lettered notes
    """
    note_re = _NOTE_BROAD_RE
    with _open_debug(debug_path) as dbg:
        for i, blk in enumerate(blocks):
            blk.is_notes = False
            if getattr(blk, "is_header", False):
                continue  # Skip header blocks
            if blk.rows:
                first_row = blk.rows[0]
                texts = [
                    b.text
                    for b in sorted(first_row.boxes, key=lambda b: b.x0)
                    if b.text
                ]
                if texts:
                    row_text = " ".join(texts).strip()
                    # Check full row text first; if that fails, check
                    # individual word texts (handles margin reference
                    # numbers that precede the actual note number).
                    is_note_start = bool(note_re.match(row_text))
                    if not is_note_start:
                        for t in texts:
                            if note_re.match(t.strip()):
                                is_note_start = True
                                break
                    if is_note_start:
                        blk.is_notes = True
                        blk.label = "notes_block"


# ── Suspect word flagging ──────────────────────────────────────────────


def flag_suspect_header_words(
    blocks: List[BlockCluster],
    min_word_len: int = 10,
    debug_path: str | None = None,
) -> List["SuspectRegion"]:
    """Flag suspiciously long all-caps words in header blocks for VOCR inspection.

    After ``mark_headers`` has run, this function scans each header block's
    first row for words that are:
      - ALL CAPS
      - Longer than *min_word_len* characters
      - Not a common English / engineering word (checked against a short
        allow-list)

    These are likely **fused compound words** where a separator (``/``,
    ``-``, space) was lost during PDF authoring.  The returned
    ``SuspectRegion`` list records the **exact glyph bbox** so the VOCR
    symbol rectifier can re-OCR that region from the raster image.

    Returns:
        List of SuspectRegion, one per suspicious word.
    """
    from plancheck.models import SuspectRegion

    # Common long all-caps words that are NOT fused compounds
    _allowed_long = {
        "CONSTRUCTION",
        "DEPARTMENT",
        "INFORMATION",
        "TRANSPORTATION",
        "SPECIFICATIONS",
        "REINFORCEMENT",
        "REINFORCED",
        "REQUIREMENTS",
        "INSTALLATION",
        "GEOTECHNICAL",
        "APPLICATIONS",
        "CONTRACTOR",
        "SUBCONTRACTOR",
        "ABBREVIATIONS",
        "RESPONSIBILITY",
        "RESPONSIBLE",
        "DESCRIPTION",
        "DESCRIPTIONS",
        "MISCELLANEOUS",
        "ENVIRONMENTAL",
        "COMPRESSIVE",
        "REPLACEMENT",
        "DETERMINING",
        "RECOMMENDED",
        "DEMOLITION",
        "EXCAVATION",
        "STRUCTURAL",
        "FOUNDATION",
        "ELECTRICAL",
        "MECHANICAL",
        "EARTHWORK",
    }

    suspects: list[SuspectRegion] = []

    with _open_debug(debug_path) as dbg:
        dbg.write("\n[SUSPECT] --- Suspect header word scan ---\n")
        for i, blk in enumerate(blocks):
            if not getattr(blk, "is_header", False):
                continue
            if not blk.rows:
                continue

            header_text = _block_first_row_text(blk)
            for box in blk.rows[0].boxes:
                word = (box.text or "").strip()
                if len(word) < min_word_len:
                    continue
                # Must be all uppercase (allows digits mixed in)
                if not re.match(r"^[A-Z0-9]+$", word):
                    continue
                if word in _allowed_long:
                    continue

                sr = SuspectRegion(
                    page=blk.page,
                    x0=box.x0,
                    y0=box.y0,
                    x1=box.x1,
                    y1=box.y1,
                    word_text=word,
                    context=header_text,
                    reason=f"long_allcaps_word({len(word)}ch)",
                    source_label=getattr(blk, "label", ""),
                    block_index=i,
                )
                suspects.append(sr)
                dbg.write(
                    f"[SUSPECT] Block {i}: word='{word}' ({len(word)}ch)  "
                    f"bbox=({box.x0:.1f},{box.y0:.1f},{box.x1:.1f},{box.y1:.1f})  "
                    f"header='{header_text}'\n"
                )

        dbg.write(f"[SUSPECT] Total: {len(suspects)} suspect region(s)\n")
    return suspects
