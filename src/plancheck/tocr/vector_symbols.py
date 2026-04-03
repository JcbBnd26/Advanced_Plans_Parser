"""Vector symbol recovery — inject CAD-vectorised symbols into the TOCR stream.

CAD-generated PDFs (AutoCAD, Revit, MicroStation) often render certain
symbols as vector graphics rather than text characters.  The text layer
might read ``2 34`` while the page visually shows ``2 3/4"`` because
the fraction slash is a diagonal line *path*, not a text glyph.

This module scans the raw vector graphics already present in
:class:`~plancheck.ingest.ingest.PageContext` (lines, rects, curves)
and matches orphan symbol-sized graphics to gaps in the text token
stream.  Matched candidates are injected as new
:class:`~plancheck.models.tokens.GlyphBox` entries with
``origin="vector_symbol"``.

Pipeline position::

    ingest → tocr → **vector_symbol_recovery** → prune/deskew → …

Public API
----------
- :func:`recover_vector_symbols` — main entry point
"""

from __future__ import annotations

import logging
import math
from statistics import median
from typing import Any, Dict, List, Optional, Tuple

from ..config import GroupingConfig
from ..models.tokens import GlyphBox

log = logging.getLogger(__name__)

# Re-usable type aliases
_Bbox = Tuple[float, float, float, float]
_LineDct = Dict[str, Any]
_CurveDct = Dict[str, Any]

# ── Digit detection ────────────────────────────────────────────────────

_DIGIT_CHARS = frozenset("0123456789")


def _has_digit(text: str) -> bool:
    """Return True if *text* contains at least one ASCII digit."""
    return any(c in _DIGIT_CHARS for c in text)


# ── Coordinate helpers for pdfplumber dicts ────────────────────────────
# pdfplumber uses  x0 / top / x1 / bottom  (not y0 / y1).


def _line_bbox(d: _LineDct) -> _Bbox:
    """Extract normalised ``(x0, y0, x1, y1)`` from a pdfplumber line dict."""
    x0 = min(d["x0"], d["x1"])
    y0 = min(d["top"], d["bottom"])
    x1 = max(d["x0"], d["x1"])
    y1 = max(d["top"], d["bottom"])
    return (x0, y0, x1, y1)


def _curve_bbox(d: _CurveDct) -> _Bbox:
    """Derive bbox from a pdfplumber curve dict's ``pts`` list."""
    pts: list[tuple[float, float]] = d.get("pts", [])
    if not pts:
        # Fallback to explicit keys if pts is empty.
        return (d.get("x0", 0), d.get("top", 0), d.get("x1", 0), d.get("bottom", 0))
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return (min(xs), min(ys), max(xs), max(ys))


def _bbox_dims(bb: _Bbox) -> Tuple[float, float]:
    """Return ``(width, height)`` of a bbox."""
    return (bb[2] - bb[0], bb[3] - bb[1])


def _bboxes_v_overlap(a: _Bbox, b: _Bbox) -> bool:
    """Return True if *a* and *b* share any vertical extent."""
    return a[1] < b[3] and b[1] < a[3]


# ── Angle / length helpers ─────────────────────────────────────────────


def _line_angle_deg(d: _LineDct) -> float:
    """Compute the angle (0–90°) of a pdfplumber line from horizontal."""
    dx = abs(d["x1"] - d["x0"])
    dy = abs(d["bottom"] - d["top"])
    if dx == 0 and dy == 0:
        return 0.0
    return math.degrees(math.atan2(dy, dx))


def _line_length(d: _LineDct) -> float:
    """Euclidean length of a pdfplumber line segment."""
    dx = d["x1"] - d["x0"]
    dy = d["bottom"] - d["top"]
    return math.hypot(dx, dy)


# ── Spatial helpers ────────────────────────────────────────────────────


def _estimate_char_width(tokens: List[GlyphBox]) -> float:
    """Estimate a representative character width from digit-bearing tokens.

    Falls back to 6.0 pts (a reasonable default for 10-12 pt fonts) when
    no digit tokens are available.
    """
    widths: list[float] = []
    for t in tokens:
        if _has_digit(t.text) and len(t.text) > 0 and t.width() > 0:
            widths.append(t.width() / len(t.text))
    return median(widths) if widths else 6.0


def _estimate_font_size(tokens: List[GlyphBox]) -> float:
    """Median font_size from tokens with font_size > 0, fallback 10.0."""
    sizes = [t.font_size for t in tokens if t.font_size > 0]
    return median(sizes) if sizes else 10.0


def _find_digit_neighbours(
    bbox: _Bbox,
    tokens: List[GlyphBox],
    max_gap: float,
) -> Tuple[Optional[GlyphBox], Optional[GlyphBox]]:
    """Find the nearest digit-bearing token to the left and right.

    Only tokens whose vertical span overlaps *bbox* are considered.

    Returns ``(left_neighbour, right_neighbour)`` — either may be None.
    """
    left: Optional[GlyphBox] = None
    right: Optional[GlyphBox] = None
    left_dist = float("inf")
    right_dist = float("inf")

    for t in tokens:
        if not _has_digit(t.text):
            continue
        tb = t.bbox()
        if not _bboxes_v_overlap(bbox, tb):
            continue
        # Left neighbour: token's right edge is left of bbox's left edge.
        gap_l = bbox[0] - tb[2]
        if 0 <= gap_l <= max_gap and gap_l < left_dist:
            left_dist = gap_l
            left = t
        # Right neighbour: token's left edge is right of bbox's right edge.
        gap_r = tb[0] - bbox[2]
        if 0 <= gap_r <= max_gap and gap_r < right_dist:
            right_dist = gap_r
            right = t

    return left, right


def _token_covers_position(
    tokens: List[GlyphBox],
    bbox: _Bbox,
    symbol: str,
) -> bool:
    """Return True if an existing token at *bbox* already contains *symbol*."""
    for t in tokens:
        if symbol not in t.text:
            continue
        tb = t.bbox()
        # Check overlap: centres within half a char-width of each other.
        cx = (bbox[0] + bbox[2]) / 2.0
        tcx = (tb[0] + tb[2]) / 2.0
        tw = tb[2] - tb[0]
        if tw <= 0:
            continue
        if abs(cx - tcx) < tw and _bboxes_v_overlap(bbox, tb):
            return True
    return False


# ── GlyphBox factory ──────────────────────────────────────────────────


def _make_symbol_glyph(
    text: str,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    page: int,
    font_size: float,
) -> GlyphBox:
    """Create a GlyphBox for an injected vector symbol."""
    return GlyphBox(
        page=page,
        x0=x0,
        y0=y0,
        x1=x1,
        y1=y1,
        text=text,
        origin="vector_symbol",
        font_size=font_size,
        confidence=0.95,
    )


# ── Size filter ────────────────────────────────────────────────────────


def _is_symbol_sized(bbox: _Bbox, max_size: float) -> bool:
    """Return True if bbox dimensions are within symbol scale.

    Allows zero-height bboxes (perfectly horizontal lines) as long as
    width is positive and within *max_size*.
    """
    w, h = _bbox_dims(bbox)
    return 0 < w <= max_size and 0 <= h <= max_size


# ── Circle detection ──────────────────────────────────────────────────


def _is_small_circle(
    curve: _CurveDct,
    max_size: float,
    max_aspect: float,
) -> bool:
    """Return True if *curve* looks like a small circle or arc.

    Checks:
    - At least 4 control points (Bézier circle approximation)
    - Bounding box roughly square (aspect ratio within *max_aspect*)
    - Both dimensions ≤ *max_size*
    """
    pts = curve.get("pts", [])
    if len(pts) < 4:
        return False
    bb = _curve_bbox(curve)
    w, h = _bbox_dims(bb)
    if w <= 0 or h <= 0:
        return False
    if w > max_size or h > max_size:
        return False
    aspect = w / h
    return (1.0 / max_aspect) <= aspect <= max_aspect


# ── Classifiers ────────────────────────────────────────────────────────


def _classify_as_slash(
    line: _LineDct,
    tokens: List[GlyphBox],
    char_width: float,
    page: int,
    cfg: GroupingConfig,
) -> Optional[GlyphBox]:
    """Classify a pdfplumber line as a fraction slash ``/``.

    A slash candidate is a short diagonal line sitting between two
    digit-bearing text tokens on the same text line.
    """
    bb = _line_bbox(line)
    if not _is_symbol_sized(bb, cfg.tocr_vector_symbol_max_size):
        return None

    length = _line_length(line)
    if length < 3.0:
        return None

    angle = _line_angle_deg(line)
    if not (cfg.tocr_vector_symbol_slash_angle_min
            <= angle
            <= cfg.tocr_vector_symbol_slash_angle_max):
        return None

    max_gap = char_width * cfg.tocr_vector_symbol_proximity
    left, right = _find_digit_neighbours(bb, tokens, max_gap)
    if left is None or right is None:
        return None

    if _token_covers_position(tokens, bb, "/"):
        return None

    avg_fs = (left.font_size + right.font_size) / 2.0
    if avg_fs <= 0:
        avg_fs = bb[3] - bb[1]  # fallback: line height

    return _make_symbol_glyph("/", bb[0], bb[1], bb[2], bb[3], page, avg_fs)


def _classify_as_degree(
    curve: _CurveDct,
    tokens: List[GlyphBox],
    char_width: float,
    font_size: float,
    page: int,
    cfg: GroupingConfig,
) -> Optional[GlyphBox]:
    """Classify a small circle-like curve as a degree symbol ``°``.

    A degree symbol is a single small closed curve at superscript height
    relative to the preceding digit token.
    """
    circle_max_dim = font_size * 0.55
    if not _is_small_circle(curve, circle_max_dim, cfg.tocr_vector_symbol_circle_max_aspect):
        return None

    bb = _curve_bbox(curve)
    w, h = _bbox_dims(bb)

    # Must be noticeably smaller than text height.
    if h > font_size * 0.55:
        return None

    max_gap = char_width * cfg.tocr_vector_symbol_proximity
    left, _ = _find_digit_neighbours(bb, tokens, max_gap)
    if left is None:
        return None

    # Must be at superscript height: vertical centre above the midpoint
    # of the adjacent digit token.
    symbol_cy = (bb[1] + bb[3]) / 2.0
    digit_mid = (left.y0 + left.y1) / 2.0
    if symbol_cy >= digit_mid:
        return None

    if _token_covers_position(tokens, bb, "°"):
        return None

    return _make_symbol_glyph("°", bb[0], bb[1], bb[2], bb[3], page, left.font_size)


def _classify_as_percent(
    lines: List[_LineDct],
    curves: List[_CurveDct],
    tokens: List[GlyphBox],
    char_width: float,
    font_size: float,
    page: int,
    cfg: GroupingConfig,
) -> List[GlyphBox]:
    """Find percent signs built from two circles + a diagonal line.

    Returns a (possibly empty) list of GlyphBox entries for ``%``.
    """
    results: List[GlyphBox] = []
    max_dim = font_size * 0.55
    max_aspect = cfg.tocr_vector_symbol_circle_max_aspect
    max_gap = char_width * cfg.tocr_vector_symbol_proximity

    # Collect small circles from curves.
    circles: list[tuple[_CurveDct, _Bbox]] = []
    for c in curves:
        if _is_small_circle(c, max_dim, max_aspect):
            circles.append((c, _curve_bbox(c)))

    if len(circles) < 2:
        return results

    # Track used circles to avoid double-matching.
    used_circles: set[int] = set()

    for i, (c1, bb1) in enumerate(circles):
        if i in used_circles:
            continue
        for j, (c2, bb2) in enumerate(circles):
            if j <= i or j in used_circles:
                continue
            # Two circles must be within 1.5 * font_size of each other.
            dist = math.hypot(
                (bb1[0] + bb1[2]) / 2 - (bb2[0] + bb2[2]) / 2,
                (bb1[1] + bb1[3]) / 2 - (bb2[1] + bb2[3]) / 2,
            )
            if dist > font_size * 1.8:
                continue

            # Combined bbox of the two circles.
            combo_bb = (
                min(bb1[0], bb2[0]),
                min(bb1[1], bb2[1]),
                max(bb1[2], bb2[2]),
                max(bb1[3], bb2[3]),
            )

            # Look for a diagonal line between/overlapping the two circles.
            has_diag = False
            for ln in lines:
                lbb = _line_bbox(ln)
                if not _is_symbol_sized(lbb, cfg.tocr_vector_symbol_max_size):
                    continue
                angle = _line_angle_deg(ln)
                if not (30.0 <= angle <= 85.0):
                    continue
                # Line must be roughly within the combined circle bbox
                # (with some tolerance).
                tol = font_size * 0.3
                if (lbb[0] >= combo_bb[2] + tol or lbb[2] <= combo_bb[0] - tol
                        or lbb[1] >= combo_bb[3] + tol or lbb[3] <= combo_bb[1] - tol):
                    continue
                has_diag = True
                break

            if not has_diag:
                continue

            # Expand combo to include the diagonal line's extent.
            pct_bb = combo_bb

            # Must have a digit token to the left.
            left, _ = _find_digit_neighbours(pct_bb, tokens, max_gap)
            if left is None:
                continue

            if _token_covers_position(tokens, pct_bb, "%"):
                continue

            results.append(
                _make_symbol_glyph(
                    "%", pct_bb[0], pct_bb[1], pct_bb[2], pct_bb[3],
                    page, left.font_size,
                )
            )
            used_circles.add(i)
            used_circles.add(j)
            break  # one percent per circle pair

    return results


def _classify_as_hash(
    lines: List[_LineDct],
    tokens: List[GlyphBox],
    char_width: float,
    font_size: float,
    page: int,
    cfg: GroupingConfig,
) -> List[GlyphBox]:
    """Find hash ``#`` signs built from ~4 intersecting line segments.

    Returns a (possibly empty) list of GlyphBox entries for ``#``.
    """
    results: List[GlyphBox] = []
    max_size = cfg.tocr_vector_symbol_max_size
    max_gap = char_width * cfg.tocr_vector_symbol_proximity

    # Collect symbol-sized lines.
    small_lines: list[tuple[_LineDct, _Bbox, float]] = []
    for ln in lines:
        bb = _line_bbox(ln)
        if _is_symbol_sized(bb, max_size):
            small_lines.append((ln, bb, _line_angle_deg(ln)))

    if len(small_lines) < 4:
        return results

    # Cluster nearby lines based on bbox centre proximity.
    used: set[int] = set()
    for i, (ln_i, bb_i, ang_i) in enumerate(small_lines):
        if i in used:
            continue
        cluster_idx = [i]
        cx_i = (bb_i[0] + bb_i[2]) / 2
        cy_i = (bb_i[1] + bb_i[3]) / 2
        for j, (ln_j, bb_j, ang_j) in enumerate(small_lines):
            if j <= i or j in used:
                continue
            cx_j = (bb_j[0] + bb_j[2]) / 2
            cy_j = (bb_j[1] + bb_j[3]) / 2
            if math.hypot(cx_i - cx_j, cy_i - cy_j) <= font_size * 1.2:
                cluster_idx.append(j)
        if len(cluster_idx) < 4:
            continue

        # Need a mix: at least 2 roughly horizontal (0-25°) and 2 roughly
        # vertical or diagonal (55-90°).
        angles = [small_lines[k][2] for k in cluster_idx]
        n_horiz = sum(1 for a in angles if a <= 25)
        n_vert = sum(1 for a in angles if a >= 55)
        if n_horiz < 2 or n_vert < 2:
            continue

        # Combined bbox.
        combo_bb = (
            min(small_lines[k][1][0] for k in cluster_idx),
            min(small_lines[k][1][1] for k in cluster_idx),
            max(small_lines[k][1][2] for k in cluster_idx),
            max(small_lines[k][1][3] for k in cluster_idx),
        )

        # Hash typically precedes a digit (rebar: #4, #5).
        _, right = _find_digit_neighbours(combo_bb, tokens, max_gap)
        if right is None:
            continue
        if _token_covers_position(tokens, combo_bb, "#"):
            continue

        results.append(
            _make_symbol_glyph(
                "#", combo_bb[0], combo_bb[1], combo_bb[2], combo_bb[3],
                page, right.font_size,
            )
        )
        for k in cluster_idx:
            used.add(k)

    return results


def _classify_minus_lines(
    lines: List[_LineDct],
    tokens: List[GlyphBox],
    char_width: float,
    page: int,
    cfg: GroupingConfig,
) -> List[GlyphBox]:
    """Detect short horizontal lines that may be minus / en-dash signs.

    Returns a (possibly empty) list of GlyphBox entries for ``-``.
    """
    results: List[GlyphBox] = []
    max_size = cfg.tocr_vector_symbol_max_size
    max_gap = char_width * cfg.tocr_vector_symbol_proximity

    for ln in lines:
        bb = _line_bbox(ln)
        if not _is_symbol_sized(bb, max_size):
            continue
        length = _line_length(ln)
        if length < 2.5:
            continue
        angle = _line_angle_deg(ln)
        if angle > 10.0:
            continue  # Must be (near-)horizontal.

        left, right = _find_digit_neighbours(bb, tokens, max_gap)
        if left is None:
            continue  # At minimum need a digit on the left.

        # A true minus sits near the vertical midpoint of the adjacent text.
        # Reject lines at the very top (superscript) or bottom (underline /
        # tick mark) of the text line — these are drawing geometry.
        line_cy = (bb[1] + bb[3]) / 2.0
        digit_cy = (left.y0 + left.y1) / 2.0
        digit_h = left.height()
        if digit_h > 0 and abs(line_cy - digit_cy) > digit_h * 0.4:
            continue

        if _token_covers_position(tokens, bb, "-"):
            continue

        fs = left.font_size if left.font_size > 0 else (bb[3] - bb[1])
        results.append(_make_symbol_glyph("-", bb[0], bb[1], bb[2], bb[3], page, fs))

    return results


# ── Public entry point ─────────────────────────────────────────────────


def recover_vector_symbols(
    tokens: List[GlyphBox],
    lines: List[_LineDct],
    curves: List[_CurveDct],
    page_num: int,
    cfg: GroupingConfig,
) -> Tuple[List[GlyphBox], Dict[str, Any]]:
    """Scan vector graphics for symbol-sized shapes and inject matches.

    Parameters
    ----------
    tokens
        TOCR-extracted text tokens for one page.
    lines
        Raw pdfplumber line dicts (``x0, top, x1, bottom, …``).
    curves
        Raw pdfplumber curve dicts (``pts, …``).
    page_num
        Zero-based page index.
    cfg
        Pipeline configuration.

    Returns
    -------
    (augmented_tokens, diagnostics)
        *augmented_tokens* is the original list plus any injected symbol
        tokens (sorted by position).  *diagnostics* is a summary dict.
    """
    diag: Dict[str, Any] = {
        "vector_symbols_found": 0,
        "by_type": {},
        "candidates_rejected": 0,
        "rejection_reasons": {},
    }

    if not cfg.tocr_vector_symbols_enabled:
        return tokens, diag

    if not lines and not curves:
        return tokens, diag

    char_width = _estimate_char_width(tokens)
    font_size = _estimate_font_size(tokens)

    injected: List[GlyphBox] = []

    # Track consumed graphics by index to prevent cross-classifier
    # double-matching (e.g. a diagonal line counted as both slash and
    # percent-diagonal, or a circle counted as both degree and percent).
    used_line_idx: set[int] = set()
    used_curve_idx: set[int] = set()

    # ── Slash (/) ──────────────────────────────────────────────────
    for i, ln in enumerate(lines):
        g = _classify_as_slash(ln, tokens, char_width, page_num, cfg)
        if g is not None:
            injected.append(g)
            used_line_idx.add(i)

    # ── Degree (°) ─────────────────────────────────────────────────
    for i, c in enumerate(curves):
        g = _classify_as_degree(c, tokens, char_width, font_size, page_num, cfg)
        if g is not None:
            injected.append(g)
            used_curve_idx.add(i)

    # ── Percent (%) ────────────────────────────────────────────────
    remaining_lines = [ln for i, ln in enumerate(lines) if i not in used_line_idx]
    remaining_curves = [c for i, c in enumerate(curves) if i not in used_curve_idx]
    pct_results = _classify_as_percent(
        remaining_lines, remaining_curves, tokens, char_width, font_size, page_num, cfg,
    )
    injected.extend(pct_results)

    # ── Hash (#) ───────────────────────────────────────────────────
    hash_results = _classify_as_hash(
        remaining_lines, tokens, char_width, font_size, page_num, cfg,
    )
    injected.extend(hash_results)

    # ── Minus / en-dash (-) ────────────────────────────────────────
    minus_results = _classify_minus_lines(
        [ln for i, ln in enumerate(lines) if i not in used_line_idx],
        tokens, char_width, page_num, cfg,
    )
    injected.extend(minus_results)

    # ── Build diagnostics ──────────────────────────────────────────
    by_type: Dict[str, int] = {}
    for g in injected:
        by_type[g.text] = by_type.get(g.text, 0) + 1

    diag["vector_symbols_found"] = len(injected)
    diag["by_type"] = by_type

    log.info(
        "Vector symbol recovery: page %d — found %d symbols %s",
        page_num,
        len(injected),
        by_type,
    )

    if not injected:
        return tokens, diag

    if not cfg.tocr_vector_symbols_inject:
        log.info(
            "Vector symbol recovery: diagnostic-only mode — "
            "not injecting %d symbols",
            len(injected),
        )
        return tokens, diag

    # Merge + sort by reading order (top-to-bottom, left-to-right).
    merged = list(tokens) + injected
    merged.sort(key=lambda g: (g.y0, g.x0))

    return merged, diag
