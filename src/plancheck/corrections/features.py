"""Feature extraction for BlockCluster and region objects.

Converts pipeline objects into flat feature dicts suitable for
JSON serialisation and downstream classification.

Feature schema (37 keys):
  - 3 font metrics: font_size_pt, font_size_max_pt, font_size_min_pt
  - 8 text properties: is_all_caps, is_bold, token_count, row_count,
        text_length, avg_chars_per_token, unique_word_ratio,
        uppercase_word_frac, avg_word_length
  - 7 bbox fractions: x_frac, y_frac, x_center_frac, y_center_frac,
        width_frac, height_frac, aspect_ratio
  - 4 content flags: contains_digit, starts_with_digit, has_colon,
        has_period_after_num
  - 7 keyword indicators: kw_notes_pattern, kw_header_pattern,
        kw_legend_pattern, kw_abbreviation_pattern, kw_revision_pattern,
        kw_title_block_pattern, kw_detail_pattern
  - 1 neighbour: neighbor_count
  - 1 zone: zone (string, one-hot encoded by classifier)
  - 3 discriminative: text_density, x_dist_to_right_margin,
        line_width_variance
  - 2 OCR confidence: mean_token_confidence, min_token_confidence
"""

from __future__ import annotations

import bisect
import json
import logging
import re
import statistics
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from plancheck.models import BlockCluster, GlyphBox

# ── Label-registry pattern cache ───────────────────────────────────────

_REGISTRY_PATTERNS: dict[str, list[re.Pattern[str]]] | None = None
_REGISTRY_PATH = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "data"
    / "label_registry.json"
)

# Hardcoded fallback patterns (used when registry file is missing)
_FALLBACK_PATTERNS: dict[str, list[str]] = {
    "notes_column": [r"^\d+\.", r"^•", r"SHALL", r"PER CODE", r"ALL.*SHALL"],
    "header": [r"^[A-Z\s/\-]+$", r"PLAN", r"ELEVATION", r"SECTION", r"DETAIL"],
    "legend": [r"LEGEND", r"SYMBOL", r"KEY"],
    "abbreviations": [r"ABBREVIATION", r"ABBREV", r"[A-Z]{2,6}\s*[=:]"],
    "revision": [r"REV", r"REVISION", r"DATE", r"DESCRIPTION"],
    "title_block": [r"PROJECT", r"SHEET", r"DRAWN BY", r"CHECKED", r"SCALE"],
    "standard_detail": [r"STD", r"STANDARD", r"SEE SHEET"],
}


def _load_label_patterns() -> dict[str, list[re.Pattern[str]]]:
    """Load and compile text_patterns from label_registry.json (cached)."""
    global _REGISTRY_PATTERNS
    if _REGISTRY_PATTERNS is not None:
        return _REGISTRY_PATTERNS

    raw: dict[str, list[str]] = {}
    try:
        data = json.loads(_REGISTRY_PATH.read_text(encoding="utf-8"))
        for entry in data.get("label_registry", []):
            label = entry.get("label", "")
            patterns = entry.get("text_patterns", [])
            if label and patterns:
                raw[label] = patterns
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        log.warning(
            "Could not load label_registry.json (%s), using fallback patterns",
            exc,
            exc_info=True,
        )
        raw = _FALLBACK_PATTERNS
    except Exception as exc:
        log.exception("Unexpected error loading label_registry")
        raw = _FALLBACK_PATTERNS

    if not raw:
        raw = _FALLBACK_PATTERNS

    _REGISTRY_PATTERNS = {}
    for label, patterns in raw.items():
        compiled: list[re.Pattern[str]] = []
        for p in patterns:
            try:
                compiled.append(re.compile(p, re.IGNORECASE))
            except re.error as exc:
                log.warning("Invalid regex pattern %r for label %r: %s", p, label, exc)
        _REGISTRY_PATTERNS[label] = compiled

    return _REGISTRY_PATTERNS


def _keyword_scores(text: str) -> dict[str, int]:
    """Return binary keyword-match indicators for each label type."""
    patterns = _load_label_patterns()
    result: dict[str, int] = {}
    label_to_key = {
        "notes_column": "kw_notes_pattern",
        "header": "kw_header_pattern",
        "legend": "kw_legend_pattern",
        "abbreviations": "kw_abbreviation_pattern",
        "revision": "kw_revision_pattern",
        "title_block": "kw_title_block_pattern",
        "standard_detail": "kw_detail_pattern",
    }
    for label, key in label_to_key.items():
        pats = patterns.get(label, [])
        result[key] = int(any(p.search(text) for p in pats)) if pats else 0
    return result


def featurize(
    block: "BlockCluster",
    page_width: float,
    page_height: float,
    all_blocks: Optional[List["BlockCluster"]] = None,
    zone: str = "unknown",
    tokens: Optional[List["GlyphBox"]] = None,
) -> dict:
    """Convert a *BlockCluster* into a flat feature dict.

    Parameters
    ----------
    block : BlockCluster
        The block to featurize.
    page_width, page_height : float
        Page dimensions in PDF points, used for positional fractions.
    all_blocks : list[BlockCluster], optional
        All blocks on the page — needed for ``neighbor_count``.
    zone : str
        Zone label (e.g. ``"right_margin"``).
    tokens : list[GlyphBox], optional
        Token list to pass through to ``block.get_all_boxes()`` when
        ``block._tokens`` is not set.

    Returns
    -------
    dict
        Flat dict with 21 keys (see module-level docstring for schema).
        Booleans are stored as ``int`` (0/1) for LightGBM compatibility.
    """
    boxes = block.get_all_boxes(tokens=tokens)

    # ── Font metrics ───────────────────────────────────────────────
    font_sizes = [b.font_size for b in boxes if b.font_size > 0] or [0.0]
    font_size_pt = float(statistics.median(font_sizes))
    font_size_max_pt = float(max(font_sizes))
    font_size_min_pt = float(min(font_sizes))

    # ── Text properties ────────────────────────────────────────────
    all_text = " ".join(b.text for b in boxes)
    alpha_chars = [c for c in all_text if c.isalpha()]
    upper_count = sum(1 for c in alpha_chars if c.isupper())
    is_all_caps = (
        int((upper_count / max(len(alpha_chars), 1)) > 0.8) if alpha_chars else 0
    )

    fontnames = [b.fontname for b in boxes if b.fontname]
    bold_count = sum(1 for fn in fontnames if "bold" in fn.lower())
    is_bold = int(bold_count > len(fontnames) / 2) if fontnames else 0

    token_count = len(boxes)
    row_count = len(block.rows)
    text_length = len(all_text)
    avg_chars_per_token = text_length / max(token_count, 1)

    # ── Bbox fractions ─────────────────────────────────────────────
    bbox = block.bbox()
    x0, y0, x1, y1 = bbox
    pw = max(page_width, 1.0)
    ph = max(page_height, 1.0)
    w = x1 - x0
    h = y1 - y0

    x_frac = x0 / pw
    y_frac = y0 / ph
    x_center_frac = ((x0 + x1) / 2) / pw
    y_center_frac = ((y0 + y1) / 2) / ph
    width_frac = w / pw
    height_frac = h / ph
    aspect_ratio = w / max(h, 1.0)

    # ── Token content flags ────────────────────────────────────────
    contains_digit = int(any(c.isdigit() for c in all_text))
    first_text = boxes[0].text if boxes else ""
    starts_with_digit = int(bool(first_text) and first_text[0].isdigit())
    has_colon = int(any(":" in b.text for b in boxes))
    has_period_after_num = int(bool(re.match(r"^\d+\.", first_text)))

    # ── Neighbours (O(n log n) via sorted y-scan) ─────────────────
    neighbor_count = 0
    if all_blocks:
        # Pre-sort other blocks by y0 and use bisect to check only
        # blocks whose vertical extent overlaps [y0 - 20, y1 + 20].
        others = sorted(
            (ob for ob in all_blocks if ob is not block),
            key=lambda b: b.bbox()[1],
        )
        other_y0s = [b.bbox()[1] for b in others]
        # We want blocks where ob.y0 <= y1 + 20 AND ob.y3 >= y0 - 20.
        # bisect_right finds rightmost index where other_y0 <= y1 + 20.
        hi = bisect.bisect_right(other_y0s, y1 + 20)
        for idx in range(hi):
            ob = others[idx].bbox()
            if ob[3] >= y0 - 20:
                neighbor_count += 1

    # ── Text-content features ──────────────────────────────────────
    words = all_text.split()
    n_words = len(words)
    lower_words = [w.lower() for w in words]
    unique_words = set(lower_words)
    unique_word_ratio = round(len(unique_words) / max(n_words, 1), 4)
    uppercase_word_frac = round(
        sum(1 for w in words if w.isupper() and w.isalpha()) / max(n_words, 1), 4
    )
    avg_word_length = round(sum(len(w) for w in words) / max(n_words, 1), 2)

    kw_scores = _keyword_scores(all_text)

    # ── Discriminative features (v3) ───────────────────────────────
    area = max(w * h, 1.0)
    text_density = round(text_length / area, 6)
    x_dist_to_right_margin = round((pw - x1) / pw, 4)
    # Row width variance: stdev of per-row text widths (0 if <=1 row)
    if len(block.rows) > 1:
        row_widths = []
        for row in block.rows:
            if row.boxes:
                rxs = [b.x0 for b in row.boxes] + [b.x1 for b in row.boxes]
                row_widths.append(max(rxs) - min(rxs))
        line_width_variance = (
            round(statistics.pstdev(row_widths), 2) if row_widths else 0.0
        )
    else:
        line_width_variance = 0.0

    # ── OCR confidence features (v4) ───────────────────────────────
    confs = [b.confidence for b in boxes]
    mean_token_confidence = round(statistics.mean(confs), 4) if confs else 1.0
    min_token_confidence = round(min(confs), 4) if confs else 1.0

    return {
        "font_size_pt": round(font_size_pt, 2),
        "font_size_max_pt": round(font_size_max_pt, 2),
        "font_size_min_pt": round(font_size_min_pt, 2),
        "is_all_caps": is_all_caps,
        "is_bold": is_bold,
        "token_count": token_count,
        "row_count": row_count,
        "x_frac": round(x_frac, 4),
        "y_frac": round(y_frac, 4),
        "x_center_frac": round(x_center_frac, 4),
        "y_center_frac": round(y_center_frac, 4),
        "width_frac": round(width_frac, 4),
        "height_frac": round(height_frac, 4),
        "aspect_ratio": round(aspect_ratio, 3),
        "contains_digit": contains_digit,
        "starts_with_digit": starts_with_digit,
        "has_colon": has_colon,
        "has_period_after_num": has_period_after_num,
        "text_length": text_length,
        "avg_chars_per_token": round(avg_chars_per_token, 2),
        "zone": zone,
        "neighbor_count": neighbor_count,
        "unique_word_ratio": unique_word_ratio,
        "uppercase_word_frac": uppercase_word_frac,
        "avg_word_length": avg_word_length,
        "text_density": text_density,
        "x_dist_to_right_margin": x_dist_to_right_margin,
        "line_width_variance": line_width_variance,
        "mean_token_confidence": mean_token_confidence,
        "min_token_confidence": min_token_confidence,
        **kw_scores,
    }


def featurize_region(
    region_type: str,
    bbox: tuple[float, float, float, float],
    header_block: "BlockCluster | None",
    page_width: float,
    page_height: float,
    entry_count: int = 0,
) -> dict:
    """Featurize a region-level element (legend, abbreviation, etc.).

    Produces the same schema as :func:`featurize` with available fields
    populated from *bbox* and *header_block*; the rest are zeroed.

    Parameters
    ----------
    region_type : str
        Element type label (``"legend"``, ``"abbreviations"``, …).
    bbox : tuple
        ``(x0, y0, x1, y1)`` in PDF points.
    header_block : BlockCluster | None
        The header block of the region, if any.
    page_width, page_height : float
        Page dimensions.
    entry_count : int
        Number of entries in the region (legend entries, abbreviation
        entries, etc.).
    """
    x0, y0, x1, y1 = bbox
    pw = max(page_width, 1.0)
    ph = max(page_height, 1.0)
    w = x1 - x0
    h = y1 - y0

    # Extract font info from header block if available
    font_size_pt = 0.0
    font_size_max_pt = 0.0
    font_size_min_pt = 0.0
    is_all_caps = 0
    is_bold = 0
    token_count = entry_count
    row_count = 0
    text_length = 0
    contains_digit = 0
    starts_with_digit = 0
    has_colon = 0
    has_period_after_num = 0

    all_text_str = ""
    if header_block is not None:
        boxes = header_block.get_all_boxes()
        if boxes:
            sizes = [b.font_size for b in boxes if b.font_size > 0] or [0.0]
            font_size_pt = float(statistics.median(sizes))
            font_size_max_pt = float(max(sizes))
            font_size_min_pt = float(min(sizes))
            all_text_str = " ".join(b.text for b in boxes)
            alpha = [c for c in all_text_str if c.isalpha()]
            upper = sum(1 for c in alpha if c.isupper())
            is_all_caps = int((upper / max(len(alpha), 1)) > 0.8) if alpha else 0
            fnames = [b.fontname for b in boxes if b.fontname]
            bc = sum(1 for fn in fnames if "bold" in fn.lower())
            is_bold = int(bc > len(fnames) / 2) if fnames else 0
            token_count = max(token_count, len(boxes))
            row_count = len(header_block.rows)
            text_length = len(all_text_str)
            contains_digit = int(any(c.isdigit() for c in all_text_str))
            first_text = boxes[0].text if boxes else ""
            starts_with_digit = int(bool(first_text) and first_text[0].isdigit())
            has_colon = int(any(":" in b.text for b in boxes))
            has_period_after_num = int(bool(re.match(r"^\d+\.", first_text)))

    words = all_text_str.split()
    n_words = len(words)
    unique_word_ratio = round(len(set(w.lower() for w in words)) / max(n_words, 1), 4)
    uppercase_word_frac = round(
        sum(1 for w in words if w.isupper() and w.isalpha()) / max(n_words, 1), 4
    )
    avg_word_length = round(sum(len(w) for w in words) / max(n_words, 1), 2)
    kw_scores = _keyword_scores(all_text_str)

    # ── Discriminative features (v3) ───────────────────────────────
    area = max(w * h, 1.0)
    text_density = round(text_length / area, 6)
    x_dist_to_right_margin = round((pw - x1) / pw, 4)
    line_width_variance = 0.0  # regions don't have per-row data

    # ── OCR confidence features (v4) ───────────────────────────────
    if header_block is not None:
        hboxes = header_block.get_all_boxes()
        hconfs = [b.confidence for b in hboxes] if hboxes else []
        mean_token_confidence = round(statistics.mean(hconfs), 4) if hconfs else 1.0
        min_token_confidence = round(min(hconfs), 4) if hconfs else 1.0
    else:
        mean_token_confidence = 1.0
        min_token_confidence = 1.0

    return {
        "font_size_pt": round(font_size_pt, 2),
        "font_size_max_pt": round(font_size_max_pt, 2),
        "font_size_min_pt": round(font_size_min_pt, 2),
        "is_all_caps": is_all_caps,
        "is_bold": is_bold,
        "token_count": token_count,
        "row_count": row_count,
        "x_frac": round(x0 / pw, 4),
        "y_frac": round(y0 / ph, 4),
        "x_center_frac": round(((x0 + x1) / 2) / pw, 4),
        "y_center_frac": round(((y0 + y1) / 2) / ph, 4),
        "width_frac": round(w / pw, 4),
        "height_frac": round(h / ph, 4),
        "aspect_ratio": round(w / max(h, 1.0), 3),
        "contains_digit": contains_digit,
        "starts_with_digit": starts_with_digit,
        "has_colon": has_colon,
        "has_period_after_num": has_period_after_num,
        "text_length": text_length,
        "avg_chars_per_token": round(text_length / max(token_count, 1), 2),
        "zone": "unknown",
        "neighbor_count": 0,
        "unique_word_ratio": unique_word_ratio,
        "uppercase_word_frac": uppercase_word_frac,
        "avg_word_length": avg_word_length,
        "text_density": text_density,
        "x_dist_to_right_margin": x_dist_to_right_margin,
        "line_width_variance": line_width_variance,
        "mean_token_confidence": mean_token_confidence,
        "min_token_confidence": min_token_confidence,
        **kw_scores,
    }
