"""Debug overlays for OCR reconciliation results.

Public API
----------
draw_reconcile_debug   – render a multi-layer debug overlay
draw_symbol_overlay    – render a simple overlay with green boxes around symbols
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

from PIL import Image, ImageDraw, ImageFont

from ..models import GlyphBox
from ..reconcile.reconcile import (
    ReconcileResult,
    _center,
    _has_allowed_symbol,
    _has_numeric_symbol_context,
)

log = logging.getLogger("plancheck.ocr_reconcile")


def draw_reconcile_debug(
    result: ReconcileResult,
    page_width: float,
    page_height: float,
    out_path: Path | str,
    scale: float = 1.0,
    background: Optional[Image.Image] = None,
) -> None:
    """Render a debug overlay showing OCR reconciliation results.

    Colour key
    ----------
    * **Light grey** – all OCR tokens detected on the page (raw OCR output).
    * **Orange outline** – OCR tokens containing an allowed symbol that were
      rejected (filtered out).
    * **Green box + label** – accepted / injected tokens.
    * **Blue line** – match line from OCR token to its matched PDF token.
    * **Cyan outline** – digit anchors used in Case C composite matching.
    * **Red outline + label** – rejected Case C candidates with reason.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    canvas_w = int(page_width * scale)
    canvas_h = int(page_height * scale)

    if background is not None:
        base = background.copy().convert("RGBA")
        if base.size != (canvas_w, canvas_h):
            base = base.resize((canvas_w, canvas_h), Image.LANCZOS)
    else:
        base = Image.new("RGBA", (canvas_w, canvas_h), (255, 255, 255, 255))

    overlay = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    try:
        font = ImageFont.truetype("arial.ttf", max(10, int(8 * scale)))
    except OSError:
        font = ImageFont.load_default()

    allowed = "%" + "/°±"  # default; could pull from stats if wanted

    def _rect(b: GlyphBox) -> Tuple[float, float, float, float]:
        """Scale a glyph box's bbox for overlay rendering."""
        return (b.x0 * scale, b.y0 * scale, b.x1 * scale, b.y1 * scale)

    # Layer 1: all OCR tokens in light grey
    for t in result.all_ocr_tokens:
        draw.rectangle(_rect(t), outline=(180, 180, 180, 80), width=1)

    # Build sets for quick lookup
    added_set = set(id(t) for t in result.added_tokens)

    # Layer 2: symbol-bearing OCR tokens that were NOT accepted → orange
    # (only tokens that pass numeric-context filter, not headings)
    for m in result.matches:
        if (
            _has_numeric_symbol_context(m.ocr_box.text, allowed)
            and id(m.ocr_box) not in added_set
        ):
            # Check it wasn't accepted (added tokens have different identity)
            draw.rectangle(_rect(m.ocr_box), outline=(255, 140, 0, 180), width=2)

    # Layer 3: match lines (blue) for matched pairs
    for m in result.matches:
        if m.pdf_box is not None and m.match_type in ("iou", "center"):
            ocr_cx, ocr_cy = _center(m.ocr_box)
            pdf_cx, pdf_cy = _center(m.pdf_box)
            if _has_allowed_symbol(m.ocr_box.text, allowed):
                draw.line(
                    [
                        (ocr_cx * scale, ocr_cy * scale),
                        (pdf_cx * scale, pdf_cy * scale),
                    ],
                    fill=(0, 120, 255, 120),
                    width=1,
                )

    # Layer 4: accepted/injected tokens → green with label
    for t in result.added_tokens:
        r = _rect(t)
        draw.rectangle(r, outline=(0, 200, 0, 220), width=2)
        label = t.text
        draw.text((r[0], r[1] - 12 * scale), label, fill=(0, 200, 0, 255), font=font)

    # Layer 5: Case-C anchors → cyan outlines (from injection_log)
    injection_log = result.stats.get("injection_log", []) if result.stats else []
    for entry in injection_log:
        for anc in entry.get("anchors", []):
            bb = anc.get("bbox")
            if bb:
                r = (bb[0] * scale, bb[1] * scale, bb[2] * scale, bb[3] * scale)
                draw.rectangle(r, outline=(0, 220, 220, 160), width=1)

    # Layer 6: rejected candidates → red dashed outline + label
    for entry in injection_log:
        for cand in entry.get("candidates", []):
            if cand.get("status") == "rejected":
                bb = cand.get("bbox")
                if bb:
                    r = (bb[0] * scale, bb[1] * scale, bb[2] * scale, bb[3] * scale)
                    draw.rectangle(r, outline=(220, 40, 40, 160), width=1)
                    reason = cand.get("reason", "")[:20]
                    draw.text(
                        (r[0], r[3] + 1),
                        f"✗ {cand.get('symbol','')} {reason}",
                        fill=(220, 40, 40, 200),
                        font=font,
                    )

    # Composite and save
    out = Image.alpha_composite(base, overlay)
    out.convert("RGB").save(str(out_path))


def draw_symbol_overlay(
    result: ReconcileResult,
    page_width: float,
    page_height: float,
    out_path: Path | str,
    scale: float = 1.0,
    background: Optional[Image.Image] = None,
    allowed_symbols: str = "%/°±",
    show_labels: bool = True,
) -> None:
    """Render a simple overlay showing symbol-bearing OCR tokens in green boxes.

    Parameters
    ----------
    result : ReconcileResult
        Output from the reconciliation pipeline.
    page_width, page_height : float
        PDF page dimensions in points.
    out_path : Path | str
        Where to save the PNG.
    scale : float
        Render scale factor (1.0 = 72 DPI).
    background : Image, optional
        Background image (e.g. page render). If None, uses white.
    allowed_symbols : str
        Characters considered symbols. Default: "%/°±".
    show_labels : bool
        If True, draw the token text above each box.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    canvas_w = int(page_width * scale)
    canvas_h = int(page_height * scale)

    if background is not None:
        base = background.copy().convert("RGBA")
        if base.size != (canvas_w, canvas_h):
            base = base.resize((canvas_w, canvas_h), Image.LANCZOS)
    else:
        base = Image.new("RGBA", (canvas_w, canvas_h), (255, 255, 255, 255))

    overlay = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    try:
        font = ImageFont.truetype("arial.ttf", max(12, int(10 * scale)))
    except OSError:
        font = ImageFont.load_default()

    def _rect(b: GlyphBox) -> Tuple[float, float, float, float]:
        """Scale a glyph box's bbox for overlay rendering."""
        return (b.x0 * scale, b.y0 * scale, b.x1 * scale, b.y1 * scale)

    # Find all symbol-bearing tokens
    symbol_tokens = [
        t for t in result.all_ocr_tokens if _has_allowed_symbol(t.text, allowed_symbols)
    ]

    # Draw green boxes around symbol tokens
    for t in symbol_tokens:
        r = _rect(t)
        # Green box outline (RGB: 0, 200, 0)
        draw.rectangle(r, outline=(0, 200, 0, 255), width=2)
        if show_labels:
            label = t.text
            draw.text(
                (r[0], r[1] - 14 * scale),
                label,
                fill=(0, 200, 0, 255),
                font=font,
            )

    # Composite and save
    out = Image.alpha_composite(base, overlay)
    out.convert("RGB").save(str(out_path))
    log.info("Symbol overlay saved: %s (%d symbols)", out_path.name, len(symbol_tokens))
