"""Sheet recreation export – text-only PDF reproduction of plan sheets.

Renders every detected ``GlyphBox`` at its original (x, y) position on a
blank page, producing a searchable PDF that mirrors the spatial layout of the
original plan sheet.  This serves as both an integration test for pipeline
coordinate fidelity and a useful artifact for downstream consumers.

Features
--------
* Text width fitting via ``setHorizScale`` — words are compressed or
  stretched to match their original bounding-box width.
* Block boundary rectangles — shows grouping structure with colour-coded
  outlines (header / notes / table / regular).
* Semantic margin labels — ``[HEADER]``, ``[NOTES]``, ``[TABLE]`` tags on
  classified blocks.
* PDF layers (Optional Content Groups) — toggle TOCR / VOCR / reconcile
  tokens, block structure, labels, and watermark independently.
* Original-page watermark — faint background image of the source page for
  visual diff.
* PDF metadata and per-page footer with token counts.

Public API
----------
* ``draw_sheet_recreation``  – single-page render → appends to a
  ``reportlab.pdfgen.canvas.Canvas``
* ``recreate_sheet``         – multi-page convenience; reads serialised
  artifacts, writes one PDF
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from reportlab.lib.utils import ImageReader
from reportlab.pdfgen.canvas import Canvas

from ..models import BlockCluster, GlyphBox
from .font_map import resolve_font
from .page_data import deserialize_page

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------

ColorRGB = Tuple[float, float, float]  # 0-1 range

DEFAULT_COLOR: ColorRGB = (0.0, 0.0, 0.0)  # black

ORIGIN_COLORS: Dict[str, ColorRGB] = {
    "text": (0.0, 0.0, 0.0),  # black  – TOCR
    "ocr_full": (0.0, 0.0, 0.8),  # blue   – VOCR
    "reconcile": (0.8, 0.0, 0.0),  # red    – reconcile injected
    "header_candidate": (0.0, 0.5, 0.0),  # green  – header candidates
}

# Block-type outline colours (stroke)
_BLOCK_COLORS: Dict[str, ColorRGB] = {
    "header": (0.3, 0.3, 0.8),  # blue-gray
    "notes": (0.2, 0.6, 0.2),  # green-gray
    "table": (0.8, 0.5, 0.1),  # orange-gray
    "default": (0.75, 0.75, 0.75),  # light gray
}


def _resolve_color(
    origin: str,
    color_map: Optional[Dict[str, ColorRGB]],
) -> ColorRGB:
    """Pick the fill colour for a glyph based on its origin."""
    if color_map is not None:
        return color_map.get(origin, DEFAULT_COLOR)
    return DEFAULT_COLOR


def _block_color(block: BlockCluster) -> ColorRGB:
    """Pick stroke colour for a block boundary based on semantic type."""
    if getattr(block, "is_header", False):
        return _BLOCK_COLORS["header"]
    if getattr(block, "is_notes", False):
        return _BLOCK_COLORS["notes"]
    if getattr(block, "is_table", False):
        return _BLOCK_COLORS["table"]
    return _BLOCK_COLORS["default"]


def _block_tag(block: BlockCluster) -> Optional[str]:
    """Return a short label for semantically classified blocks."""
    if getattr(block, "is_header", False):
        return "[HEADER]"
    if getattr(block, "is_notes", False):
        return "[NOTES]"
    if getattr(block, "is_table", False):
        return "[TABLE]"
    label = getattr(block, "label", None)
    if label:
        return f"[{label}]"
    return None


# ---------------------------------------------------------------------------
# Font-size estimation
# ---------------------------------------------------------------------------

_MIN_FONT_SIZE = 4.0  # floor to avoid invisible text
_MAX_FONT_SIZE = 72.0  # cap to avoid absurd outliers

# Horizontal-scale sanity bounds (percentage).
_HSCALE_MIN = 50.0  # don't compress below 50%
_HSCALE_MAX = 200.0  # don't stretch above 200%


def _effective_font_size(glyph: GlyphBox) -> float:
    """Return usable font size in PDF points.

    Uses the stored ``font_size`` when available (TOCR tokens); otherwise
    estimates from bounding-box height (VOCR / reconcile tokens).
    """
    size = glyph.font_size if glyph.font_size > 0 else glyph.height()
    return max(_MIN_FONT_SIZE, min(size, _MAX_FONT_SIZE))


# ---------------------------------------------------------------------------
# Single-page renderer
# ---------------------------------------------------------------------------


def draw_sheet_recreation(
    canvas: Canvas,
    page_width: float,
    page_height: float,
    tokens: List[GlyphBox],
    color_map: Optional[Dict[str, ColorRGB]] = None,
    blocks: Optional[List[BlockCluster]] = None,
    draw_blocks: bool = True,
    page_label: Optional[str] = None,
    use_layers: bool = False,
    watermark_img: Optional[Any] = None,
) -> None:
    """Render *tokens* onto a ReportLab canvas page.

    Parameters
    ----------
    canvas : reportlab.pdfgen.canvas.Canvas
        An open canvas; caller is responsible for ``save()``.
    page_width, page_height : float
        Page dimensions in PDF points (1 pt = 1/72 in).
    tokens : list[GlyphBox]
        All tokens to render.
    color_map : dict, optional
        ``{origin: (r, g, b)}`` with 0-1 floats.  If *None*, all text is
        rendered in black.
    blocks : list[BlockCluster], optional
        Block structure for boundary drawing.
    draw_blocks : bool
        Whether to draw block boundary rectangles (default ``True``).
    page_label : str, optional
        Footer text (page number, source, token counts).
    use_layers : bool
        When ``True``, organise content into PDF Optional Content Groups.
    watermark_img : PIL Image, optional
        Faint background image of the original page.
    """
    canvas.setPageSize((page_width, page_height))

    # Check if the canvas supports PDF layers (Optional Content Groups).
    # Not all ReportLab editions expose beginLayer/endLayer.
    _has_layers = use_layers and hasattr(canvas, "beginLayer")
    if use_layers and not _has_layers:
        logger.debug("Canvas does not support layers; ignoring use_layers flag")

    # ── Watermark layer ───────────────────────────────────────────────
    if watermark_img is not None:
        if _has_layers:
            canvas.beginLayer("Original Page", visible=False)
        canvas.saveState()
        canvas.drawImage(
            ImageReader(watermark_img),
            0,
            0,
            width=page_width,
            height=page_height,
        )
        canvas.restoreState()
        if _has_layers:
            canvas.endLayer()

    # ── Token rendering ───────────────────────────────────────────────
    if _has_layers:
        # Group tokens by origin and render each group in its own layer
        origin_groups: Dict[str, List[GlyphBox]] = {}
        for glyph in tokens:
            if not glyph.text:
                continue
            origin_groups.setdefault(glyph.origin, []).append(glyph)

        layer_names = {
            "text": "TOCR Text Layer",
            "ocr_full": "VOCR OCR Tokens",
            "reconcile": "Reconcile Injected",
        }
        for origin, group in origin_groups.items():
            lname = layer_names.get(origin, f"Origin: {origin}")
            canvas.beginLayer(lname, visible=True)
            _render_tokens(canvas, group, page_height, color_map)
            canvas.endLayer()
    else:
        renderable = [g for g in tokens if g.text]
        _render_tokens(canvas, renderable, page_height, color_map)

    # ── Block boundaries ──────────────────────────────────────────────
    if draw_blocks and blocks:
        if _has_layers:
            canvas.beginLayer("Block Structure", visible=True)
        _render_blocks(canvas, blocks, page_height)
        if _has_layers:
            canvas.endLayer()

        # ── Margin labels ─────────────────────────────────────────────
        if _has_layers:
            canvas.beginLayer("Labels", visible=True)
        _render_block_labels(canvas, blocks, page_height)
        if _has_layers:
            canvas.endLayer()

    # ── Page footer ───────────────────────────────────────────────────
    if page_label:
        canvas.saveState()
        canvas.setFont("Helvetica", 6)
        canvas.setFillColorRGB(0.5, 0.5, 0.5)
        label_w = canvas.stringWidth(page_label, "Helvetica", 6)
        canvas.drawString(
            (page_width - label_w) / 2.0,
            4.0,  # 4pt from bottom
            page_label,
        )
        canvas.restoreState()

    canvas.showPage()


# ---------------------------------------------------------------------------
# Internal rendering helpers
# ---------------------------------------------------------------------------


def _render_tokens(
    canvas: Canvas,
    glyphs: List[GlyphBox],
    page_height: float,
    color_map: Optional[Dict[str, ColorRGB]],
) -> None:
    """Render a list of glyphs with per-token state isolation and width fitting.

    Uses ``PDFTextObject.setHorizScale`` for width fitting since the method
    only exists on text objects, not on the canvas directly.
    """
    for glyph in glyphs:
        rl_font = resolve_font(glyph.fontname)
        font_size = _effective_font_size(glyph)

        rl_x = glyph.x0
        rl_y = page_height - glyph.y1

        r, g, b = _resolve_color(glyph.origin, color_map)

        canvas.saveState()

        try:
            canvas.setFont(rl_font, font_size)
        except KeyError:
            rl_font = "Helvetica"
            canvas.setFont(rl_font, font_size)

        # Width fitting: scale text horizontally to match original bbox width
        hscale: float | None = None
        target_w = glyph.x1 - glyph.x0
        if target_w > 0:
            predicted_w = canvas.stringWidth(glyph.text, rl_font, font_size)
            if predicted_w > 0:
                ratio = (target_w / predicted_w) * 100.0
                if _HSCALE_MIN <= ratio <= _HSCALE_MAX:
                    hscale = ratio

        # Use a text object so we can apply setHorizScale
        tx = canvas.beginText(rl_x, rl_y)
        tx.setFont(rl_font, font_size)
        tx.setFillColor((r, g, b))
        if hscale is not None:
            tx.setHorizScale(hscale)
        tx.textOut(glyph.text)
        canvas.drawText(tx)

        canvas.restoreState()


def _render_blocks(
    canvas: Canvas,
    blocks: List[BlockCluster],
    page_height: float,
) -> None:
    """Draw light rectangles around each block cluster."""
    canvas.saveState()
    canvas.setLineWidth(0.5)
    for block in blocks:
        bx0, by0, bx1, by1 = block.bbox()
        if bx0 == bx1 or by0 == by1:
            continue  # degenerate bbox
        rl_y = page_height - by1
        w = bx1 - bx0
        h = by1 - by0

        cr, cg, cb = _block_color(block)
        canvas.setStrokeColorRGB(cr, cg, cb)
        canvas.rect(bx0, rl_y, w, h, stroke=1, fill=0)
    canvas.restoreState()


def _render_block_labels(
    canvas: Canvas,
    blocks: List[BlockCluster],
    page_height: float,
) -> None:
    """Draw small semantic tags at the top-left of classified blocks."""
    canvas.saveState()
    canvas.setFont("Helvetica", 5)
    for block in blocks:
        tag = _block_tag(block)
        if tag is None:
            continue
        bx0, by0, _bx1, _by1 = block.bbox()
        # Position tag just above the block's top-left corner
        rl_y = page_height - by0 + 1.5  # 1.5pt above top edge
        cr, cg, cb = _block_color(block)
        canvas.setFillColorRGB(cr, cg, cb)
        canvas.drawString(bx0, rl_y, tag)
    canvas.restoreState()


# ---------------------------------------------------------------------------
# Token count helpers
# ---------------------------------------------------------------------------


def _token_counts(tokens: List[GlyphBox]) -> Dict[str, int]:
    """Count tokens by origin."""
    counts: Dict[str, int] = {}
    for t in tokens:
        counts[t.origin] = counts.get(t.origin, 0) + 1
    return counts


def _make_page_label(
    page_num: int,
    tokens: List[GlyphBox],
    pdf_stem: str,
    run_name: str,
) -> str:
    """Build page footer string with token breakdown."""
    total = len(tokens)
    counts = _token_counts(tokens)
    parts = [f"Page {page_num}", pdf_stem, run_name, f"{total} tokens"]
    detail_parts = []
    if counts.get("text", 0):
        detail_parts.append(f"{counts['text']} TOCR")
    if counts.get("ocr_full", 0):
        detail_parts.append(f"{counts['ocr_full']} VOCR")
    if counts.get("reconcile", 0):
        detail_parts.append(f"{counts['reconcile']} reconcile")
    if detail_parts:
        parts[-1] += f" ({' / '.join(detail_parts)})"
    return " | ".join(parts)


# ---------------------------------------------------------------------------
# Multi-page convenience wrapper
# ---------------------------------------------------------------------------


def recreate_sheet(
    run_dir: str | Path,
    out_path: str | Path | None = None,
    pages: Optional[List[int]] = None,
    color_map: Optional[Dict[str, ColorRGB]] = None,
    pdf_stem: Optional[str] = None,
    draw_blocks: bool = True,
    use_layers: bool = False,
    source_pdf: Optional[str | Path] = None,
    watermark_opacity: float = 0.15,
) -> Path:
    """Build a recreation PDF from serialised pipeline artifacts.

    Parameters
    ----------
    run_dir : path
        Root of an existing pipeline run (contains ``artifacts/``).
    out_path : path, optional
        Explicit output path.  Defaults to
        ``{run_dir}/exports/{stem}_recreation.pdf``.
    pages : list[int], optional
        1-based page numbers to include.  *None* = all found artifacts.
    color_map : dict, optional
        Passed through to ``draw_sheet_recreation``.
    pdf_stem : str, optional
        PDF stem override for the output filename.
    draw_blocks : bool
        Draw block boundary rectangles (default ``True``).
    use_layers : bool
        Organise content into togglable PDF layers (default ``False``).
    source_pdf : path, optional
        Path to the original PDF for watermark rendering.
    watermark_opacity : float
        Opacity for watermark background (0.0–1.0, default 0.15).

    Returns
    -------
    Path
        The written PDF path.
    """
    run_dir = Path(run_dir)
    artifacts_dir = run_dir / "artifacts"
    if not artifacts_dir.is_dir():
        raise FileNotFoundError(f"Artifacts directory not found: {artifacts_dir}")

    # Discover artifact files
    artifact_files = sorted(artifacts_dir.glob("*_extraction.json"))
    if not artifact_files:
        raise FileNotFoundError(f"No extraction artifacts in {artifacts_dir}")

    # Determine PDF stem from first artifact filename if not given
    if pdf_stem is None:
        name = artifact_files[0].stem
        m = re.match(r"^(.+?)_page_\d+_extraction$", name)
        pdf_stem = m.group(1) if m else name

    # Filter to requested pages
    page_artifacts: list[tuple[int, Path]] = []
    for fp in artifact_files:
        m = re.search(r"_page_(\d+)_extraction\.json$", fp.name)
        if not m:
            continue
        page_num = int(m.group(1))
        if pages is None or page_num in pages:
            page_artifacts.append((page_num, fp))

    if not page_artifacts:
        raise FileNotFoundError(
            f"No matching page artifacts for pages={pages} in {artifacts_dir}"
        )

    page_artifacts.sort(key=lambda t: t[0])

    # Output path
    if out_path is None:
        exports_dir = run_dir / "exports"
        exports_dir.mkdir(parents=True, exist_ok=True)
        out_path = exports_dir / f"{pdf_stem}_recreation.pdf"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Optionally open source PDF for watermark rendering
    pdf_handle = None
    if source_pdf is not None:
        try:
            import pdfplumber

            pdf_handle = pdfplumber.open(source_pdf)
        except (ImportError, OSError, ValueError) as exc:
            logger.warning("Could not open source PDF for watermark: %s", exc)
            pdf_handle = None

    try:
        # Build PDF
        c = Canvas(str(out_path))

        # PDF metadata
        c.setTitle(f"{pdf_stem} — Sheet Recreation")
        c.setSubject(f"Text-only spatial recreation from pipeline run {run_dir.name}")
        c.setAuthor("Advanced Plan Parser")
        c.setCreator(f"sheet_recreation.py — {datetime.now().isoformat()}")

        run_name = run_dir.name

        for page_num, artifact_path in page_artifacts:
            data = json.loads(artifact_path.read_text(encoding="utf-8"))
            tokens, blocks_list, _cols, page_w, page_h = deserialize_page(data)

            # Token count logging
            counts = _token_counts(tokens)
            logger.info(
                "Page %d: %d tokens (%d TOCR, %d VOCR, %d reconcile)",
                page_num,
                len(tokens),
                counts.get("text", 0),
                counts.get("ocr_full", 0),
                counts.get("reconcile", 0),
            )

            # Watermark image
            wm_img = None
            if pdf_handle is not None:
                try:
                    pdfp_page = pdf_handle.pages[page_num]
                    pil_img = pdfp_page.to_image(resolution=150).original.copy()
                    # Apply opacity via RGBA conversion
                    pil_img = pil_img.convert("RGBA")
                    alpha_val = int(watermark_opacity * 255)
                    # Set uniform alpha channel
                    alpha = pil_img.split()[3].point(lambda _: alpha_val)
                    pil_img.putalpha(alpha)
                    wm_img = pil_img
                except (IndexError, ValueError, OSError, RuntimeError) as exc:
                    logger.warning(
                        "Could not render watermark for page %d: %s",
                        page_num,
                        exc,
                    )

            page_label = _make_page_label(page_num, tokens, pdf_stem, run_name)

            draw_sheet_recreation(
                c,
                page_w,
                page_h,
                tokens,
                color_map=color_map,
                blocks=blocks_list if draw_blocks else None,
                draw_blocks=draw_blocks,
                page_label=page_label,
                use_layers=use_layers,
                watermark_img=wm_img,
            )

        c.save()
        logger.info("Sheet recreation saved -> %s", out_path)
    finally:
        if pdf_handle is not None:
            pdf_handle.close()

    return out_path
