# Advanced Plan Parser

Geometry-first PDF plan-sheet analysis pipeline for automated plan checking.
Extracts text, graphics, and structural regions from CAD-origin PDFs, then
runs semantic checks against engineering standards.

## Pipeline

Nine stages, executed per page:

```
ingest → tocr ‖ vocrpp → vocr → reconcile → grouping → analysis → checks → export
```

| Stage | Description |
|-------|-------------|
| **ingest** | Validate PDF, extract metadata, render page image |
| **tocr** | Text-layer OCR via pdfplumber (fonts, glyphs, diagnostics) |
| **vocrpp** | Pre-process page image for visual OCR (grayscale, CLAHE, etc.) |
| **vocr** | Visual OCR via Surya (symbols missing from text layer) |
| **reconcile** | Merge text-layer and visual-OCR tokens (symbol injection) |
| **grouping** | Row/line/span clustering → `BlockCluster` / `NotesColumn` |
| **analysis** | Graphics extraction, structural boxes, legends, abbreviations, revisions, zoning |
| **checks** | Semantic rule checks (missing notes, numbering gaps, etc.) |
| **export** | Overlays (PNG), structured CSVs, extraction JSON |

## Layout

```
src/plancheck/          Core package
  config.py             GroupingConfig (~130 tunables)
  models.py             GlyphBox, BlockCluster, NotesColumn, region models
  pipeline.py           Stage gating, timing, StageResult
  ingest/               PDF validation & page-image rendering
  tocr/                 Text-layer extraction & preprocessing
  vocrpp/               OCR image preprocessing (CLAHE, binarize, sharpen)
  vocr/                 Surya OCR backend & token extraction
  reconcile/            Dual-source OCR reconciliation
  grouping/             Clustering, font metrics, notes/header detection
  analysis/             Legends, abbreviations, revisions, zoning, graphics
  checks/               Semantic rule checks
  export/               Overlay rendering, CSV export, page serialisation
scripts/
  runners/              CLI entry points (run_pdf_batch, run_pdf_page, new_run)
  diagnostics/          Benchmarking & tuning harnesses
  gui/                  Tkinter GUI + overlay viewer
  overlays/             Standalone overlay scripts
  utils/                Page extraction, tag listing
tests/                  pytest suite (~1280 tests)
docs/                   Implementation notes & status logs
```

## Quick start

```bash
pip install -r requirements.txt
# Install the `plancheck` package (src/ layout) so scripts can import it:
pip install -e .
# Process a full PDF (all pages):
python -m scripts.runners.run_pdf_batch input/your.pdf
# Process a single page:
python -m scripts.runners.run_pdf_page input/your.pdf --page 0
# Launch the GUI:
launch_gui.bat
```

## Dependencies

See [requirements.txt](requirements.txt). Key runtime dependencies:
pdfplumber, Pillow, numpy, reportlab, OpenCV (cv2), Surya OCR 0.17.x.

## Testing

```bash
pip install pytest
python -m pytest tests/ -q
```