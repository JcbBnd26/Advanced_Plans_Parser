# Advanced Plan Parser (Fresh Start)

Minimal geometry-first grouping skeleton for plan checking. The old backbone codebase lives outside this folder; this is a clean slate with new naming.

## Layout
- src/plancheck: core grouping package (config, models, preprocess, grouping, zoning)
- scripts/run_grouping.py: CLI to run grouping on JSON boxes
- samples/demo_boxes.json: toy input to sanity-check grouping
- requirements.txt: runtime deps (numpy only for now)

## Quick start
1. Install deps: `pip install -r requirements.txt`
2. Option A: Create an empty run folder: `python scripts/new_run.py` (creates runs/run_YYYYmmdd_HHMMSS with subfolders)
3. Option B: Process a PDF page directly (extract boxes, group, overlay, and bundle):
	`python scripts/run_pdf_page.py input/your.pdf --page 0`
	This copies the PDF into a new run, saves page boxes JSON, and writes an overlay PNG under runs/.../overlays/.
4. To run grouping on an existing boxes JSON: `python scripts/run_grouping.py <boxes.json> --overlay <out.png>`

The CLI prints grouped blocks and marks table-like clusters. If you pass `--overlay`, it also saves a PNG overlay so you can see grouping. Coordinates are expected in page space; skew handling is stubbed and limited to small rotations.
