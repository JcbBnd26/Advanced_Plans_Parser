# Workspace Cleanup Log — February 18, 2026

## Summary

Removed **36 files/directories** of vestigial, generated, and dead code.  
Updated **13 import statements** across **9 script files** to use canonical subpackage paths.  
All **528 tests pass** after cleanup.

---

## Deleted Files & Directories

### 1. `__pycache__` directories (18 dirs, ~150 .pyc files)

Auto-generated Python bytecode cache. Regenerated on next run.

- `scripts/__pycache__/`
- `src/plancheck/__pycache__/`
- `src/plancheck/analysis/__pycache__/`
- `src/plancheck/checks/__pycache__/`
- `src/plancheck/export/__pycache__/`
- `src/plancheck/grouping/__pycache__/`
- `src/plancheck/reconcile/__pycache__/`
- `src/plancheck/tocr/__pycache__/`
- `src/plancheck/vocr/__pycache__/`
- `src/plancheck/vocrpp/__pycache__/`
- `tests/__pycache__/`
- `tests/analysis/__pycache__/`
- `tests/checks/__pycache__/`
- `tests/export/__pycache__/`
- `tests/grouping/__pycache__/`
- `tests/reconcile/__pycache__/`
- `tests/tocr/__pycache__/`
- `tests/vocrpp/__pycache__/`

### 2. Debug artifacts

- `debug_headers.txt` — 6,766-line debug log dump left in repo root

### 3. One-time instruction files

- `RESTRUCTURE_AGENT_INSTRUCTIONS.md` — 1,011-line restructuring guide, already completed

### 4. Dead re-export wrappers (not imported by anything)

These were backward-compatibility wrappers left from the flat→subpackage restructure.  
No code imported from them.

- `src/plancheck/_abbreviation_detect.py`
- `src/plancheck/_graphics.py`
- `src/plancheck/_misc_title_detect.py`
- `src/plancheck/_ocr_engine.py`
- `src/plancheck/_region_helpers.py`
- `src/plancheck/_revision_detect.py`
- `src/plancheck/_standard_detail_detect.py`
- `src/plancheck/ocr_reconcile.py`
- `src/plancheck/preprocess.py`
- `src/plancheck/zoning.py`

### 5. Still-imported re-export wrappers (imports updated, then deleted)

These wrappers were still imported by scripts. Updated all 13 import statements  
in 9 script files to use canonical subpackage paths, then deleted the wrappers.

- `src/plancheck/_structural_boxes.py` → `plancheck.analysis.structural_boxes`
- `src/plancheck/font_metrics.py` → `plancheck.grouping.font_metrics`
- `src/plancheck/legends.py` → `plancheck.analysis.legends`
- `src/plancheck/overlay.py` → `plancheck.export.overlay`
- `src/plancheck/page_data.py` → `plancheck.export.page_data`
- `src/plancheck/semantic_checks.py` → `plancheck.checks.semantic_checks`
- `src/plancheck/ocr_preprocess_pipeline.py` → `plancheck.vocrpp.preprocess`

### 6. Ad-hoc debug scripts

Hardcoded to specific PDFs/pages, not part of test suite.

- `scripts/debug/_debug_char_gaps.py`
- `scripts/debug/_debug_slash.py`
- `scripts/debug/_debug_nc3.py`
- `scripts/debug/_debug_notes.py`
- `scripts/debug/` (directory removed)

### 7. Empty directories

- `files/` — empty, unused
- `runs/` — empty, gitignored

---

## Import Updates

| File | Changes |
|------|---------|
| `scripts/runners/run_pdf_batch.py` | 5 imports updated |
| `scripts/runners/run_pdf_page.py` | 1 import updated |
| `scripts/diagnostics/run_font_metrics_diagnostics.py` | 1 import updated |
| `scripts/diagnostics/run_ocr_preprocess.py` | 1 import updated |
| `scripts/overlays/overlay_headers_red.py` | 2 imports updated |
| `scripts/overlays/overlay_notes_green.py` | 2 imports updated |
| `scripts/overlays/overlay_notes_purple.py` | 2 imports updated |
| `scripts/gui/overlay_viewer.py` | 2 imports updated |
| `scripts/utils/extract_page.py` | 1 import updated |
| `src/plancheck/grouping/font_metrics.py` | docstring updated |
| `src/plancheck/checks/semantic_checks.py` | docstring updated |

---

## Kept (reviewed but not removed)

- `PipelineDiagram.html` — helpful pipeline visualization
- `docs/OCR_RECONCILE_STATUS.md` — historical reference for OCR decisions
- `docs/OCR_Phase0-3_Implementation.md` — implementation documentation
- `docs/TAGS.md` — tag reference for the data model
- `samples/demo_boxes.json` — toy input referenced in README
- `launch_gui.bat` — utility launcher
