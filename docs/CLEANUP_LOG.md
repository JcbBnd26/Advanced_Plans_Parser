# Workspace Cleanup Log

## Overlay PNG Removal (2026-03-01)

Removed static overlay PNG generation from both pipeline runners. The GUI's
interactive ML Trainer canvas and Overlay Viewer render detections, lines,
and columns in real time, making the file-based PNGs redundant (~3 MB and
2–3 s saved per page).

### Deleted Files

| File | Reason |
|------|--------|
| `scripts/overlays/overlay_headers_red.py` | Standalone CLI overlay — superseded by GUI viewer |
| `scripts/overlays/overlay_notes_green.py` | Standalone CLI overlay — superseded by GUI viewer |
| `scripts/overlays/overlay_notes_purple.py` | Standalone CLI overlay — superseded by GUI viewer |

### Modified Files

| File | Changes |
|------|---------|
| `scripts/runners/run_pdf_batch.py` | Removed `draw_lines_overlay`, `draw_columns_overlay`, `draw_reconcile_debug`, `draw_symbol_overlay` calls; dropped overlay import; removed `overlay_png` from manifest artifacts |
| `scripts/runners/run_pdf_page.py` | Removed `draw_overlay`, `draw_lines_overlay` imports and overlay generation; removed `overlay_png` from manifest artifacts |

### Kept

| File | Reason |
|------|--------|
| `src/plancheck/export/overlay.py` | Drawing library — kept for programmatic export use |

---

## MLOps Tab Removal (2026-03-01)

Removed the dedicated MLOps tab. Drift monitoring and feature caching run
automatically in the pipeline; retraining is accessible from the ML Trainer
tab and CLI; experiment history lives in the Database tab.

### Deleted Files

| File | Reason |
|------|--------|
| `scripts/gui/tab_mlops.py` | Tab removed — features covered elsewhere |
| `tests/gui/test_tab_mlops.py` | Test for removed tab |

### Modified Files

| File | Changes |
|------|---------|
| `scripts/gui/gui.py` | Removed MLOps import/instantiation, dropped Ctrl+8 binding, updated docstring |
| `scripts/gui/tab_annotation.py` | Renamed tab label from "Annotation" to "ML Trainer" |

---

## Phase 4 — Production ML Infrastructure (2026-02-24)

Implemented the final phase of the ML Upgrade Roadmap.
All **1,499 tests pass** (3 skipped) after this phase — up from ~1,370 before Phase 4.

### New Modules (4)

| File | Purpose |
|------|---------|
| `src/plancheck/corrections/drift_detection.py` | Percentile-bounds data drift detection (fit / check / persist) |
| `src/plancheck/corrections/retrain_trigger.py` | Automated retrain logic + startup check |
| `src/plancheck/corrections/experiment_tracker.py` | Lightweight experiment listing, comparison, CSV export |
| `scripts/gui/tab_mlops.py` | GUI MLOps tab — Drift Monitor, Retrain Control, Feature Cache, Experiment History *(removed 2026-03-01)* |

### Modified Modules (6)

| File | Changes |
|------|---------|
| `src/plancheck/config.py` | 6 new fields: `ml_drift_enabled/threshold/stats_path`, `ml_retrain_threshold/on_startup`, `ml_feature_cache_enabled` |
| `src/plancheck/pipeline.py` | `drift_warnings` on `PageResult`, feature cache lookups in `_apply_ml_feedback`, `predict_from_vector` path |
| `src/plancheck/corrections/store.py` | `feature_cache` table, 4 `training_runs` migrations (hyperparams, feature_set, training_curves, feature_version), retrain helpers, cache CRUD |
| `src/plancheck/corrections/classifier.py` | `FEATURE_VERSION = 5`, `predict_from_vector()`, hyperparams/feature_set in `train()` return |
| `scripts/train_model.py` | Fixed `restore_snapshot` bug (tag → path), drift detector fitting, extended `save_training_run` |
| `scripts/gui/gui.py` | Registered MLOps tab *(later removed 2026-03-01)* |

### New Test Files (6 files, ~129 tests)

- `tests/corrections/test_drift_detection.py`
- `tests/corrections/test_retrain_trigger.py`
- `tests/corrections/test_feature_cache.py`
- `tests/corrections/test_experiment_tracker.py`
- `tests/corrections/test_phase4_config.py`
- `tests/gui/test_tab_mlops.py`

### Bugs Fixed

- `store.restore_snapshot()` in `train_model.py` was called with a tag string instead of a `Path`
- `count_corrections_since()` referenced wrong column (`created_at` → `corrected_at`)
- `get_training_history()` ORDER BY was non-deterministic for same-millisecond rows (added `rowid DESC` tiebreaker)

---

## Initial Cleanup — February 18, 2026

### Summary

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
