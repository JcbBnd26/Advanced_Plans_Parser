# PlanCheck Combined Remediation Plan

**Sources:** `plancheck_ml_analysis.md` (ML Review) · `CODE_REVIEW_ISSUES.md` (Code Review)
**Overall Grade:** B+ (both reviews agree)
**Date:** March 22, 2026

---

## Guiding Principles

1. **Observability first** — You can't fix what you can't see. Logging and diagnostics
   unblock every other improvement.
2. **Trustworthy metrics before new features** — The calibration data leak means current
   model-selection decisions may be wrong. Fix the ruler before measuring.
3. **Low-effort / high-impact before big refactors** — Most observability fixes are
   one-line changes. Do those before multi-day restructures.
4. **Tests follow fixes** — New test coverage should target the code being changed, not
   be a disconnected coverage drive.

---

## Phase 1 — Diagnostic Foundations (Effort: Low)

*Goal: Make the system tell you what's happening when things go wrong.*

All items here are low-effort, high-impact changes that can be done file-by-file
without architectural risk.

### 1.1 Add logging to silent exception handlers (32 locations)

**Sources:** Code Review Issue 1, ML Review Issue 3 (feature truncation)

Add `log.warning("...", exc_info=True)` to every `except Exception` block that
currently swallows silently. Keep the broad catch (it's correct defensive architecture),
just make it audible.

**Key files and counts:**

| File | Locations |
|------|-----------|
| `ml_feedback.py` | 155, 169, 203, 283, 358, 361 |
| `pipeline_stages.py` | 291 |
| `corrections/hierarchical_classifier.py` | 306, 423 |
| `corrections/text_embeddings.py` | 107, 158, 203 |
| `analysis/layout_model.py` | 172, 321 |
| `ingest/ingest.py` | 337 |
| `corrections/retrain_trigger.py` | 264, 283 |
| `pipeline.py` | 200, 530, 856 |

**Definition of done:** Every `# noqa: BLE001` block has at minimum a
`log.warning` or `log.error` call with `exc_info=True`.

### 1.2 Promote error log severity (20 locations)

**Source:** Code Review Issue 6

Change `log.debug` → `log.warning` (with `exc_info=True`) for runtime failures.
Keep `log.debug` only for truly expected optional-dependency absence (`ImportError`
when a user hasn't installed an extras group).

**Priority targets:**

| File | Lines | What's hidden |
|------|-------|---------------|
| `ml_feedback.py` | 156, 170, 204, 359, 362 | Entire ML feedback subsystem |
| `pipeline_stages.py` | 574 | Layout model prediction |
| `corrections/text_embeddings.py` | 159 | Embedding encode failure |
| `corrections/retrain_trigger.py` | 265, 284 | Auto-rollback, drift stats |
| `export/run_loader.py` | 313 | Config reconstruction |
| `pipeline.py` | 202 | Stage callback |

**Rule of thumb:** If the exception type could be `ImportError` and the module is
optional → `log.debug`. Everything else → `log.warning(..., exc_info=True)`.

### 1.3 Add entry/exit logging to pipeline functions (15 functions)

**Source:** Code Review Issue 2

Each `_run_*_stage` function should log entry (with key input counts) and exit
(with key output counts). `gate()` should log skipped stages with the reason.

**Target functions:**

| File | Function |
|------|----------|
| `pipeline.py` | `gate()`, `_build_page_context()`, `_run_early_stages()`, `_run_vocr_phases()`, `_run_late_stages()`, `_persist_corrections()` |
| `pipeline_stages.py` | `_run_tocr_stage()`, `_run_vocrpp_stage()`, `_run_prune_deskew()`, `_run_vocr_candidates_stage()`, `_run_grouping_stage()` |

**Pattern:**
```python
log.info("_run_tocr_stage: entering with %d pages", len(pages))
# ... existing logic ...
log.info("_run_tocr_stage: produced %d chars across %d pages", total_chars, len(pages))
```

### 1.4 Add feature-truncation warnings

**Source:** ML Review Issue 3

When `classifier.py` or `subtype_classifier.py` truncates a feature vector to match
an older model, log a warning and surface it as a user-visible flag.

```python
if x.shape[1] > self._n_features_in:
    log.warning(
        "Model expects %d features but received %d — truncating. "
        "Retrain to use newer features.",
        self._n_features_in, x.shape[1],
    )
    x = x[:, :self._n_features_in]
```

Expose via a `model_stale` boolean on the classifier so the GUI can show a
"Retrain recommended" indicator.

---

## Phase 2 — ML Metric Integrity (Effort: Medium)

*Goal: Ensure model-selection decisions are based on trustworthy numbers.*

### 2.1 Fix calibration/evaluation data contamination

**Source:** ML Review Issue 1

**Current state:** `training_loop.py` uses validation data for both isotonic
calibration fitting and evaluation metric computation.

**Fix:** Three-way split routing:

| Split | Purpose |
|-------|---------|
| Train (80%) | Fit the model |
| Val (10%) | Fit isotonic calibration via `CalibratedClassifierCV` |
| Test (10%) | Compute final F1, accuracy, per-class metrics |

`training_data.py` already produces the test split — it just needs to be passed
through to evaluation in `training_loop.py` and used in `retrain_trigger.py` for
the rollback comparison.

**Files to change:**
- `corrections/training_loop.py` — accept + use test split for metrics
- `corrections/training_data.py` — verify test split is returned
- `corrections/retrain_trigger.py` — compare using test-set F1

### 2.2 Add cross-validation for metric reporting

**Source:** ML Review Issue 5

Add stratified k-fold (k=5) cross-validation for metric reporting only. The final
shipped model is still trained on all available data.

**Approach:**
1. After the main train, run 5-fold CV on the full dataset
2. Report mean ± std for F1, accuracy, per-class precision/recall
3. Store CV metrics in the experiment tracker alongside the single-split metrics
4. Flag classes where CV variance exceeds a threshold (e.g., std > 0.15)

**File to change:** `corrections/training_loop.py`

### 2.3 Verify ensemble sample-weight propagation

**Source:** ML Review Issue 6

Write an explicit integration test that:
1. Creates a `VotingClassifier` with `HistGradientBoostingClassifier`, `LGBMClassifier`, `XGBClassifier`
2. Passes `sample_weight` to `.fit()`
3. Verifies each estimator's internal sample weights are non-uniform

This is a verification task, not a code change. If any estimator silently ignores
weights, add a `set_fit_request(sample_weight=True)` call per sklearn's metadata
routing API.

**File to create:** `tests/corrections/test_ensemble_weights.py`

### 2.4 Document negative-class confidence scaling interaction

**Source:** ML Review Issue 2

The `(1.0 - p_negative)` scaling in `classifier.py:369` is intentionally
post-calibration. Add an inline comment acknowledging the calibration trade-off
and documenting why the confidence suppression is worth it:

```python
# NOTE: This scaling breaks the isotonic calibration guarantee for elements
# with high p_negative. We accept this because suppressing false-positive
# confidence at threshold boundaries is more important than perfectly
# calibrated probabilities for deletion-like regions.
```

No code change needed — just documentation of the design decision.

---

## Phase 3 — Defensive Programming (Effort: Low-Medium)

*Goal: Catch data corruption at stage boundaries before it propagates.*

### 3.1 Add assertions at pipeline stage boundaries

**Source:** Code Review Issue 5

Add runtime assertions at the seams between pipeline stages. These are cheap
checks that catch bugs early.

**Priority assertion points:**

| Boundary | Assertion |
|----------|-----------|
| Post-ingest | Page dimensions > 0, page count > 0 |
| Post-TOCR | All char bboxes have x0 < x1, y0 < y1 |
| Post-VOCR | Confidence values in [0, 1] |
| Pre-grouping | Block list non-empty (or explicitly empty with reason) |
| Pre-analysis | Feature vectors have consistent dimensionality |
| Pre-export | Output labels are in known label set |

**Files:** `pipeline.py`, `pipeline_stages.py` (at stage entry/exit points)

### 3.2 Narrow exception types where possible

**Source:** Code Review Issue 10

Audit the 42 `# noqa: BLE001` suppressions. For each, determine if the catch can
be narrowed:

| Current catch | Narrowed to |
|---------------|-------------|
| Optional dependency load | `ImportError` |
| Database operations | `sqlite3.Error` |
| Data parsing | `ValueError`, `KeyError` |
| File operations | `OSError` |
| Model prediction | `ValueError`, `RuntimeError` |

Keep `except Exception` (with `# noqa: BLE001`) only for true last-resort handlers
where the exception type genuinely can't be predicted (e.g., third-party model
inference, plugin callbacks).

---

## Phase 4 — Code Clarity & Documentation (Effort: Low-Medium)

*Goal: Make the codebase self-explaining for future maintainers.*

### 4.1 Add "why" comments to complex functions

**Source:** Code Review Issue 4

Target 20-25% comment density for functions over 100 lines. Focus on:
- Threshold values and their empirical basis
- Memory management decisions (`gc.collect()`, freeing `ctx.chars`)
- Ordering dependencies between steps
- Heuristic reasoning in detection functions

**Priority functions (lowest comment rates):**

| Function | File | Current rate | Target |
|----------|------|-------------|--------|
| `_run_document_checks()` | `pipeline.py` | 5% | 20%+ |
| `_apply_ml_feedback()` | `ml_feedback.py` | 11% | 20%+ |
| `run_document()` | `pipeline.py` | 11% | 20%+ |
| `featurize()` | `corrections/features.py` | 12% | 20%+ |

### 4.2 Add missing docstrings (33 functions)

**Source:** Code Review Issue 7

Prioritize by API visibility:

1. **GNN module:** `train_gnn()`, `save_gnn()`, `load_gnn()` — training/loading contract
2. **LLM module:** `materials()`, `dimensions()`, `equipment()`, `summary()`, `get()` — domain entity accessors and cache behavior
3. **Corrections:** `to_dict()` methods on `drift_detection.py` and `retrain_trigger.py`

### 4.3 Document manual GC calls

**Source:** Code Review Issue 9

Add inline comments at both `gc.collect()` sites explaining:
- What memory is being freed and approximate size
- What expensive operation follows that needs the headroom
- Move `import gc` to module level in `pipeline.py`

---

## Phase 5 — Structural Refactoring (Effort: Medium-High)

*Goal: Reduce complexity to make the code more testable and maintainable.*

### 5.1 Break up oversized functions

**Source:** Code Review Issue 3

| Function | Lines | Target |
|----------|-------|--------|
| `_apply_ml_feedback()` | 347 | Split into 4-5 sub-functions by responsibility |
| `detect_standard_detail_regions()` | 274 | Extract shared detection framework |
| `run_document()` | 270 | Extract phase orchestration helpers |
| `featurize()` | 232 | Group feature categories into sub-functions |
| `train_classifier()` | 218 | Separate data prep, training, evaluation |
| `detect_misc_title_regions()` | 203 | Use shared detection framework |
| `detect_abbreviation_regions()` | 198 | Use shared detection framework |
| `detect_legend_regions()` | 197 | Use shared detection framework |
| `build_document_graph()` | 187 | Extract edge-building helpers |
| `_grow_region_from_anchor()` | 163 | Extract boundary validation |

**Shared detection framework:** The `detect_*` functions in `analysis/` share a
common pattern: scan blocks → match criteria → grow region → validate boundaries.
Extract this into a base framework or utility module, then each detector becomes
a thin configuration layer on top.

### 5.2 Integrate GNN into classification pipeline

**Source:** ML Review Issue 4

The GAT model is implemented and trainable but not wired into
`hierarchical_classifier.py`. Integration plan:

1. After Stage 1 classification, build a page-level graph from classified elements
2. Run GNN message passing to refine per-element embeddings
3. Use GNN output as additional features for Stage 2 (or as a confidence modifier)
4. Gate behind `config.gnn.enabled` with graceful fallback (match existing pattern)

**Files:**
- `corrections/hierarchical_classifier.py` — add GNN call after Stage 1
- `analysis/gnn/model.py` — expose inference API
- `analysis/gnn/graph.py` — ensure `build_document_graph()` works with live pipeline data
- `config/subconfigs.py` — add GNN sub-config if not present

---

## Phase 6 — Test Coverage Expansion (Effort: Medium-High)

*Goal: Cover the highest-risk untested code with synthetic-data tests.*

### 6.1 Analysis detector tests (6 modules)

**Source:** Code Review Issue 8

| Module | Priority | Approach |
|--------|----------|----------|
| `standard_details.py` | Highest | Synthetic blocks with detail-like patterns |
| `abbreviations.py` | High | Blocks with abbreviation table patterns |
| `misc_titles.py` | High | Blocks with title-like text properties |
| `legends.py` | High | Blocks with legend-like spatial layout |
| `semantic_regions.py` | Medium | Mock anchor blocks and growth patterns |
| `revisions.py` | Medium | Blocks with revision table patterns |

Each test file should cover: positive detection, negative (no match), edge cases
(partial matches, overlapping regions), and boundary conditions.

### 6.2 Corrections subsystem tests (8 modules)

**Source:** Code Review Issue 8

Focus on the modules not already covered by store mixin tests:
`candidate_classifier.py`, `candidate_features.py`, `candidate_outcomes.py`,
`db_helpers.py`, `db_lock.py`, `snapshots.py`, `store_utils.py`, `box_groups.py`

### 6.3 Export pipeline tests (5 modules)

**Source:** Code Review Issue 8

`artifacts.py`, `csv_export.py`, `font_map.py`, `overlays/colors.py`,
`overlays/structural.py` — test with mock stage results and verify output format.

### 6.4 ML metric integrity tests

**Source:** ML Review Issues 1, 5, 6

- Test that calibration and evaluation use separate splits (2.1)
- Test that CV metrics are recorded in experiment tracker (2.2)
- Test ensemble weight propagation (2.3)

---

## Execution Order & Dependencies

```
Phase 1 (no dependencies — start immediately)
  ├── 1.1 Silent exception logging ──────┐
  ├── 1.2 Log severity promotion ────────┤
  ├── 1.3 Pipeline entry/exit logging ───┤── All can be done in parallel
  └── 1.4 Feature truncation warnings ───┘

Phase 2 (independent of Phase 1)
  ├── 2.1 Fix calibration data leak ─────┐
  ├── 2.2 Add cross-validation ──────────┤── 2.2 depends on 2.1
  ├── 2.3 Verify ensemble weights ───────┤── Independent
  └── 2.4 Document neg-class scaling ────┘── Independent

Phase 3 (benefits from Phase 1 logging being in place)
  ├── 3.1 Add stage-boundary assertions
  └── 3.2 Narrow exception types ────────── Depends on 1.1 being done

Phase 4 (independent — can interleave with anything)
  ├── 4.1 Add "why" comments
  ├── 4.2 Add missing docstrings
  └── 4.3 Document GC calls

Phase 5 (benefits from Phase 1-3 being stable)
  ├── 5.1 Break up oversized functions
  └── 5.2 Integrate GNN ────────────────── Depends on 2.1 for metrics

Phase 6 (parallel with Phase 5, benefits from all prior phases)
  ├── 6.1 Analysis detector tests
  ├── 6.2 Corrections subsystem tests
  ├── 6.3 Export pipeline tests
  └── 6.4 ML metric integrity tests
```

---

## Summary

| Phase | Items | Effort | Impact |
|-------|-------|--------|--------|
| 1 — Diagnostic Foundations | 4 | Low | **High** — unlocks debugging for everything else |
| 2 — ML Metric Integrity | 4 | Medium | **High** — ensures model selection is trustworthy |
| 3 — Defensive Programming | 2 | Low-Med | **Medium** — catches bugs at boundaries |
| 4 — Code Clarity | 3 | Low-Med | **Medium** — reduces onboarding friction |
| 5 — Structural Refactoring | 2 | Med-High | **Medium** — improves testability and maintainability |
| 6 — Test Coverage | 4 | Med-High | **Medium** — prevents regressions |

Total: **19 work items** across 6 phases. Phases 1-2 deliver the highest
value and should be completed first. Phases 3-4 are low-effort polish.
Phases 5-6 are larger investments that pay off as the codebase grows.

---

*Both reviews converge on the same message: the architecture is sound, the ML
design shows full-lifecycle thinking, and the gaps are in observability and
verification — not in fundamental design. Fix the instrumentation, fix the
metrics, then build on the solid foundation.*
