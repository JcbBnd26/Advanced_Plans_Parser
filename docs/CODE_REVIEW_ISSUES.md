# Advanced Plans Parser — Code Review Issues

**Grade: B+**
**Review Date: March 22, 2026**
**Scope: Diagnostic functions, observability, code clarity, and structural hygiene across `src/plancheck/`**

---

## Issue 1: Silent Exception Swallowing (32 locations)

**Severity: High**
**Category: Observability / Cross-Cutting Concern**

32 `except Exception` blocks catch errors without logging them at all — no `log.error`, no `log.warning`, no `exc_info=True`. The exception is caught, suppressed with a `# noqa: BLE001` comment, and execution continues with zero evidence that anything went wrong.

This is the single most impactful issue in the codebase. When a user reports a bad parse result in production, these are the breadcrumbs that won't exist.

**Worst offenders:**

| File | Line | Context |
|------|------|---------|
| `ml_feedback.py` | 155, 169, 203, 283, 358, 361 | 6 silent catches across the entire ML feedback flow |
| `pipeline_stages.py` | 291 | VOCR failure silenced — this is a core pipeline stage |
| `corrections/hierarchical_classifier.py` | 306, 423 | Stage 2 classifier and LLM tiebreaker failures invisible |
| `corrections/text_embeddings.py` | 107, 158, 203 | Embedding model load, encode, and batch encode all silent |
| `analysis/layout_model.py` | 172, 321 | Layout model load and prediction failures invisible |
| `ingest/ingest.py` | 337 | Metadata extraction failure silent |
| `corrections/retrain_trigger.py` | 264, 283 | Auto-rollback and drift stats failures invisible |
| `pipeline.py` | 200, 530, 856 | Stage callback, header text fallback, GNN embedder all silent |

**Recommendation:** Every one of these should have at minimum `log.warning("...", exc_info=True)`. The `# noqa: BLE001` comment is correct — broad catches are the right pattern here to prevent cascade failures. But "don't crash" and "don't tell anyone" are two different decisions. The first is good defensive architecture. The second is a diagnostic blackout.

---

## Issue 2: Pipeline Stage Functions with Zero Logging (15 functions)

**Severity: High**
**Category: Observability**

15 functions in the core pipeline path (`pipeline.py` and `pipeline_stages.py`) contain zero `log.*` calls despite being 17–112 lines long. These are the functions that orchestrate the entire 9-stage flow.

| File | Function | Lines |
|------|----------|-------|
| `pipeline.py` | `gate()` | 71 |
| `pipeline.py` | `_build_page_context()` | 24 |
| `pipeline.py` | `_run_early_stages()` | 26 |
| `pipeline.py` | `_run_vocr_phases()` | 38 |
| `pipeline.py` | `_run_late_stages()` | 54 |
| `pipeline.py` | `_persist_corrections()` | 112 |
| `pipeline_stages.py` | `_run_tocr_stage()` | 24 |
| `pipeline_stages.py` | `_run_vocrpp_stage()` | 37 |
| `pipeline_stages.py` | `_run_prune_deskew()` | 17 |
| `pipeline_stages.py` | `_run_vocr_candidates_stage()` | 67 |
| `pipeline_stages.py` | `_run_grouping_stage()` | 30 |

The `run_stage()` context manager provides timing, but the individual stage functions don't announce entry, exit, or key decisions. When a page produces bad output, there's no log trail showing which stages ran, what they received, or what they produced — only whether they succeeded or failed.

**Recommendation:** Each `_run_*_stage` function should log at least entry (with key input counts) and exit (with key output counts). The `gate()` function should log skipped stages with the reason, since that's critical diagnostic context when debugging "why didn't VOCR run on this page?"

---

## Issue 3: Oversized Functions (Top 10)

**Severity: Medium**
**Category: Maintainability**

These functions exceed reasonable size and complexity thresholds, making them harder to test, debug, and modify:

| File | Function | Lines | Branches |
|------|----------|-------|----------|
| `ml_feedback.py` | `_apply_ml_feedback()` | 347 | 59 |
| `analysis/standard_details.py` | `detect_standard_detail_regions()` | 274 | 31 |
| `pipeline.py` | `run_document()` | 270 | 29 |
| `corrections/features.py` | `featurize()` | 232 | 19 |
| `corrections/training_loop.py` | `train_classifier()` | 218 | 18 |
| `analysis/misc_titles.py` | `detect_misc_title_regions()` | 203 | 35 |
| `analysis/abbreviations.py` | `detect_abbreviation_regions()` | 198 | 22 |
| `analysis/legends.py` | `detect_legend_regions()` | 197 | 24 |
| `analysis/gnn/graph.py` | `build_document_graph()` | 187 | 24 |
| `analysis/semantic_regions.py` | `_grow_region_from_anchor()` | 163 | 29 |

`_apply_ml_feedback()` at 347 lines with 59 branch points is doing the work of at least 4-5 separate functions. `detect_standard_detail_regions()` and the other `detect_*` functions all follow the same pattern — they're ripe for extracting shared logic into a base detection framework.

**Recommendation:** Break functions over 150 lines into named, testable sub-functions. The analysis detectors share common patterns (scan blocks, match criteria, grow region, validate boundaries) that could be extracted into a shared detection framework, reducing each detector by 40-60%.

---

## Issue 4: Low Comment Density in Complex Functions

**Severity: Medium**
**Category: Code Clarity**

The six most complex functions in the codebase have inline comment rates between 5% and 19% of code lines. For functions with 20+ branch points, this is too low — especially when the functions contain domain-specific heuristics and thresholds.

| Function | Lines | Comments | Rate |
|----------|-------|----------|------|
| `_run_document_checks()` | 158 | 7 | 5% |
| `_apply_ml_feedback()` | 347 | 31 | 11% |
| `run_document()` | 270 | 24 | 11% |
| `featurize()` | 232 | 23 | 12% |
| `detect_standard_detail_regions()` | 274 | 37 | 18% |
| `_grow_region_from_anchor()` | 163 | 23 | 19% |

`_run_document_checks()` at 5% is the worst — 158 lines with only 7 comments means most of the cross-page validation logic is unexplained. `run_document()` has memory management decisions (freeing `ctx.chars`, calling `gc.collect()`) that have no inline explanation of why.

**Recommendation:** Target 20-25% comment density for functions over 100 lines. Focus comments on "why" decisions: threshold values, memory management choices, ordering dependencies between steps, and heuristic reasoning. The "what" is already clear from the code; the "why" is what's missing.

---

## Issue 5: Near-Zero Assertions in Source Code (2 runtime assertions)

**Severity: Medium**
**Category: Defensive Programming**

The entire `src/plancheck/` directory contains only 2 runtime assertions, both in `corrections/candidate_features.py` (verifying feature vector dimensions). For a codebase that processes construction documents with complex spatial logic, geometric calculations, and multi-stage data transformations, this means assumptions about data integrity go completely unverified.

**Key places where assertions would catch bugs early:**

- Pipeline stage inputs: assert boxes, page dimensions, and config values are sane before expensive computation
- Geometric operations: assert bounding boxes have x0 < x1 and y0 < y1
- OCR token processing: assert confidence values are in [0, 1]
- Feature vectors: assert consistent dimensionality (the two existing assertions prove this catches real bugs)
- Model predictions: assert output labels are in the known set

**Recommendation:** Add assertions at pipeline stage boundaries — the seams between ingest/tocr/vocr/reconcile/grouping/analysis. These are the points where bad data from one stage propagates into the next. An assertion that catches a negative bounding box at the tocr/vocr boundary is worth more than a traceback six stages later in the export.

---

## Issue 6: Errors Logged at Wrong Severity Level

**Severity: Medium**
**Category: Observability**

20 exception handlers log failures at `debug` or `warning` level when they should be `error` or at minimum `warning` with `exc_info=True`. In production, most logging configurations filter out `debug` entirely, making these failures invisible.

**Worst pattern — failures logged at `debug`:**

| File | Line | What's hidden |
|------|------|---------------|
| `ml_feedback.py` | 156, 170 | Vision feature extractor and text embedder init failures |
| `ml_feedback.py` | 204, 359, 362 | Feature cache reads, drift detection, and the *entire ML feedback flow* |
| `pipeline_stages.py` | 574 | Layout model prediction failure |
| `corrections/text_embeddings.py` | 159 | Text embedding encode failure |
| `corrections/retrain_trigger.py` | 265, 284 | Auto-rollback and drift stats update failures |
| `export/run_loader.py` | 313 | Config reconstruction failure |
| `pipeline.py` | 202 | Stage callback failure |

The entire ML feedback subsystem (`ml_feedback.py`) logs all its failures at `debug` level. If a user reports that ML predictions aren't improving despite corrections, the diagnostic trail is empty at default log levels.

**Recommendation:** Any catch that represents an unexpected failure (not just an expected "library not installed" case) should be at minimum `log.warning` with `exc_info=True`. Reserve `log.debug` for truly expected optional-dependency absence, not for runtime failures.

---

## Issue 7: Public Functions Missing Docstrings (33 functions)

**Severity: Low-Medium**
**Category: Code Clarity / API Surface**

33 public functions (no leading underscore) lack docstrings. The overall rate is good — 488 out of 521 public functions (94%) do have docstrings — but the gaps are in visible API surface areas:

**Notable gaps:**

| Module | Function | Why it matters |
|--------|----------|----------------|
| `analysis/gnn/model.py` | `train_gnn()`, `save_gnn()`, `load_gnn()` | GNN model lifecycle — users need to understand training/loading contract |
| `corrections/drift_detection.py` | `to_dict()` | Serialization contract unclear |
| `corrections/retrain_trigger.py` | `to_dict()` | Retrain result serialization |
| `llm/entity_extraction.py` | `materials()`, `dimensions()`, `equipment()`, `summary()` | Domain entity accessors — the whole point of the extraction |
| `llm/query_engine.py` | `get()` | Cache retrieval — callers need to know return-on-miss behavior |

**Recommendation:** Prioritize the LLM and GNN modules since those are the most complex and least self-evident interfaces. A one-line docstring is fine for `to_dict()` methods, but `train_gnn()` and `get()` need parameter/return documentation.

---

## Issue 8: Test Coverage Gaps (58 modules with no matching test file)

**Severity: Medium**
**Category: Testability**

58 out of 121 source modules (48%) have no corresponding test file. The well-tested areas (pipeline, grouping, export, corrections store) are strong. The gaps cluster in:

**Analysis detectors (6 modules):** `abbreviations.py`, `box_classifier.py`, `misc_titles.py`, `revisions.py`, `semantic_regions.py`, `standard_details.py` — these contain the core domain logic for identifying regions on construction drawings. They have the highest branch counts in the codebase and are the most likely to produce user-visible bugs.

**Corrections subsystem (8 modules):** `box_groups.py`, `candidate_classifier.py`, `candidate_features.py`, `candidate_outcomes.py`, `db_helpers.py`, `db_lock.py`, `snapshots.py`, `store_utils.py` — the store itself is well-tested via mixins, but the individual mixin modules and utilities lack direct test coverage.

**Export pipeline (5 modules):** `artifacts.py`, `csv_export.py`, `font_map.py` (partially tested), `overlays/colors.py`, `overlays/structural.py` — the final output path that users directly see.

**VOCR internals (9 modules):** `adaptive_stats_base.py`, `backends/base.py`, `backends/surya.py`, `candidates/cross_line.py`, `candidates/density_grid.py`, `candidates/dimension_analysis.py`, `candidates/encoding_quality.py`, `candidates/gap_patterns.py`, `candidates/rendering_analysis.py`

**Recommendation:** Prioritize test coverage for the analysis detector modules — they're the highest-complexity, highest-impact code with the least coverage. Each detector function can be tested with synthetic block/graphics data without needing real PDFs.

---

## Issue 9: Manual Garbage Collection Without Explanation

**Severity: Low**
**Category: Code Clarity**

Two locations use explicit `gc.collect()` calls:

- `pipeline.py:752` — between Phase 1 and Phase 2 of `run_document()`
- `vocr/backends/surya.py:156` — after unloading the Surya model

Neither has an inline comment explaining why the GC call is necessary, what memory pressure it's responding to, or what happens if it's removed. The `pipeline.py` call also has `import gc` inside the function body (line 750) rather than at module level, suggesting it was added as a hotfix.

**Recommendation:** Add a comment explaining the memory budget reasoning (e.g., "Free ~500MB of TOCR char data before loading the 1.5GB Surya model to stay within the 4GB container limit"). Move the `import gc` to module level. If the GC is a workaround for a specific deployment constraint, note that — otherwise future developers may remove it as unnecessary.

---

## Issue 10: Lint Suppression Accumulation (99 total, 42 BLE001)

**Severity: Low**
**Category: Code Hygiene**

99 `# noqa` or `# type: ignore` comments across the source. 42 of those are `# noqa: BLE001` (broad exception catches). While each individual suppression is justified, the volume indicates the codebase has normalized broad exception handling as the default pattern rather than a deliberate choice.

**Recommendation:** Audit the BLE001 suppressions in a dedicated pass. Many of the `except Exception` blocks could be narrowed to specific exception types: `ImportError` for optional dependency loads, `sqlite3.Error` for database operations, `ValueError`/`KeyError` for data parsing. Narrowing the catch means unexpected exceptions (the ones you *want* to find) will still propagate and produce tracebacks instead of being silently absorbed.

---

## Summary: Priority Order

| Priority | Issue | Impact | Effort |
|----------|-------|--------|--------|
| 1 | Silent exception swallowing (32 locations) | High | Low — add `log.warning(..., exc_info=True)` to each |
| 2 | Pipeline functions with zero logging (15 functions) | High | Low — add entry/exit log lines |
| 3 | Wrong severity level on error logging (20 locations) | Medium | Low — promote `debug` → `warning` |
| 4 | Test coverage for analysis detectors (6 modules) | Medium | Medium — need synthetic test data |
| 5 | Oversized functions (top 10) | Medium | Medium — extract sub-functions |
| 6 | Low comment density in complex functions | Medium | Low — add "why" comments |
| 7 | Near-zero assertions | Medium | Low — add at stage boundaries |
| 8 | Missing docstrings (33 functions) | Low-Med | Low |
| 9 | GC without explanation | Low | Trivial |
| 10 | Lint suppression accumulation | Low | Medium — narrow exception types |

Issues 1-3 are the highest-value, lowest-effort improvements. They're all observability fixes — making the code tell you what it's doing when things go wrong. That's the diagnostic gap that separates "good clean code" from "code I can debug at 2 AM."
