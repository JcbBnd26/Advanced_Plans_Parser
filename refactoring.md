# Advanced Plan Parser — Master Refactoring Plan

**Date:** March 3, 2026
**Scope:** Internal structural refactoring only — no public API changes
**Baseline:** 1,499+ passing tests (as of CLEANUP_LOG round 3, Mar 1 2026)
**Guardrails:** Respects `EXECUTION_PLAN.md` — Phases 0–3 feature work proceeds independently; this plan targets infrastructure/maintainability debt only

---

## Context & Prior Work

### Already Completed (CLEANUP_LOG.md)

| Round | Date | Changes |
|-------|------|---------|
| 1 | Feb 18 | Removed 36 dead files (wrappers, pycache, debug scripts), updated 13 imports across 9 scripts. 528 tests passing. |
| 2 | Feb 24 | Phase 4 ML infrastructure added (drift detection, retrain trigger, experiments). 1,499 tests. |
| 3 | Mar 1 | Overlay PNGs removed (3 scripts), MLOps tab removed, tab renamed. |

### Active PR

PR #1 (`pr/copilot-swe-agent/1`) — "Refactor tab_annotation.py monolith into focused GUI modules" — reduced `tab_annotation.py` from 4,415 → 962 lines by extracting mixins (`event_handler.py`, `canvas_renderer.py`, `context_menu.py`, `pdf_loader.py`, `annotation_store.py`, `annotation_state.py`).

### production_review.md Flags (Refactoring-Relevant)

| # | Issue | Addressed In |
|---|-------|-------------|
| 1 | No error recovery/checkpoint/resume in batch pipeline | Phase 1 (pipeline split) |
| 5 | Silent failures — bare `except: pass` everywhere | Phase 4 §25 |
| 6 | No data validation at API boundaries | Phase 2 §8, Phase 3 §22 |
| 7 | 130+ config params with no versioning/validation | Phase 2 §14 |
| 8 | SQLite as SPOF, no backup/migration | Phase 1 §4 |

---

## Phase 1 — Critical God Modules

> Files >1,500 lines each. Highest structural debt, highest regression risk.
> Run full `pytest` after each split. Verify `python -c "from plancheck import *"` still works.

---

### 1.1 Split `src/plancheck/pipeline.py` (2,127 lines)

**What it does today:** Monolithic 9-stage pipeline orchestrator containing data containers (`PageResult`, `DocumentResult`, `StageResult`), 8 `_run_*_stage()` helpers, `run_pipeline()`, `_apply_ml_feedback()` (~200 lines with 4 sequential passes and deep nesting), `_run_document_checks()`, and `run_document()`.

**Code smells:**
- God module — data containers, orchestration, ML logic, and document checks all in one file
- `_apply_ml_feedback()` at ~200 lines with 4 sequential passes (prior corrections, ML relabelling, confidence scoring, drift detection) and 4+ nesting levels
- `PageResult.to_dict` / `from_dict` — ~200 lines each of manual serialization
- Law of Demeter violation: `_apply_ml_feedback` directly accesses `store._conn.execute(...)`, bypassing the `CorrectionStore` public API

**Target structure:**

```
src/plancheck/
├── pipeline.py              # Thin orchestrator (~300 lines): run_pipeline(), run_document()
├── page_result.py           # PageResult, DocumentResult, StageResult dataclasses + serialization
├── pipeline_stages.py       # _run_tocr_stage(), _run_vocr_stage(), etc. — one function per stage
├── ml_feedback.py           # _apply_ml_feedback() and its helpers
└── document_checks.py       # _run_document_checks()
```

**Steps:**

1. Create `page_result.py` and move `PageResult`, `DocumentResult`, `StageResult` (and any supporting types like `CheckResult`) into it. Update all imports across the codebase (`grep -r "from plancheck.pipeline import PageResult"` etc.).
2. Create `pipeline_stages.py` and extract the 8 `_run_*_stage()` functions. Each function takes explicit parameters instead of reaching into `self` or module globals. Update `pipeline.py` to import and call them.
3. Create `ml_feedback.py` and move `_apply_ml_feedback()` plus its internal helpers. **Fix the encapsulation violation**: replace `store._conn.execute(...)` calls with new public methods on `CorrectionStore` (coordinate with §1.4 below).
4. Create `document_checks.py` and move `_run_document_checks()`.
5. Verify `pipeline.py` is now ≤300 lines — a thin orchestrator that imports from the four new modules.
6. Update `src/plancheck/__init__.py` re-exports if any public symbols moved.
7. Run `pytest` — all 1,499+ tests must pass.

**Risk:** Highest-traffic module. Import changes will touch many files.
**Mitigation:** Keep all public function signatures identical. Only internal structure changes.

---

### 1.2 Split `src/plancheck/grouping/clustering.py` (2,078 lines)

**What it does today:** Four distinct algorithmic concerns in one file: row-truth layer (`build_lines`), spatial clustering (`build_clusters_v2`), notes column grouping (`group_notes_columns`, `link_continued_columns`), and header/notes marking (`mark_headers`, `mark_notes`). Contains ~15 internal `_helper` functions making it hard to trace which helper serves which concern.

**Code smells:**
- Mixed responsibilities — four independent algorithms sharing a filename
- `build_clusters_v2` and `group_notes_columns` each 200+ lines with complex geometry logic
- ~15 internal helper functions with unclear ownership

**Target structure:**

```
src/plancheck/grouping/
├── __init__.py              # Re-exports (update to point to new modules)
├── clustering.py            # build_clusters_v2 + spatial helpers (~500 lines)
├── row_truth.py             # build_lines, compute_median_space_gap, split_line_spans
├── notes_grouping.py        # group_notes_columns, link_continued_columns
└── header_marking.py        # mark_headers, mark_notes
```

**Steps:**

1. Identify which of the ~15 helper functions belong to which concern. Map each helper to its consumer(s).
2. Create `row_truth.py` — move `build_lines`, `compute_median_space_gap`, `split_line_spans`, and their dedicated helpers.
3. Create `notes_grouping.py` — move `group_notes_columns`, `link_continued_columns`, and their helpers.
4. Create `header_marking.py` — move `mark_headers`, `mark_notes`, and their helpers.
5. Leave `clustering.py` with `build_clusters_v2` and spatial helpers only.
6. Update `src/plancheck/grouping/__init__.py` — currently re-exports 14 functions from `clustering.py`; update to re-export from the correct new modules. Keep `__all__` intact.
7. Fix any cross-helper dependencies (some helpers may be shared — extract shared ones into a `grouping_utils.py` if needed).
8. Run `pytest tests/grouping/` — verify all clustering/grouping tests pass.
9. Run full `pytest`.

**Risk:** Geometry helpers may have subtle cross-dependencies.
**Mitigation:** Map helper call-graphs before moving anything. Use `__init__.py` re-exports as a compatibility layer.

---

### 1.3 Split `scripts/gui/event_handler.py` (1,883 lines)

**What it does today:** A single massive mixin class extracted from `tab_annotation.py` in the active PR. Contains mouse events, keyboard shortcuts, drag logic, undo/redo stack, multi-select, merge, group management, and copy/paste — all in one class. Directly manipulates `self._canvas`, `self._boxes`, `self._undo_stack`, `self.state`.

**Code smells:**
- Too many methods in one mixin — ~40+ methods covering 6+ distinct concerns
- Deep coupling to canvas/box/undo internals throughout every method

**Target structure:**

```
scripts/gui/
├── event_handler.py         # REMOVED or becomes a thin re-export facade
├── mouse_handler.py         # Click, drag, hover, resize events
├── keyboard_handler.py      # Keyboard shortcuts, copy/paste
├── undo_redo.py             # Undo/redo stack management
├── selection.py             # Multi-select, lasso, group select
└── merge_handler.py         # Merge, group, split operations
```

**Steps:**

1. Inventory all methods in `EventHandlerMixin`. Group by concern:
   - **Mouse**: `_on_click`, `_on_drag`, `_on_hover`, `_on_resize_*`, `_on_motion`, `_on_release`
   - **Keyboard**: `_on_key_*`, `_copy`, `_paste`, `_delete_selected`
   - **Undo/Redo**: `_push_undo`, `_undo`, `_redo`, `_snapshot_state`
   - **Selection**: `_select_box`, `_deselect_all`, `_lasso_*`, `_group_select`
   - **Merge**: `_merge_boxes`, `_split_box`, `_group_*`
2. Create each new mixin class. Each mixin inherits from the same abstract base or uses the same `self._canvas` / `self._boxes` protocol.
3. Update `tab_annotation.py` to inherit from the 5 new mixins instead of `EventHandlerMixin`.
4. If any method in one mixin calls a method in another mixin, document the dependency and ensure MRO resolves correctly.
5. Run GUI smoke test via `launch_gui.bat` — exercise every annotation tool.
6. Run `pytest tests/gui/`.

**Risk:** Mixin MRO (method resolution order) can cause subtle bugs.
**Mitigation:** Define a clear mixin protocol (abstract methods on a base mixin) for shared state access. Test cooperatively with `tab_annotation.py`.

---

### 1.4 Split `src/plancheck/corrections/store.py` (1,714 lines)

**What it does today:** God class `CorrectionStore` containing 15+ SQL table DDL statements, 30+ CRUD methods, migration logic (`_migrate_locked` with imperative ALTER TABLE checks), and export utilities. Covers documents, detections, corrections, box groups, training examples, candidate outcomes, training runs, feature cache, and snapshots.

**Code smells:**
- God class — 9 distinct data domains in one class
- ~170 lines of inline DDL at the top of the file
- Migration logic interleaved with business logic
- External code (`pipeline.py`'s `_apply_ml_feedback`) bypasses the public API and accesses `store._conn.execute(...)` directly

**Target structure:**

```
src/plancheck/corrections/
├── store.py                 # Composition facade — CorrectionStore delegates to sub-stores
├── schema.py                # All DDL (CREATE TABLE), migrations (_migrate_locked)
├── detection_store.py       # Detection CRUD methods
├── correction_store.py      # Correction CRUD methods
├── group_store.py           # Box group CRUD methods
├── training_store.py        # Training runs, feature cache, snapshots, candidate outcomes
└── (existing files unchanged)
```

**Steps:**

1. Create `schema.py` — extract all `CREATE TABLE` statements and `_migrate_locked` logic. Expose `ensure_schema(conn)` and `migrate(conn)` functions.
2. Create `detection_store.py` — extract detection-related CRUD methods into a `DetectionStore` class that takes a `sqlite3.Connection`.
3. Create `correction_store.py` — extract correction-related CRUD.
4. Create `group_store.py` — extract box group CRUD.
5. Create `training_store.py` — extract training runs, feature cache, snapshots, candidate outcomes.
6. Refactor `store.py` `CorrectionStore` to compose the sub-stores:
   ```python
   class CorrectionStore:
       def __init__(self, db_path):
           self._conn = sqlite3.connect(db_path)
           ensure_schema(self._conn)
           migrate(self._conn)
           self.detections = DetectionStore(self._conn)
           self.corrections = CorrectionSubStore(self._conn)
           self.groups = GroupStore(self._conn)
           self.training = TrainingStore(self._conn)
       # Delegate or re-expose methods for backward compat
   ```
7. Add the public methods needed by `_apply_ml_feedback` (from §1.1 step 3) so external code no longer accesses `_conn` directly.
8. Update `src/plancheck/corrections/__init__.py` — keep `CorrectionStore` as the public API.
9. Run `pytest tests/corrections/` (18 test files).
10. Run full `pytest`.

**Risk:** SQLite connection sharing between sub-stores must be thread-safe.
**Mitigation:** All sub-stores share the same connection object; the facade owns the connection lifecycle. No new connections created.

---

### 1.5 Refactor `src/plancheck/vocr/candidates.py` (1,517 lines)

**What it does today:** Contains 18 `_detect_*` methods, each a separate heuristic strategy for detecting VOCR candidates. Every method follows the same pattern: iterate blocks → check geometric/textual conditions → create `VocrCandidate`.

**Code smells:**
- 18 parallel methods with duplicated iterate → check → create boilerplate
- Adding a new detection strategy requires modifying this file
- No way to enable/disable individual strategies via config

**Target approach:** Strategy pattern

**Target structure:**

```
src/plancheck/vocr/
├── candidates.py            # CandidateOrchestrator + base class + strategy registry
├── strategies/
│   ├── __init__.py          # Auto-imports all strategy modules
│   ├── table_detector.py    # TableDetector(CandidateStrategy)
│   ├── note_detector.py     # NoteDetector(CandidateStrategy)
│   ├── legend_detector.py   # LegendDetector(CandidateStrategy)
│   ├── ...                  # One file per detection method (18 total)
│   └── registry.py          # Strategy registry + @register_strategy decorator
```

**Steps:**

1. Define the `CandidateStrategy` abstract base class:
   ```python
   class CandidateStrategy(ABC):
       name: str
       @abstractmethod
       def detect(self, blocks, page_width, page_height, config) -> list[VocrCandidate]: ...
   ```
2. Create the `strategies/` sub-package with `registry.py` containing a `@register_strategy` decorator.
3. Convert each `_detect_*` method into a `class XxxDetector(CandidateStrategy)` in its own file under `strategies/`.
4. Refactor `candidates.py` to iterate `get_registered_strategies()` instead of calling 18 methods.
5. Update `src/plancheck/vocr/__init__.py` re-exports.
6. Run `pytest tests/vocr/` (8 test files).
7. Run full `pytest`.

**Risk:** Some `_detect_*` methods may share helper functions.
**Mitigation:** Extract shared helpers into `strategies/_helpers.py`. Keep the `CandidateStrategy` interface minimal.

---

## Phase 2 — High-Severity Structural Issues

> Files 800–1,500 lines with significant but less critical structural problems.
> Run `pytest` after each item.

---

### 2.1 Split `src/plancheck/analysis/structural_boxes.py` (1,329 lines)

**What it does today:** Three independent analysis concerns in one file:
- `detect_structural_boxes` — box detection via image processing
- `classify_structural_boxes` — box classification using heuristics
- `detect_semantic_regions` — semantic region growth from classified boxes

Each concern is 300–400 lines.

**Target structure:**

```
src/plancheck/analysis/
├── structural_boxes.py      # KEPT — just detect_structural_boxes (~400 lines)
├── box_classifier.py        # classify_structural_boxes (~400 lines)
└── semantic_regions.py      # detect_semantic_regions (~400 lines)
```

**Steps:**

1. Identify internal helpers used by each of the three functions. Map dependencies.
2. Create `box_classifier.py` — extract `classify_structural_boxes` and its helpers.
3. Create `semantic_regions.py` — extract `detect_semantic_regions` and its helpers.
4. Update `src/plancheck/analysis/__init__.py` re-exports.
5. Run `pytest tests/analysis/` (14 test files).

---

### 2.2 Fix `scripts/gui/overlay_viewer.py` (1,316 lines)

**Two problems:**

**A. Duplicated pipeline orchestration:** `render_overlay()` manually imports and calls individual analysis functions (TOCR → clustering → analysis), reimplementing what `run_pipeline()` already does. This means bug fixes to the pipeline may not propagate to the overlay viewer.

**B. Mixed concerns:** Pure rendering function and tkinter GUI tab in one file.

**Target structure:**

```
scripts/gui/
├── overlay_renderer.py      # render_overlay() — pure rendering (no tkinter)
└── overlay_tab.py           # OverlayViewerTab(ttk.Frame) — GUI shell
```

**Steps:**

1. Refactor `render_overlay()` to call `run_pipeline()` with appropriate stage-gate parameters instead of reimplementing stages. Map out which pipeline stages the overlay viewer currently calls and which `run_pipeline()` parameters produce the same result.
2. Split the file: `overlay_renderer.py` (rendering logic) and `overlay_tab.py` (tkinter tab).
3. Update `scripts/gui/gui.py` to import `OverlayViewerTab` from `overlay_tab.py`.
4. Run GUI smoke test — verify overlays render identically before and after.

---

### 2.3 Refactor `src/plancheck/models.py` (1,233 lines)

**What it does today:** ~15 dataclasses each with manually-written `to_dict()` and `from_dict()` classmethod. The serialization pattern for region types (`Legend`, `Abbreviation`, `Revision`, `StandardDetail`) is nearly identical — `header_block_index`, bbox, and entries serialization repeated 4 times. No `__post_init__` validation on any class.

**Code smells:**
- ~800 lines of boilerplate serialization (out of 1,233 total)
- Copy-paste pattern across 4 region types
- No validation — addresses `production_review.md` item #6

**Approach:** Pydantic is already a project dependency. Migrate dataclasses to Pydantic `BaseModel` subclasses, which provides `.model_dump()` / `.model_validate()` for free, plus automatic validation.

**Steps:**

1. Audit all `to_dict()` / `from_dict()` methods. Categorize:
   - **Trivial** (just field copying) — Pydantic handles automatically
   - **Custom** (e.g., nested object serialization, enum conversion) — override `model_serializer` / `model_validator`
2. Convert each dataclass to a Pydantic `BaseModel`. Start with leaf types (no nested models) and work up.
3. For the 4 region types (`Legend`, `Abbreviation`, `Revision`, `StandardDetail`), create a shared `RegionBase` model with common fields.
4. Add `__post_init__` / `model_validator` checks for required invariants (e.g., bbox has 4 elements, confidence ∈ [0,1]).
5. Update all call sites: `.to_dict()` → `.model_dump()`, `.from_dict(d)` → `Model.model_validate(d)`.
6. Run `pytest` — verify round-trip serialization is identical.

**Risk:** Pydantic models have different identity/hashing semantics than dataclasses.
**Mitigation:** Run full test suite. Pay particular attention to tests that compare objects by identity or use them as dict keys.

---

### 2.4 Refactor `src/plancheck/checks/semantic_checks.py` (1,096 lines)

**What it does today:** 15 independent check functions, each returning a list of `CheckResult`. A manual `run_all_checks()` at line ~1010 calls each check function explicitly. Adding a new check requires editing both the check function AND the orchestrator.

**Target approach:** Registry/decorator pattern

**Steps:**

1. Create a `_CHECK_REGISTRY: list[Callable]` at module level.
2. Create a `@register_check` decorator:
   ```python
   def register_check(fn):
       _CHECK_REGISTRY.append(fn)
       return fn
   ```
3. Decorate each of the 15 check functions with `@register_check`.
4. Rewrite `run_all_checks()` to iterate `_CHECK_REGISTRY`:
   ```python
   def run_all_checks(page_result, config):
       results = []
       for check_fn in _CHECK_REGISTRY:
           results.extend(check_fn(page_result, config))
       return results
   ```
5. Verify function signatures are uniform. If not, normalize them.
6. Run `pytest tests/checks/` (2 test files).

---

### 2.5 Refactor `src/plancheck/reconcile/reconcile.py` (1,059 lines)

**What it does today:** Multi-stage OCR reconciliation. Contains a symbol-injection pass, confidence-scoring pass, and debug-output generation. Private functions `_center`, `_has_allowed_symbol`, `_has_numeric_symbol_context` are consumed by `src/plancheck/export/reconcile_overlay.py` via private import — a coupling concern.

**Target structure:**

```
src/plancheck/reconcile/
├── reconcile.py             # Main reconciliation orchestrator (~400 lines)
├── symbol_injection.py      # Symbol injection pass
├── confidence_scoring.py    # Confidence scoring pass
├── debug_output.py          # Debug output generation
└── helpers.py               # center(), has_allowed_symbol(), has_numeric_symbol_context() — PUBLIC
```

**Steps:**

1. Identify the three passes within `reconcile()` and their boundaries.
2. Extract symbol injection → `symbol_injection.py`.
3. Extract confidence scoring → `confidence_scoring.py`.
4. Extract debug output → `debug_output.py`.
5. Move `_center`, `_has_allowed_symbol`, `_has_numeric_symbol_context` to `helpers.py` as public functions (drop the `_` prefix).
6. Update `src/plancheck/export/reconcile_overlay.py` to import from `reconcile.helpers` instead of `reconcile.reconcile`.
7. Update `src/plancheck/reconcile/__init__.py` re-exports.
8. Run `pytest tests/reconcile/` (2 test files).

---

### 2.6 Fix `scripts/gui/tab_diagnostics.py` (983 lines)

**What it does today:** Diagnostics tab with 5 collapsible sections. Suspected code-structure bug: `_on_mousewheel` method body appears to contain UI construction code that belongs in `_build_ui`.

**Steps:**

1. Read the file and confirm the code-structure bug. If `_on_mousewheel` contains UI construction code, move it to `_build_ui`.
2. Extract the 5 collapsible sections into small widget classes (one per section), each inheriting from `CollapsibleFrame` (already available in `scripts/gui/widgets.py`).
3. `tab_diagnostics.py` becomes a thin composer that instantiates the 5 section widgets.
4. Run GUI smoke test.

---

### 2.7 Trim `scripts/gui/tab_annotation.py` (962 lines)

**What it does today:** Post-refactor coordinator for the annotation tab. Still 962 lines, which is large for what should be a "thin coordinator" composing 6 mixins.

**Steps:**

1. Audit the file. Identify:
   - How much of `__init__` is setup that could be in a `_build_ui` or `_wire_events` helper
   - Whether property forwarding / boilerplate can be reduced
   - Whether any logic crept back in that belongs in a mixin
2. Extract verbose `__init__` setup into helpers. Target ≤500 lines.
3. Run GUI smoke test + `pytest tests/gui/`.

---

### 2.8 Split `scripts/gui/tab_runs.py` (889 lines)

**What it does today:** Two responsibilities in one tab: run listing/browsing and report viewing.

**Target structure:**

```
scripts/gui/
├── tab_runs.py              # KEPT as coordinator
├── run_browser.py           # Run listing, selection, filtering
└── report_viewer.py         # Report rendering, export buttons
```

**Steps:**

1. Identify the boundary between run-browsing and report-viewing code.
2. Extract run-browsing into `run_browser.py` (list widget, sort/filter, selection).
3. Extract report-viewing into `report_viewer.py` (HTML rendering, export).
4. `tab_runs.py` becomes a coordinator that composes the two.
5. Run GUI smoke test.

---

### 2.9 Split `src/plancheck/config.py` (777 lines, ~150+ fields)

**What it does today:** A single `GroupingConfig` dataclass configuring every pipeline stage. 150+ fields with no grouping, versioning, or migration support.

**Code smells:**
- Mega-config — changes to any stage's config risk touching unrelated fields
- No config versioning (production_review.md item #7)
- Default values scattered across the single class

**Target structure:**

```python
# src/plancheck/config.py
@dataclass
class TOCRConfig:
    ...  # TOCR-specific fields

@dataclass
class VOCRConfig:
    ...  # VOCR-specific fields

@dataclass
class ReconcileConfig:
    ...

@dataclass
class GroupingStageConfig:
    ...

@dataclass
class AnalysisConfig:
    ...

@dataclass
class ExportConfig:
    ...

@dataclass
class MLConfig:
    ...

@dataclass
class PipelineConfig:
    tocr: TOCRConfig = field(default_factory=TOCRConfig)
    vocr: VOCRConfig = field(default_factory=VOCRConfig)
    reconcile: ReconcileConfig = field(default_factory=ReconcileConfig)
    grouping: GroupingStageConfig = field(default_factory=GroupingStageConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    ml: MLConfig = field(default_factory=MLConfig)
    version: int = 1  # Config versioning
```

**Steps:**

1. Categorize all ~150 fields by which pipeline stage uses them. A field may be used by multiple stages — assign it to the most specific sub-config, or to a `CommonConfig` if shared.
2. Create sub-config dataclasses within `config.py` (keep one file for now — this is a field-grouping refactor, not a file-splitting refactor).
3. Create the top-level `PipelineConfig` that composes sub-configs.
4. Add `GroupingConfig` as a backward-compatible alias:
   ```python
   # Backward compat — all existing code uses GroupingConfig
   GroupingConfig = PipelineConfig
   ```
5. Gradually update call sites to use `config.tocr.xxx` instead of `config.xxx`.
6. Add a `version` field and a `migrate_config(old_dict) -> PipelineConfig` helper.
7. Run full `pytest`.

**Risk:** Every pipeline stage and GUI panel touches config. Highest fan-out change.
**Mitigation:** The `GroupingConfig = PipelineConfig` alias maintains backward compat. Callers migrate incrementally.

---

### 2.10 Extract training loop from `src/plancheck/corrections/classifier.py` (824 lines)

**What it does today:** Mixes model management, training loop, prediction, and evaluation in one class.

**Steps:**

1. Extract the training loop into `training_loop.py` — takes a model + data, returns trained model + metrics.
2. Leave `classifier.py` with model construction, prediction, and evaluation.
3. Run `pytest tests/corrections/`.

---

## Phase 3 — Medium-Severity & Cross-Cutting Issues

> Files 500–800 lines, duplication patterns, incomplete implementations.
> Lower risk, moderate maintainability benefit.

---

### 3.1 `scripts/gui/annotation_store.py` (765 lines)

**Problem:** Mixes label registry management, filter logic, and model training trigger in one mixin.

**Steps:**

1. Extract label registry → `label_registry_mixin.py`
2. Extract filter logic → `filter_mixin.py`
3. Extract model training trigger → `training_mixin.py`
4. Update `tab_annotation.py` to compose the three new mixins.

---

### 3.2 `scripts/runners/run_pdf_batch.py` (743 lines)

**Problem:** `process_page()` partially re-implements pipeline orchestration logic instead of fully delegating to `run_pipeline()`.

**Steps:**

1. Audit `process_page()` — identify what it does beyond calling `run_pipeline()`.
2. If the extra logic is stage-gating or error wrapping, add those capabilities to `run_pipeline()` itself.
3. Simplify `process_page()` to a thin wrapper around `run_pipeline()`.

---

### 3.3 `src/plancheck/export/overlay.py` (952 lines)

**Problem:** Large PIL-based overlay drawing library. Could be split by overlay type.

**Steps:**

1. Identify distinct overlay types (detection overlay, structural overlay, grouping overlay, etc.).
2. Extract each into its own function module or class.
3. Keep `overlay.py` as the public API that delegates.

---

### 3.4 Deduplicate/Retire `scripts/runners/run_pdf_page.py` (130 lines)

**Problem:** Uses a stale code path — calls `build_clusters_v2` directly instead of `run_pipeline()`. Duplicates `summarize()` from `run_pdf_batch.py`. May produce different results than the batch runner.

**Steps:**

1. Determine if single-page processing is still needed as a separate entry point.
2. If yes: rewrite to call `run_pipeline()` with a single page range. Remove duplicated `summarize()`.
3. If no: remove the file and update any references.

---

### 3.5 Shared base class for VOCR stats modules

**Problem:** `src/plancheck/vocr/method_stats.py` (199 lines) and `src/plancheck/vocr/producer_stats.py` (188 lines) have structurally parallel code.

**Steps:**

1. Extract common logic into `adaptive_stats_base.py` with an `AdaptiveStatsBase` class.
2. Make `MethodStats` and `ProducerStats` inherit from it.
3. Run `pytest tests/vocr/`.

---

### 3.6 Create `analysis/gnn/` sub-package

**Problem:** `gnn_model.py` (350 lines) and `document_graph.py` (402 lines) in `src/plancheck/analysis/` are tightly coupled and share an optional `torch_geometric` dependency.

**Steps:**

1. Create `src/plancheck/analysis/gnn/` package.
2. Move `gnn_model.py` and `document_graph.py` into it.
3. Create `__init__.py` with re-exports.
4. Update `src/plancheck/analysis/__init__.py`.

---

### 3.7 Complete or remove `src/plancheck/validation/schemas.py` (65 lines, 4 schemas)

**Problem:** Partially implemented Pydantic validation — only covers `GlyphBoxSchema`, `SpanSchema`, `LineSchema`, `VocrCandidateSchema`. Doesn't cover the other 11+ serializable types.

**Decision point:**

- If Phase 2 §2.3 migrates `models.py` to Pydantic, this file becomes redundant — remove it.
- If `models.py` stays as dataclasses, extend this file to cover all types.

---

## Phase 4 — Cross-Cutting Architectural Smells

> These cut across multiple files and require coordinated changes.

---

### 4.1 Eliminate Duplicated Pipeline Orchestration

**Problem:** Three separate locations independently run the pipeline:

| Location | What it does | Canonical? |
|----------|-------------|-----------|
| `src/plancheck/pipeline.py` → `run_pipeline()` | Full 9-stage pipeline | **Yes** |
| `scripts/gui/overlay_viewer.py` → `render_overlay()` | Reimplements TOCR → clustering → analysis inline | No |
| `scripts/runners/run_pdf_batch.py` → `process_page()` | Wraps `run_pipeline()` but adds parallel concerns | Partially |

**Steps:**

1. Add stage-gating parameters to `run_pipeline()`:
   ```python
   def run_pipeline(image, config, *,
                    start_stage="ingest", end_stage="export",
                    skip_stages=None):
   ```
2. Refactor `render_overlay()` to call `run_pipeline(start_stage="ingest", end_stage="analysis")`.
3. Refactor `process_page()` to be a thin wrapper around `run_pipeline()`.
4. Verify all three code paths produce identical results for the same input.

---

### 4.2 Fix Dependency Injection

**Problem:**
- `CorrectionStore()` is instantiated with hardcoded `Path("data/corrections.db")` in `run_pdf_batch.py`, pipeline code, and GUI code.
- `GroupingConfig()` defaults are scattered — created in runner scripts, GUI tabs, and tests with different defaults.

**Steps:**

1. Create a `plancheck.defaults` module:
   ```python
   DEFAULT_DB_PATH = Path("data/corrections.db")
   DEFAULT_CONFIG = PipelineConfig()
   ```
2. All instantiation sites import from `plancheck.defaults` instead of hardcoding paths.
3. Add a `PipelineConfig.from_file(path)` classmethod for config loading.
4. Add a `PipelineConfig.to_file(path)` method for config persistence.

---

### 4.3 Audit and Fix Silent Failures

**Problem:** `production_review.md` item #5 — bare `except: pass` blocks scattered throughout. Confirmed in `scripts/gui/gui.py` `_on_close`. Likely present in other files.

**Steps:**

1. `grep -rn "except:" src/ scripts/` — find all bare except clauses.
2. For each occurrence:
   - If the exception should be logged: add `logger.exception(...)`.
   - If it should propagate: remove the bare except or narrow it.
   - If it's genuinely expected (e.g., optional import): narrow to specific exception type and add a comment.
3. Run full `pytest` to verify no behavior change.

---

### 4.4 Add Missing `__all__` Declarations

**Files:**
- `src/plancheck/analysis/__init__.py` — re-exports from 8 sub-modules but has no `__all__`
- `src/plancheck/vocr/__init__.py` — re-exports from 6 sub-modules but has no `__all__`

**Steps:**

1. Add `__all__` to each file listing all re-exported symbols.
2. Verify `from plancheck.analysis import *` works correctly.

---

## Phase 5 — Bugs Found During Audit

> Not refactoring per se, but discovered during the code review.

---

### 5.1 Fix `scripts/gui/tab_recreation.py` — Missing `import sys`

**Bug:** The `on_done` callback (~line 295) references `sys.platform` to decide whether to call `os.startfile()`, but `sys` is never imported in the file. This will crash on Windows when the user clicks "Open PDF?" after sheet recreation completes.

**Fix:** Add `import sys` to the imports section.

---

## Verification & Rollback Strategy

### After Each Phase

1. **Run full test suite:** `pytest` — all 1,499+ tests must pass.
2. **Import check:** `python -c "from plancheck import *"` — verify all re-exports work.
3. **Circular import check:** `python -m py_compile` on all new/modified modules.
4. **GUI smoke test:** `launch_gui.bat` — cycle through every tab, exercise core workflows.
5. **Pipeline regression test:** Run a sample through `scripts/runners/run_from_args.py` and diff output against a baseline run from `runs/`.

### Rollback

Each phase is an independent commit (or small commit series). If a phase breaks something:
1. `git revert` the phase's commits.
2. Investigate and re-attempt with fixes.
3. No phase depends on a later phase — they can be re-ordered or skipped.

---

## Priority Summary

| Phase | Items | Risk | Effort | Impact |
|-------|-------|------|--------|--------|
| **1** | 5 critical god modules | High | ~5 days | Unlocks independent testing/modification of core modules |
| **2** | 10 high-severity files | Medium | ~5 days | Improves maintainability across pipeline stages and GUI |
| **3** | 7 medium-severity items | Low | ~3 days | Reduces duplication, completes partial implementations |
| **4** | 4 cross-cutting smells | Medium | ~3 days | Eliminates architectural debt across the codebase |
| **5** | 1 bug fix | Trivial | ~5 min | Prevents Windows crash |

**Total estimated effort:** ~16 days of focused refactoring work.

---

## Files NOT Flagged (Clean Code)

The following files were reviewed and found to be well-structured — no refactoring needed:

- `scripts/gui/gui.py` (173 lines) — clean `GuiState` pub/sub
- `scripts/gui/widgets.py` (~350 lines) — 5 reusable widgets, well-documented
- `scripts/gui/worker.py` (~215 lines) — clean threading pattern
- `scripts/gui/annotation_state.py` (~100 lines) — pure dataclass + helpers
- `scripts/gui/canvas_renderer.py` (~260 lines) — clean rendering mixin
- `src/plancheck/vocr/engine.py` (96 lines) — PaddleOCR singleton cache
- `src/plancheck/vocr/targeted.py` (209 lines) — focused per-candidate OCR
- `src/plancheck/corrections/db_lock.py` (77 lines) — small utility
- `src/plancheck/corrections/active_learning.py` (108 lines) — clean
- `src/plancheck/corrections/metrics.py` (157 lines) — lightweight utility
- `src/plancheck/export/font_map.py` (141 lines) — clean utility
- `src/plancheck/export/page_data.py` (78 lines) — clean serialization
- `src/plancheck/export/report.py` (298 lines) — clean HTML generation
- `src/plancheck/export/run_loader.py` (356 lines) — clean Phase 0.1 implementation
- `src/plancheck/llm/client.py` (363 lines) — clean unified LLM client
- `src/plancheck/llm/cost.py` (133 lines) — clean utility
- `src/plancheck/llm/query_engine.py` (428 lines) — clean RAG engine
- All `__init__.py` files — clean re-exports, no business logic
