# Refactoring Plan — Phases 3–5

**Created:** 2026-03-04
**Scope:** Medium-severity issues, cross-cutting architectural smells, and bug fixes identified during code audit.
**Guardrails:** Per [EXECUTION_PLAN.md](docs/EXECUTION_PLAN.md), no public API changes; internal refactoring only.

---

## Phase 3 — Medium-Severity & Cross-Cutting Issues

### 16. tab_annotation.py (4,363 lines) — Extract Label Registry, Filter Logic, and Model Training

> **Note:** The original audit referenced `annotation_store.py` (765 lines), but that file does not exist. The actual monolith is [scripts/gui/tab_annotation.py](scripts/gui/tab_annotation.py) at 4,363 lines.

**Problem:** Three distinct concerns are entangled in a single GUI tab class:
- Label registry persistence (lines 1071–1189)
- Filter control logic (lines 1186–1214, `_filter_label_vars`, `_apply_filters()`)
- Model training trigger (lines 4001–4080, `_on_train_model()`)

**Steps:**
1. Create `scripts/gui/mixins/` sub-package with `__init__.py`
2. Extract `LabelRegistryMixin` to `mixins/label_registry.py`:
   - Move `_label_registry_path()`, `_load_label_registry_json()`, `_save_label_registry_json()`, `_persist_element_type_to_registry()`
   - Keep registry state as mixin instance vars
3. Extract `FilterControlsMixin` to `mixins/filter_controls.py`:
   - Move `_filter_label_vars`, `_rebuild_filter_controls()`, `_apply_filters()`
   - Expose a `filtered_elements` property
4. Extract `ModelTrainingMixin` to `mixins/model_training.py`:
   - Move `_on_train_model()` and related progress callbacks
   - Delegate to `ElementClassifier.train()` as today
5. Update `AnnotationTab` to inherit from all three mixins
6. Run `python -m py_compile scripts/gui/tab_annotation.py` — verify no import cycles

**Verification:**
- Manual smoke test via `launch_gui.bat` — annotation tab loads, filter controls work, training completes
- `pytest tests/gui/` passes

---

### 17. run_pdf_batch.py (743 lines) — Delegate Fully to run_pipeline()

**File:** [scripts/runners/run_pdf_batch.py](scripts/runners/run_pdf_batch.py)

**Problem:** `process_page()` (line 63) wraps `run_pipeline()` but also contains artifact-materialization logic that belongs in the export layer. This creates a secondary orchestration path.

**Steps:**
1. Identify all post-pipeline steps in `process_page()`:
   - Overlay rendering → already in `export/overlay.py`
   - JSON artifact saving → candidate for `export/artifacts.py`
   - Log/summary generation
2. Create `src/plancheck/export/artifacts.py`:
   - `save_page_artifacts(page_result, run_dir, cfg)` — writes JSON, overlay PNGs, debug files
   - Reuse `PageResult.to_dict()` from Phase 0 prereq
3. Refactor `process_page()` to:
   ```python
   result = run_pipeline(ctx, cfg)
   save_page_artifacts(result, run_dir, cfg)
   return result
   ```
4. Ensure parallel-page logic in `run_pdf()` only controls concurrency, not pipeline stages
5. Update imports in `run_pdf_batch.py`

**Verification:**
- `python scripts/runners/run_pdf_batch.py samples/IFC_Operations_Facility.pdf --pages 1-3`
- Diff output artifacts against baseline run — no behavioral changes

---

### 18. overlay.py (952 lines) — Split by Overlay Type

**File:** [src/plancheck/export/overlay.py](src/plancheck/export/overlay.py)

**Problem:** Three conceptually distinct overlay types coexist:
- **Detection overlay** — `draw_overlay()`, block/row/glyph visualization
- **Structural overlay** — `draw_columns_overlay()`, `draw_lines_overlay()`
- **Reconcile overlay** — already in [reconcile_overlay.py](src/plancheck/export/reconcile_overlay.py)

**Steps:**
1. Create `src/plancheck/export/overlays/` sub-package
2. Move detection overlay to `overlays/detection.py`:
   - `draw_overlay()`, `_draw_blocks()`, `_draw_rows()`, `_draw_glyphs()`
3. Move structural overlay to `overlays/structural.py`:
   - `draw_columns_overlay()`, `draw_lines_overlay()`
4. Move color constants and shared helpers to `overlays/colors.py`
5. Retain `overlay.py` as a thin façade re-exporting public functions for backward compatibility:
   ```python
   from .overlays.detection import draw_overlay
   from .overlays.structural import draw_columns_overlay, draw_lines_overlay
   ```
6. Update `src/plancheck/__init__.py` — ensure `draw_overlay` still re-exported

**Verification:**
- `python -c "from plancheck import draw_overlay"` succeeds
- Visual inspection of overlays via `scripts/overlays/render_overlay.py`

---

### 19. Remove run_pdf_page.py (130 lines)

**File:** [scripts/runners/run_pdf_page.py](scripts/runners/run_pdf_page.py)

**Problem:** Calls `build_clusters_v2()` directly, bypassing the full 9-stage pipeline. Contains duplicate `summarize()` logic. This is a stale code path with no documented use case.

**Steps:**
1. Grep codebase and docs for references to `run_pdf_page.py`
2. Check `launch_gui.bat` and any shell scripts
3. Delete `scripts/runners/run_pdf_page.py`
4. Remove any imports/references found in step 1–2

**Verification:**
- `pytest` — no test failures
- Search `runs/*/` logs for invocations of `run_pdf_page.py` — confirm none exist

---

### 20. Shared Base Class for method_stats.py & producer_stats.py

**Files:**
- [src/plancheck/vocr/method_stats.py](src/plancheck/vocr/method_stats.py) (199 lines)
- [src/plancheck/vocr/producer_stats.py](src/plancheck/vocr/producer_stats.py) (188 lines)

**Problem:** Near-identical structure:
- `load_*_stats(path)` → load JSON or empty skeleton
- `update_*_stats(path, stats)` → merge and persist
- `get_*_confidence(method, stats, fallback)` → compute hit-rate

**Steps:**
1. Create `src/plancheck/vocr/adaptive_stats_base.py`:
   ```python
   class AdaptiveStatsBase:
       VERSION: int  # override in subclass
       KEY_FIELD: str  # "method" or "producer"

       @classmethod
       def load(cls, path: Path) -> dict: ...
       @classmethod
       def update(cls, path: Path, run_stats: dict) -> None: ...
       @classmethod
       def get_confidence(cls, key: str, stats: dict, fallback: float) -> float: ...
   ```
2. Refactor `MethodStats(AdaptiveStatsBase)` with `VERSION=2`, `KEY_FIELD="method"`
3. Refactor `ProducerStats(AdaptiveStatsBase)` with `VERSION=1`, `KEY_FIELD="producer"`
4. Preserve existing function signatures as module-level aliases for backward compat:
   ```python
   def load_method_stats(path): return MethodStats.load(path)
   ```

**Verification:**
- `pytest tests/vocr/test_method_stats.py tests/vocr/test_producer_stats.py`
- Verify `data/candidate_method_stats.json` and `data/producer_method_stats.json` still load

---

### 21. Create analysis/gnn/ Sub-Package

**Files:**
- [src/plancheck/analysis/gnn_model.py](src/plancheck/analysis/gnn_model.py) (350 lines)
- [src/plancheck/analysis/document_graph.py](src/plancheck/analysis/document_graph.py) (402 lines)

**Problem:** These files are tightly coupled and have an optional `torch_geometric` dependency. Isolating them improves modularity.

**Steps:**
1. Create `src/plancheck/analysis/gnn/` sub-package
2. Move files:
   - `gnn_model.py` → `gnn/model.py`
   - `document_graph.py` → `gnn/graph.py`
3. Create `gnn/__init__.py`:
   ```python
   try:
       from .model import GNNModel
       from .graph import DocumentGraph
       __all__ = ["GNNModel", "DocumentGraph"]
   except ImportError:
       # torch_geometric not installed
       GNNModel = None
       DocumentGraph = None
       __all__ = []
   ```
4. Update imports in `src/plancheck/analysis/__init__.py` — re-export from `gnn/`
5. Update any direct imports elsewhere in codebase

**Verification:**
- `python -c "from plancheck.analysis.gnn import GNNModel"` — succeeds if torch_geometric installed, graceful None otherwise
- `python -m py_compile src/plancheck/analysis/gnn/__init__.py`

---

### 22. Document schemas.py Scope

**File:** [src/plancheck/validation/schemas.py](src/plancheck/validation/schemas.py) (65 lines)

**Current state:** 4 Pydantic schemas defined:
- `GlyphBoxSchema` — glyph box validation
- `SpanSchema` — span with token indices
- `LineSchema` — line with spans
- `VocrCandidateSchema` — VOCR candidate with outcome

**Decision:** Keep these 4 schemas — they cover the VOCR serialization path which is the most error-prone. Other types use `.model_dump()` / `.model_validate()` directly on their dataclass definitions.

**Steps:**
1. Update module docstring to clarify scope:
   ```python
   """Pydantic schemas for VOCR-related serialization.

   Other types (PageResult, GroupingConfig) use native Pydantic
   methods on their dataclass definitions.
   """
   ```
2. Add `__all__` export list
3. Ensure all 4 schemas are covered by unit tests

**Verification:**
- `pytest tests/validation/`

---

## Phase 4 — Cross-Cutting Architectural Smells

### 23. Eliminate Duplicated Pipeline Orchestration

**Problem:** Three locations independently orchestrate pipeline stages:

| Location | Role | Issue |
|----------|------|-------|
| [src/plancheck/pipeline.py](src/plancheck/pipeline.py) | Canonical 9-stage pipeline | Authoritative |
| [scripts/gui/overlay_viewer.py](scripts/gui/overlay_viewer.py) | Live preview re-runs stages | Duplicates stage logic |
| [scripts/runners/run_pdf_batch.py](scripts/runners/run_pdf_batch.py) | Batch processing | Adds parallel concerns |

**Goal:** All paths go through `run_pipeline()` with stage-gate parameters.

**Steps:**
1. Extend `run_pipeline()` with explicit stage control:
   ```python
   def run_pipeline(
       ctx: PageContext,
       cfg: GroupingConfig,
       start_stage: int = 0,
       end_stage: int = 8,
       cached_result: PageResult | None = None,
   ) -> PageResult:
   ```
2. Refactor `overlay_viewer.py`:
   - Replace inline stage calls with `run_pipeline(ctx, cfg, start_stage=X, end_stage=X, cached_result=prev)`
   - Viewer caches `PageResult` and re-runs only affected stages on config change
3. Refactor `run_pdf_batch.py` (already addressed in Step 17):
   - Parallel wrapper only handles concurrency
   - Per-page logic delegates to `run_pipeline()` + `save_page_artifacts()`
4. Add guard in `pipeline.py` to prevent stage re-ordering bugs

**Verification:**
- `pytest tests/test_pipeline.py`
- Manual test: change config in overlay viewer → verify only affected stages re-run

---

### 24. Centralize Configuration with Dependency Injection

**Problem:**
- `CorrectionStore()` instantiated with hardcoded `Path("data/corrections.db")` in multiple locations
- `GroupingConfig()` defaults scattered across modules

**Steps:**
1. Create `src/plancheck/config/defaults.py`:
   ```python
   from pathlib import Path

   DEFAULT_CORRECTIONS_DB = Path("data/corrections.db")
   DEFAULT_LABEL_REGISTRY = Path("data/label_registry.json")
   DEFAULT_METHOD_STATS = Path("data/candidate_method_stats.json")
   DEFAULT_PRODUCER_STATS = Path("data/producer_method_stats.json")
   ```
2. Create `src/plancheck/config/context.py`:
   ```python
   @dataclass
   class AppContext:
       corrections_db: Path = DEFAULT_CORRECTIONS_DB
       grouping_config: GroupingConfig = field(default_factory=GroupingConfig)
       # ... other shared state

       @classmethod
       def from_env(cls) -> "AppContext":
           """Load from environment variables or config file."""
   ```
3. Refactor call sites:
   - `CorrectionStore(db_path=ctx.corrections_db)`
   - Test fixtures explicitly pass paths
4. Update CLI scripts to construct `AppContext` at entry point

**Verification:**
- `pytest` — all tests pass
- Verify `PLANCHECK_CORRECTIONS_DB` env var override works

---

### 25. Audit and Harden Exception Handling

**Finding:** No bare `except:` blocks exist, but overly broad `except Exception:` may swallow important errors.

**Steps:**
1. Grep for overly broad exception handling:
   ```bash
   grep -rn "except Exception:" src/ scripts/
   grep -rn "except BaseException:" src/ scripts/
   ```
2. Review each match — ensure:
   - Exception is logged (`logger.exception()`)
   - Re-raised if appropriate
   - Not swallowing important errors
3. Pay special attention to:
   - [scripts/gui/gui.py](scripts/gui/gui.py) `_on_close()` — confirm proper cleanup
   - Any file I/O operations
   - Network/LLM API calls
4. Add `# noqa: BLE001` comments where broad catches are intentional (with justification)

**Verification:**
- Run `ruff check src/ scripts/ --select=BLE` — no new warnings

---

### 26. Add Missing `__all__` Declarations

**Affected files:**
- [scripts/gui/__init__.py](scripts/gui/__init__.py)
- [scripts/runners/__init__.py](scripts/runners/__init__.py)
- [scripts/utils/__init__.py](scripts/utils/__init__.py)
- [scripts/diagnostics/__init__.py](scripts/diagnostics/__init__.py)
- [scripts/query/__init__.py](scripts/query/__init__.py)
- [scripts/overlays/__init__.py](scripts/overlays/__init__.py)

**Steps:**
1. For each script package, add `__all__` listing public entry points:
   ```python
   # scripts/gui/__init__.py
   """Tkinter GUI package."""
   __all__ = ["PlanCheckGUI", "AnnotationTab", "OverlayViewer"]
   ```
2. For packages with no public API (scripts run directly), document this:
   ```python
   """CLI runners package. Scripts are invoked directly, not imported."""
   __all__ = []
   ```

**Verification:**
- `python -c "from scripts.gui import *"` — no unexpected imports

---

## Phase 5 — Bugs Found During Audit

### 27. Fix tab_recreation.py — Missing `sys` Import

**File:** [scripts/gui/tab_recreation.py](scripts/gui/tab_recreation.py)

**Problem:** Line 315 references `sys.platform` but `sys` is never imported. This will crash on Windows when the user clicks "Open PDF".

**Code location:**
```python
# Line 315
if sys.platform == "win32":
```

**Fix:**
Add `import sys` to the import block after line 12.

**Verification:**
- `python -m py_compile scripts/gui/tab_recreation.py` — no syntax errors
- Manual test: `launch_gui.bat` → Recreation tab → Generate PDF → click "Open PDF"

---

## Verification Checklist (All Phases)

After completing each phase:

| Check | Command |
|-------|---------|
| Test suite passes | `pytest` (expect 1,499+ passing) |
| Re-exports work | `python -c "from plancheck import *"` |
| No import cycles | `python -m py_compile src/plancheck/**/*.py` |
| GUI smoke test | `launch_gui.bat` — cycle through all tabs |
| Pipeline baseline | `python scripts/runners/run_from_args.py samples/IFC_Operations_Facility.pdf --pages 1` — diff against baseline |

---

## Execution Order

| Order | Item | Est. Effort | Risk |
|-------|------|-------------|------|
| 1 | **Item 27** — Fix missing `sys` import | 5 minutes | None |
| 2 | **Item 19** — Remove stale `run_pdf_page.py` | 15 minutes | Low |
| 3 | **Item 22** — Document schemas.py scope | 15 minutes | None |
| 4 | **Item 26** — Add `__all__` declarations | 30 minutes | None |
| 5 | **Item 21** — Create `gnn/` sub-package | 1 hour | Low |
| 6 | **Item 20** — Extract `AdaptiveStatsBase` | 2 hours | Low |
| 7 | **Item 18** — Split `overlay.py` | 2 hours | Medium |
| 8 | **Item 16** — Extract mixins from `tab_annotation.py` | 3 hours | Medium |
| 9 | **Item 17** — Refactor `run_pdf_batch.py` | 2 hours | Medium |
| 10 | **Item 23** — Consolidate pipeline orchestration | 3 hours | Medium |
| 11 | **Item 24** — Centralize configuration | 2 hours | Medium |
| 12 | **Item 25** — Audit exception handling | 1 hour | Low |

**Total estimated effort:** 2–3 days
