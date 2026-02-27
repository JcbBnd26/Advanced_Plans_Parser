# Production Hardening Plan — Phased Execution

**Source:** `production_review.md` (Senior Engineering Review, 2026-02-25)  
**Approach:** Easy-first progression — surgical fixes first, architectural work last  
**Deployment Model:** Local desktop app (single user)

---

## Phase 1 — Silent Failures & Trivial Fixes _(Issues #5, #6 partial, #15 partial)_

Smallest possible changes with immediate quality-of-life impact. All are 1–5 line edits in isolation.

### Tasks

1. **Fix silent swallow in GUI event bus** — In `scripts/gui/gui.py` (~L75), replace bare `except Exception: pass` in `GuiState.notify` with `except Exception: logging.getLogger(__name__).warning("subscriber error", exc_info=True)`.

2. **Raise log levels on suppressed failures** — In `src/plancheck/pipeline.py` (~L786), change layout model failure from `log.debug(...)` to `log.warning(...)`. Same for `_apply_ml_feedback()` (~L1097–L1409) and LLM checks (~L836). Any `log.debug` on a caught exception that represents a feature silently not working should become `log.warning`.

3. **Add error/warning collection to `PageResult`** — In the `PageResult` class (`pipeline.py` ~L316), add `warnings: list[str] = field(default_factory=list)`. In every existing `except` block in `run_pipeline` that catches and continues, append a human-readable message to `pr.warnings`. Surface these in the GUI `LogPanel` and batch runner per-page output.

4. **Add `__post_init__` validation to `GlyphBox`** — In `src/plancheck/models.py` (~L279–L311), add coordinate sanity checks: `page >= 0`, `x1 > x0`, `y1 > y0`, `len(text) < 10_000`. Raise `ValueError` on violation. Fix any tests constructing intentionally-invalid boxes.

5. **Extract hardcoded data directory to a constant** — In `src/plancheck/config.py`, add `PLANCHECK_DATA_DIR = Path(os.environ.get("PLANCHECK_DATA_DIR", "data"))`. Update the three default model paths (`ml_model_path`, `ml_gnn_model_path`, `ml_drift_stats_path`) and the default DB path in `src/plancheck/corrections/store.py` (~L160) to derive from it.

### Milestone

- [ ] All existing tests pass
- [ ] Manually mock a layout-model failure → WARNING (not DEBUG) appears in logs
- [ ] `GlyphBox(page=-1, ...)` raises `ValueError`

---

## Phase 2 — Memory Safety & Input Guards _(Issues #2, #3)_

Prevent the two most likely desktop crash scenarios: OOM on large docs and unbounded input.

### Tasks

6. **Add `cleanup_images()` to `PageResult`** — New method that calls `.close()` on `background_image` and `preprocess_image`, sets them to `None`. Call it in `scripts/runners/run_pdf_batch.py` immediately after `_materialize_page()` writes to disk (~L614–L627). Also call it per-page in `run_document()` (`pipeline.py` ~L1618). Insert `gc.collect()` every 10 pages in the batch loop.

7. **Add input validation guard** — New helper `validate_input(pdf_path, cfg)` in `src/plancheck/pipeline.py`:
   - File exists and is readable
   - File size < `cfg.max_file_size_mb` (new config field, default 500 MB)
   - Page count < `cfg.max_pages` (new config field, default 1000)
   - Called at top of `run_document()` and in batch runner before the page loop
   - Clear error message surfaced in GUI if validation fails

### Milestone

- [ ] Unit test: `PageResult` with mock PIL image → `cleanup_images()` → fields are `None`
- [ ] Unit test: rejects too-large or too-many-pages PDF with clear `ValueError`
- [ ] Memory usage on a 50-page synthetic batch stays flat (no accumulation)

---

## Phase 3 — Progress Feedback & Error Recovery _(Issues #1, #4)_

Highest-impact UX improvements — users can see what's happening and recover from interruptions.

### Tasks

8. **Page-level progress in GUI** — Extend `PipelineWorker` (`scripts/gui/worker.py` ~L66–L217) to emit `("page_progress", current, total, elapsed_s)`. Add a `QLabel` or secondary progress bar in `scripts/gui/tab_pipeline.py` showing `Page 12 / 50 (~3m remaining)` using moving-average ETA. Wire alongside existing `StageProgressBar`.

9. **Checkpoint/resume for batch processing** — In `scripts/runners/run_pdf_batch.py`:
   - After each successful page, append page number to `checkpoint.json` in the run directory
   - Add `--resume` CLI flag: reads checkpoint, skips completed pages, reloads results from disk
   - Write partial `manifest.json` every 5 pages instead of only at the end (~L653–L668)
   - Add `--skip-errors` flag (default `True`) with clearer per-page error reporting

10. **Retry decorator for transient failures** — New `@retry(max_attempts=2, backoff=1.0)` decorator in `src/plancheck/utils.py`. Apply to PaddleOCR calls in VOCR/TOCR stages. Only retry on `RuntimeError`, `TimeoutError` — not `ValueError`.

### Milestone

- [ ] Start a batch run on 20+ pages, kill mid-run, `--resume` picks up from last checkpoint with identical final output
- [ ] GUI progress label shows page N/M with updating ETA

---

## Phase 4 — Config, Database & UX Hardening _(Issues #7, #8, #11, #13)_

Structural improvements to the parts users interact with most.

### Tasks

11. **Config versioning and migration** — Add `_schema_version: int = 1` class var to `GroupingConfig` in `src/plancheck/config.py`. Include in `to_dict()`. In `from_dict()`, read version and run migration functions for older schemas. Add `validate() -> list[str]` method checking cross-field coherence (e.g., VOCR enabled without TOCR). Call in GUI on save, display warnings inline.

12. **Config presets** — Add `FAST` (VOCR off, checks off), `BALANCED` (defaults), `ACCURATE` (all features on) preset definitions in `src/plancheck/config.py`. Add a preset selector dropdown in `scripts/gui/tab_pipeline.py` that populates config fields.

13. **Automated DB backups & integrity checks** — In `CorrectionStore` (`src/plancheck/corrections/store.py`):
    - Increment a `_write_count`; call `snapshot()` automatically every 50 writes
    - Add `verify_integrity()` via `PRAGMA integrity_check`
    - Call on `__init__` — if it fails, attempt restore from most recent snapshot and warn user
    - Add a `_schema_version` table replacing column-inspection-based migrations

14. **ML model versioning** — New lightweight `ModelRegistry` class in `src/plancheck/corrections/`. Manages a `models/` directory with versioned filenames (e.g., `element_classifier_v5_20260225.pkl`). `get_active_model()` returns latest; `rollback(version)` restores previous. Update `ElementClassifier.load()` (`src/plancheck/corrections/classifier.py` ~L155) to use registry.

15. **GUI polish** — In `scripts/gui/`:
    - Add tooltips to the 20 most impactful config fields
    - Rename confusing tabs ("MLOps" → "Training", "Diagnostics" → "Debug Tools")
    - Show inline validation messages on invalid config values using `validate()`

### Milestone

- [ ] Save a v1 config → bump schema to v2 with a renamed field → load v1 config, confirm migration applies silently
- [ ] Trigger 50+ DB writes → snapshot file appears automatically
- [ ] Preset dropdown loads correct values in GUI

---

## Phase 5 — Structured Logging, Parallelism & Testing _(Issues #9, #10, #14)_

The heavy engineering — depends on the foundation from Phases 1–4.

### Tasks

16. **Structured logging** — Add `structlog` to `requirements.txt`. Create setup function in `src/plancheck/__init__.py`: JSON output to file, human-readable to console, `run_id` context variable bound at run start. Replace ~10 `print()` calls in the batch runner with structured logger calls. Integrate with stdlib `logging` so existing `getLogger()` calls still work.

17. **Process-based parallel page processing** — Using `concurrent.futures.ProcessPoolExecutor` in batch runner (PaddleOCR thread-safety unconfirmed → process isolation):
    - Add `--workers N` CLI flag (default 1, suggest `min(4, cpu_count)`)
    - Each worker processes one page, returns serialized `PageResult`
    - Main process handles checkpoint, progress, manifest
    - PaddleOCR initialized once per worker via `initializer` argument
    - GUI stays single-threaded (parallelism is batch-mode only)

18. **Expand test coverage** — New test files:
    - `tests/stress/test_memory.py` — 50 synthetic pages, assert peak RSS stays bounded (via `tracemalloc`)
    - `tests/integration/test_batch_resume.py` — run, interrupt, resume, verify identical output
    - `tests/integration/test_large_synthetic.py` — 100-page synthetic PDF end-to-end
    - `tests/chaos/test_ocr_failure.py` — monkeypatch PaddleOCR to raise, confirm graceful degradation + `warnings` populated
    - Register under `pytest.ini` marker `slow` so they don't run on every commit

### Milestone

- [ ] `--workers 4` on a 40-page PDF shows ~3x speedup over `--workers 1`
- [ ] `pytest -m slow` passes
- [ ] Structured JSON logs appear in `runs/*/logs/`

---

## Phase 6 — Documentation & Stretch Goals _(Issues #12, #15 remaining, #16, #17)_

Polish and optional forward-looking work.

### Tasks

19. **Deployment documentation** — Create `docs/DEPLOYMENT.md`:
    - System requirements (8 GB RAM min, 4+ cores, 2 GB disk + 50 MB/run)
    - Installation steps (Python version, `pip install -e .`, PaddleOCR Windows setup)
    - Environment variables (`PLANCHECK_DATA_DIR`, `PLANCHECK_LOG_LEVEL`)
    - Scaling guidance (pages/hour at different `--workers` counts)
    - Troubleshooting (common errors → fixes)

20. **Final hardcoded path audit** — Sweep all remaining `Path("data/...")` and `Path("runs/...")`. Ensure all resolve through `PLANCHECK_DATA_DIR` or config fields. Verify the app works when launched from a different CWD.

21. **Optional: Local statistics dashboard** — Opt-in `stats.json` tracking pages processed, avg time/page, error rate by stage. Surface in a "Statistics" section of the Runs tab. Purely local, no network.

22. **Optional: API stub** — Minimal FastAPI app in `src/plancheck/api.py`: `POST /process` → job ID, `GET /status/{job_id}`. Wired to existing `run_document()`. Marked experimental for future web deployment.

### Milestone

- [ ] Follow `docs/DEPLOYMENT.md` on a clean machine — app works
- [ ] Launch from a different CWD with env vars — paths resolve correctly
- [ ] All 17 review items addressed

---

## Key Decisions

| Decision | Rationale |
|----------|-----------|
| **Rate limiting → input guards only** | Single-user desktop doesn't need concurrency pools or per-user quotas. File size + page count checks (Phase 2) are sufficient. |
| **Process-based parallelism** | PaddleOCR thread-safety unconfirmed; `ProcessPoolExecutor` provides isolation. Batch-mode only; GUI stays single-threaded. |
| **Telemetry & API are optional** | Low value for local desktop. Included as Phase 6 stretch goals. |
| **`structlog` over Prometheus** | Desktop app needs structured file logs, not a metrics server. |
| **Easy-first ordering** | Phases 1–2 are surgical, low-risk edits that build confidence. Architectural work (parallelism, logging overhaul) deferred to Phase 5 after the foundation is solid. |
| **No time estimates** | Phases are ordered by dependency and difficulty. Move to the next phase when the current milestone is green. |

---

## Issue Coverage Map

| Review Issue | Phase | Task # |
|-------------|-------|--------|
| #1 No Error Recovery | Phase 3 | 9, 10 |
| #2 Memory Leaks | Phase 2 | 6 |
| #3 No Resource Throttling | Phase 2 | 7 |
| #4 No Progress Feedback | Phase 3 | 8 |
| #5 Silent Failures | Phase 1 | 1, 2, 3 |
| #6 No Data Validation | Phase 1 | 4 |
| #7 Config Disaster | Phase 4 | 11, 12 |
| #8 SQLite SPOF | Phase 4 | 13 |
| #9 No Observability | Phase 5 | 16 |
| #10 Sequential Processing | Phase 5 | 17 |
| #11 Confusing UX | Phase 4 | 12, 15 |
| #12 No Deployment Docs | Phase 6 | 19 |
| #13 Fragile ML Models | Phase 4 | 14 |
| #14 Testing Gaps | Phase 5 | 18 |
| #15 Hardcoded Paths | Phase 1 + 6 | 5, 20 |
| #16 No Telemetry | Phase 6 | 21 (optional) |
| #17 No API Layer | Phase 6 | 22 (optional) |
