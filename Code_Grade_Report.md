# Code Grade Report: Advanced Plans Parser

**Date:** March 4, 2026
**Scope:** Post-refactoring review of full `src/plancheck/` codebase, pipeline architecture, models, config, corrections store, and test coverage.

**Overall Grade: B+** — Strong, with a few things to tighten up.

---

## What's Working Well

### Architecture & Separation

The split into `models.py`, `pipeline.py`, `pipeline_stages.py`, `page_result.py`, and `config.py` is excellent. Each file has a clear single responsibility. The pipeline stages are individually testable, and the `run_stage` context manager with gating logic is a clean pattern that keeps all your runners (GUI, CLI, tests) behaving identically.

### Serialization Hygiene

Every model has a `to_dict()` / `from_dict()` round-trip, and the critical VOCR path uses Pydantic validation schemas. This protects you from subtle data corruption when loading saved runs. That's the kind of thing most people skip and regret later.

### Security & Data

All SQL in `CorrectionStore` uses parameterized queries. The cross-platform file lock (`db_lock.py`) is well-implemented with proper timeout behavior. No SQL injection surface.

### Config

Validation in `__post_init__`, range checks, YAML/TOML/file loaders, and the `GroupingConfig` backward-compat alias all show careful migration thinking.

### Test Coverage

83 test files, ~24k lines of tests. That's proportional to the codebase and covers the major subsystems.

---

## Bugs Found

### 1. `RowBand.bbox()` will crash on empty boxes

**Severity: Medium — Runtime crash under specific conditions**

This is the most concrete bug. Every other `bbox()` method in `models.py` guards against empty inputs and returns `(0, 0, 0, 0)`. `RowBand` doesn't, so `min([])` throws a `ValueError`. Since `populate_rows_from_lines()` can produce RowBands, and lines can theoretically have empty token indices after filtering, this is reachable.

**Location:** `src/plancheck/models.py`, `RowBand.bbox()` (~line 782)

**Fix:** Add an empty guard at the top of `RowBand.bbox()` matching the pattern used everywhere else:

```python
if not self.boxes:
    return (0, 0, 0, 0)
```

### 2. `NotesColumn.to_dict()` — unguarded `.index()` call

**Severity: Low — Potential crash on object identity mismatch**

At the line `header_idx = blocks.index(self.header) if self.header in blocks else None`, the `in` check does a linear identity/equality scan. If something has gone wrong with object identity (e.g., after a deserialize/re-serialize cycle), the `in` check could pass while `.index()` raises `ValueError`. The other region `to_dict()` methods (LegendRegion, AbbreviationRegion, RevisionRegion, etc.) all use a safer try/except pattern. `NotesColumn` should do the same for consistency.

**Location:** `src/plancheck/models.py`, `NotesColumn.to_dict()` (~line 974)

**Fix:** Replace the inline conditional with the try/except pattern used by the other regions:

```python
header_idx = None
if blocks is not None and self.header is not None:
    try:
        header_idx = blocks.index(self.header)
    except ValueError:
        pass
```

---

## Things That Aren't Bugs (Yet), But Could Bite You

### 3. `clustering.py` at ~76K / ~1,600 lines

This is the largest single file by far and handles line building, span splitting, block grouping, column detection, header marking, note marking, notes-column grouping, and continuation linking. If any part of the pipeline is going to develop subtle interaction bugs during future changes, it's here. Consider splitting it into at least two or three modules along those natural boundaries (e.g., line/span building, block grouping, and notes-column logic).

### 4. Broad `except Exception` in pipeline orchestration

There are about 8 instances across `pipeline.py` and `pipeline_stages.py`. These are all logged, which is good, and they're in the right places (GNN, layout model, LLM checks — all optional stages). Just be aware that this can mask unexpected errors during development. You might want to consider narrowing to specific exceptions for the more mature stages.

### 5. Debug logging via f-strings in clustering.py

The `[DEBUG]` writes in the clustering module use f-string formatting unconditionally (writing to a `StringIO` sink when `path=None`). This is fine at current scale, but if clustering ever runs on huge documents, those formatted strings get built even when nobody's reading them. Low priority, just something to know.

### 6. `_do_tocr` scope clarity

In `_run_tocr_vocrpp_stages`, the nested function `_do_tocr` creates a local `page_w_h` tuple while the outer function has separate `page_w` and `page_h` variables. The code works correctly, but the mixed patterns (mutating `boxes[:]` via closure vs. returning `page_w_h` via return value) could trip up a future maintainer. Minor readability concern.

---

## Score Breakdown

| Category | Grade | Notes |
|---|---|---|
| Architecture | A | Clean stage pipeline, great separation |
| Models & Data | A- | One bbox crash bug, otherwise rock solid |
| Config & Validation | A | Comprehensive ranges, multi-format loading |
| Error Handling | B+ | Good patterns, slightly too broad in spots |
| Test Coverage | A- | Proportional and comprehensive |
| Maintainability | B | clustering.py needs decomposition |
| Security | A | Parameterized SQL, safe file locking |

---

## Recommended Action Items

1. **Fix `RowBand.bbox()` empty guard** — Quick win, prevents a runtime crash.
2. **Fix `NotesColumn.to_dict()` try/except** — Quick win, consistency with other regions.
3. **Plan `clustering.py` decomposition** — Bigger investment, biggest long-term payoff.
4. **Narrow broad exception handlers** — As stages mature, tighten the catches.

The two concrete fixes are quick wins. The clustering decomposition is the bigger investment but the biggest long-term payoff. Overall, this is solid refactored code — the kind you can build confidently on top of.
