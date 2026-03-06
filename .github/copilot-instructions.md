# GitHub Copilot Instructions for Advanced_Plan_Parser

These rules apply to all code in this repository.

## Project Identity

This is `plancheck` — a Python 3.10+ architectural/engineering plan parser with
OCR reconciliation, ML classification, and LLM integration. It processes PDFs of
construction drawings through a 9-stage pipeline:

    ingest → tocr → vocrpp → vocr → reconcile → grouping → analysis → checks → export

The core library lives in `src/plancheck/`. A CustomTkinter GUI lives in
`scripts/gui/`. Tests live in `tests/` and mirror the source layout.

---

## Code Organization Rules

### Single Responsibility & File Size Limits

Each module should have one reason to change. If a file grows past **800 lines**,
check whether it's doing multiple jobs:
- If it **is** doing multiple jobs → **split it** into focused modules
- If it's doing **one job** that's genuinely complex → **leave it alone**

When splitting:
1. Extract cohesive functionality into separate modules
2. Use clear, descriptive module names reflecting their single purpose
3. Maintain backward compatibility by re-exporting from the original module if needed

### Indicators a file may be doing multiple jobs:
- Multiple unrelated class definitions
- Functions that serve different domains (e.g., GUI logic mixed with data processing)
- Imports spanning many unrelated libraries
- Difficulty naming the module with a single clear purpose

---

## Python Style & Conventions

### Language & Runtime
- Python 3.10 minimum. Use 3.10+ syntax freely (match/case, `X | Y` unions, etc.)
- Use `from __future__ import annotations` at the top of every module

### Typing
- Type-hint every public function signature (args and return)
- Use `from typing import TYPE_CHECKING` for import-only types to avoid circular imports
- Prefer `Optional[X]` over `X | None` for consistency with the existing codebase
- Use `Tuple`, `List`, `Dict` from `typing` (the codebase already does this consistently)

### Data Modeling
- **Dataclasses** are the default for domain models (GlyphBox, BlockCluster, etc.)
- **Pydantic BaseModel** is used only for serialization/validation schemas (see `validation/schemas.py`)
- Do not mix the two patterns on the same concept — pick one
- Always use `ConfigDict(extra="ignore")` on Pydantic schemas for forward compatibility

### File & Path Handling
- Always use `pathlib.Path` — never `os.path`
- Use `Path` objects through the whole call chain, not just at boundaries

### Imports
- Standard library → third-party → local, separated by blank lines
- Use absolute imports from `plancheck.*` in the library
- Relative imports (`.config`, `..models`) are fine within the same package

### Naming
- `snake_case` for functions, variables, modules
- `PascalCase` for classes
- Private helpers start with `_` (e.g., `_apply_ml_feedback`, `_bbox_iou`)
- Constants are `UPPER_SNAKE_CASE` (see `config/constants.py`)

---

## Error Handling

- Use specific exception types, not bare `except Exception`
- If a broad catch is truly needed (e.g., callbacks that must not crash the pipeline),
  add `# noqa: BLE001` with a comment explaining why
- Raise domain-specific exceptions from `config/exceptions.py`
  (`ConfigLoadError`, `ConfigValidationError`) rather than generic `ValueError`
- Always log exceptions before re-raising: `log.error("...", exc_info=True)`

---

## Logging

- One logger per module: `log = logging.getLogger(__name__)`
- Use lazy formatting: `log.info("Processed %d pages", count)` — never f-strings in log calls
- Levels: `debug` for internal tracing, `info` for stage-level progress,
  `warning` for recoverable issues, `error` for failures that skip work

---

## Pipeline & Architecture

- Every pipeline stage returns a `StageResult` — never raw dicts or tuples
- Gating logic lives in `pipeline.gate()` — do not add ad-hoc skip logic in stages
- Stage functions must not perform file I/O directly; the caller handles persistence
- Keep GUI code (`scripts/gui/`) completely separate from library code (`src/plancheck/`).
  The GUI imports the library — never the reverse

---

## Config System

- All configuration flows through `PipelineConfig` (aliased as `GroupingConfig` for
  backward compatibility)
- Sub-configs are defined in `config/subconfigs.py` (TOCRConfig, VOCRConfig, etc.)
- Never hardcode thresholds or paths — put them in the config with sensible defaults
- Config migration lives in `config/pipeline.py` via `migrate_config()`

---

## Testing

- Framework: `pytest` with `pytest-cov`, `pytest-timeout`, `pytest-xdist`
- Mark tests: `@pytest.mark.unit`, `@pytest.mark.integration`, `@pytest.mark.slow`, `@pytest.mark.gpu`
- Default timeout is 60 seconds per test
- Mirror the source tree under `tests/` (e.g., `tests/tocr/`, `tests/export/`)
- Use `conftest.py` for shared fixtures within each test directory
- Every new public function should have at least one unit test

---

## Dependencies

- Check existing deps in `pyproject.toml` before adding anything new
- Core deps are minimal by design — heavy ML/OCR libs are in optional groups
  (`[ocr]`, `[vision]`, `[layout]`, `[llm]`, `[gnn]`)
- If a feature only needs a heavy dep, gate the import behind a try/except
  and raise a clear message pointing to the right optional group

---

## LLM Integration

- All LLM calls go through `plancheck.llm.LLMClient` — never call ollama/openai/anthropic directly
- Token usage and cost tracking are handled by `llm/cost.py`
- Respect the budget constraints documented in `docs/LLM_BUDGET.md`

---

## Documentation

- Every module gets a top-level docstring explaining its purpose and where it fits
  in the pipeline (see `pipeline.py` for a good example)
- Public classes and functions get docstrings; private helpers get a one-liner if
  the name isn't self-explanatory
- Keep `docs/` updated when adding new stages or changing architecture

---

## What NOT to Do

- Do not add GUI imports or tkinter references in `src/plancheck/`
- Do not use `os.path` — use `pathlib.Path`
- Do not use bare `except:` or `except Exception:` without justification
- Do not hardcode file paths, thresholds, or model names — use config
- Do not create new top-level scripts without discussing placement first
- Do not use f-strings in logging calls
- Do not add heavyweight deps to the core dependency list — use optional groups
