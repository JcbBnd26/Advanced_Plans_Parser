# Surya OCR Migration — Outstanding Issues

## Context

PaddleOCR has been replaced with Surya OCR via a new pluggable backend system in `src/plancheck/vocr/backends/`. The backend abstraction layer, Surya implementation, config changes, and consumer files (`extract.py`, `targeted.py`, `engine.py`) are all correctly wired. The subprocess files are deleted. `pyproject.toml` is updated.

**However, the following issues remain and must be fixed before the migration is functional.**

---

## P0 — BLOCKER: Pipeline gate still checks for PaddleOCR

**File:** `src/plancheck/pipeline.py`

**Problem:** The `gate()` function uses `_has_paddleocr()` to decide whether the `vocr` and `reconcile` stages should run. Since PaddleOCR is no longer installed, both checks always return `False`, which means **VOCR and reconcile stages are silently skipped at runtime**. No error is raised — OCR just never runs.

**Lines to fix:**

- **Lines 67–77** — Delete the `_has_paddleocr()` function entirely. It tries to `import paddleocr` and is no longer needed.

- **Line 163** — Inside `gate()`, the `vocr` stage block:
  ```python
  if not _has_paddleocr():
      return False, SkipReason.missing_dependency.value
  ```
  Replace with a check for Surya:
  ```python
  try:
      import surya  # noqa: F401
  except ImportError:
      return False, SkipReason.missing_dependency.value
  ```
  Or remove the dependency check entirely since `get_ocr_backend()` already raises `ImportError` with a clear message if Surya is missing.

- **Line 173** — Inside `gate()`, the `reconcile` stage block has the same `_has_paddleocr()` check. Apply the same fix — reconcile doesn't need PaddleOCR, it needs VOCR tokens to have been produced (which now comes from Surya).

- **Line 25** (docstring) — Says "Visual OCR (VOCR) – PaddleOCR full-page extraction options." Update to reference Surya.

---

## P1 — Tests reference deleted config fields

**File:** `tests/test_config.py`

**Problem:** Multiple tests construct `GroupingConfig` with `vocr_model_tier`, which no longer exists as a config field. These will raise `TypeError` on construction.

**Lines to fix:**

- **Line 25** — `GroupingConfig(enable_vocr=True, vocr_model_tier="server")` → Remove `vocr_model_tier`, optionally replace with `vocr_backend="surya"`
- **Line 29** — `assert cfg2.vocr_model_tier == "server"` → Update assertion to check `vocr_backend`
- **Lines 128–133** — Validation tests for `vocr_model_tier="turbo"` / `"mobile"` / `"server"`. Rewrite to test the new validation: `vocr_backend` must be `"surya"`, and anything else (e.g. `"paddle"`) should raise `ConfigValidationError`.

---

## P1 — Tests reference deleted engine internals

**File:** `tests/vocr/test_vocr_engine.py`

**Problem:** This entire test file imports `_engine_key` and `_ocr_cache` from `plancheck.vocr.engine`, which no longer exist. The engine module now just delegates to `backends.get_ocr_backend()`. Every test in this file references removed config fields (`vocr_model_tier`, `vocr_use_orientation_classify`, `vocr_use_doc_unwarping`, `vocr_use_textline_orientation`).

**Fix:** Rewrite the entire file to test the new backend system:
- Import `get_ocr_backend`, `clear_backend_cache`, and `_backend_key` from `plancheck.vocr.backends.base`
- Test that `_backend_key` produces correct keys from `vocr_backend` and `vocr_device` fields
- Test that `get_ocr_backend()` caches and that `clear_backend_cache()` evicts
- Test that an unknown backend name raises `ValueError`

---

## P1 — Test mocks target wrong import path

**File:** `tests/vocr/test_vocr_extract.py`

**Problem:** Tests at lines 325, 345, 369, 391, 406 use `@patch("plancheck.vocr.engine._get_ocr")` to mock the OCR engine. But `extract.py` now imports `get_ocr_backend` directly from `.backends`, not from `engine`. The mocks don't intercept the actual call path, so tests either try to load real Surya (and fail without GPU/models) or don't test what they think they're testing.

**Fix:** Change all `@patch` decorators from:
```python
@patch("plancheck.vocr.engine._get_ocr")
```
to:
```python
@patch("plancheck.vocr.extract.get_ocr_backend")
```

Also update the `_fake_predict` helper (line 290+). The old helper returns PaddleOCR-format dicts (`dt_polys`, `rec_texts`, `rec_scores`). The new backend returns `List[TextBox]` objects. The mock's `.predict` side effect must return `TextBox` instances:
```python
from plancheck.vocr.backends import TextBox

def _fake_predict(boxes):
    """Return a mock predict function that yields TextBox results."""
    def predict(image):
        return boxes
    return predict
```

---

## P1 — Test mocks target wrong import path (targeted)

**File:** `tests/vocr/test_targeted.py`

**Problem:** Same issue as extract tests. Lines 128, 162, 186, 199 mock `plancheck.vocr.engine._get_ocr`, but `targeted.py` imports `get_ocr_backend` from `.backends`.

**Fix:** Change all `@patch` decorators from:
```python
@patch("plancheck.vocr.engine._get_ocr")
```
to:
```python
@patch("plancheck.vocr.targeted.get_ocr_backend")
```

Ensure the mock's `.predict()` returns `List[TextBox]` objects, not PaddleOCR dicts.

---

## P2 — GUI labels and help text reference PaddleOCR

**File:** `scripts/gui/tab_pipeline.py`

- **Line 211** — `text="VOCR (PaddleOCR extraction)"` → Change to `"VOCR (Surya OCR extraction)"`
- **Line 216** — `text="Full-page PaddleOCR visual token extraction"` → Change to `"Full-page Surya visual token extraction"`

**File:** `scripts/gui/worker.py`

- **Line 41** — Comment says "PaddleOCR/PaddlePaddle messages that should not be shown as ERROR". Review the entire stderr reclassification block (around line 68). PaddleOCR had a known issue of dumping info messages to stderr. Surya does not have this behavior. The filter logic can likely be simplified or removed, but verify Surya's logging behavior first.

**File:** `scripts/runners/run_pdf_batch.py`

- **Line 713** — Help text says "before PaddleOCR" → Change to "before Surya OCR"

**File:** `scripts/runners/run_from_args.py`

- **Line 66** — Help text says "Run PaddleOCR full-page visual token extraction" → Change to "Run Surya full-page visual token extraction"
- **Line 71** — Help text says "before PaddleOCR" → Change to "before Surya OCR"

---

## P2 — Tuning harness references deleted config fields

**File:** `scripts/diagnostics/run_tuning_harness.py`

- **Lines 96–99** — The type map references `vocr_model_tier`, `vocr_use_orientation_classify`, `vocr_use_doc_unwarping`, `vocr_use_textline_orientation`. These fields no longer exist. Replace with the new fields: `vocr_backend` (str), `vocr_device` (str), `surya_languages` (str).

---

## P3 — Obsolete test file at project root

**File:** `test_ocr_mp.py` (project root)

**Problem:** This is a standalone test for PaddleOCR multiprocessing from a thread — the exact problem Surya was adopted to solve. It imports `_get_ocr` from `plancheck.vocr.engine` and tests `ProcessPoolExecutor` behavior. It has no relevance to Surya.

**Fix:** Delete the file entirely.

---

## P3 — Stale PaddleOCR reference in test fixture

**File:** `tests/export/test_page_result_roundtrip.py`

- **Line 559** — Error fixture contains `"message": "PaddleOCR failed"`. This is a test string for error serialization roundtrips. Change to `"Surya OCR failed"` or a generic `"OCR failed"` for accuracy.

---

## Summary Checklist

| # | Priority | File | Action |
|---|----------|------|--------|
| 1 | **P0** | `src/plancheck/pipeline.py` | Replace `_has_paddleocr()` gate checks with Surya check or remove |
| 2 | P1 | `tests/test_config.py` | Remove `vocr_model_tier` references, test `vocr_backend` validation |
| 3 | P1 | `tests/vocr/test_vocr_engine.py` | Rewrite for new backend system |
| 4 | P1 | `tests/vocr/test_vocr_extract.py` | Fix mock paths and mock return types |
| 5 | P1 | `tests/vocr/test_targeted.py` | Fix mock paths and mock return types |
| 6 | P2 | `scripts/gui/tab_pipeline.py` | Update UI labels |
| 7 | P2 | `scripts/gui/worker.py` | Review/simplify stderr filter |
| 8 | P2 | `scripts/runners/run_pdf_batch.py` | Update help text |
| 9 | P2 | `scripts/runners/run_from_args.py` | Update help text |
| 10 | P2 | `scripts/diagnostics/run_tuning_harness.py` | Update config field type map |
| 11 | P3 | `test_ocr_mp.py` | Delete file |
| 12 | P3 | `tests/export/test_page_result_roundtrip.py` | Update error message string |
