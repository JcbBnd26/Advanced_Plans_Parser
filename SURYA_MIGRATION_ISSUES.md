# Surya OCR Migration — Status Notes

## Context

The repository now uses the Surya OCR backend in `src/plancheck/vocr/backends/`.
Config, extraction, targeted OCR, and engine compatibility shims have already
been aligned with that backend.

This file is retained as a short status note rather than a live blocker list.
Most of the earlier migration work has already landed, so any future OCR work
should start from the current runtime behavior instead of the historical
migration checklist.

---

## Current State

- The canonical OCR backend contract is Surya-only.
- OCR pipeline stages are driven by the current backend factory and runtime
  configuration.
- Tests and GUI copy should use backend-neutral or Surya-specific language,
  depending on whether the code is validating generic OCR behavior or Surya
  integration specifically.

---

## Remaining Maintenance Themes

1. Keep package metadata in sync with `pyproject.toml` so the published OCR
    dependency surface matches the actual backend.
2. Keep tests aligned with the backend abstraction layer instead of older OCR
    engine internals or removed config fields.
3. Keep user-facing documentation consistent with the current Surya runtime and
    avoid stale migration-era wording.
4. Prefer generic OCR terminology in serialization fixtures and shared GUI log
    handling unless a Surya-specific detail is required for accuracy.

---

## Follow-Up Checklist

- Verify OCR-related package metadata after dependency changes.
- Re-run focused OCR/backend tests after touching extraction, targeted OCR, or
  config validation.
- Regenerate derived documentation artifacts separately when a repository-wide
  documentation refresh is needed.
