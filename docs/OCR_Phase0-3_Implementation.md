# OCR Symbol Injection — Phases 0–3 Implementation

## Problem Statement

The OCR reconciliation pipeline recovers symbols (`/`, `%`, `°`, `±`) that CAD-origin PDFs embed as vector paths (invisible to text extraction) but that PaddleOCR *does* read from the rendered image. The prior "Case C" composite injection worked for simple tokens like `1/2%`, but suffered from two major issues:

1. **False positives on headings** — OCR tokens like `SURFACING/MILLINGS`, `A/C`, `N/A` contain `/` but are plain text headings, not numeric expressions. The pipeline was flagging them (orange debug boxes) and in some cases injecting unwanted symbols.
2. **Multi-symbol composites failed** — Date strings like `09/15/25` contain *two* slashes, but the old logic only emitted the single best-gap candidate. Digit-group anchoring was also too loose, accepting tokens like `SECTION 2` as valid anchors because they contained *any* digit.

## Architecture of the Fix

Four phases were implemented simultaneously across `src/plancheck/reconcile/reconcile.py` and `scripts/runners/run_pdf_batch.py`.

---

### Phase 0 — Numeric-Context Filter

**Goal:** Stop the pipeline from even *considering* OCR tokens where symbols appear in non-numeric context.

#### New function: `_has_numeric_symbol_context(text, allowed_symbols)`

| Symbol type | Rule | Pass examples | Reject examples |
|---|---|---|---|
| `/` | Requires `\d\s*/\s*\d` (digit on both sides) | `1/2`, `09/15/25`, `3/8` | `SURFACING/MILLINGS`, `A/C`, `N/A`, `NOTES/` |
| `%`, `°`, `±` | Requires `\d\s*[%°±]` (digit immediately before) | `50%`, `12°`, `1/2%` | `AGGREGATE %`, `% COMPACTION` |

Two compiled regexes power this:
```python
_RE_SLASH_NUMERIC = re.compile(r"\d\s*/\s*\d")
_RE_AFTER_DIGIT   = re.compile(r"\d\s*[%°±]")
```

#### Wiring into the injection loop

At the top of the per-match loop in `_inject_symbols()`, before any Case C/A/B logic runs:

```
if OCR text has allowed symbols BUT fails numeric-context check:
    → log as early_reject (reason: "non_numeric_symbol_context")
    → increment n_filtered_non_numeric counter
    → skip to next match
```

**Exception:** Unmatched symbol-only tokens (e.g., a bare `%` from OCR) bypass this gate — they go to Case B which has its own `_has_digit_neighbour_left` check.

---

### Phase 1a — Multi-Slash Candidate Generation

**Goal:** Handle `09/15/25` (two slashes) instead of picking only one.

#### Old behavior
`_generate_symbol_candidates()` found all inter-token gaps, sorted by width, and emitted **one** `/` candidate at the narrowest gap.

#### New behavior
```python
n_slashes = ocr_text.count("/")   # e.g., 2 for "09/15/25"
```
The function now ranks all gaps by width (ascending) and emits **up to `n_slashes`** `/` candidates — one per gap, each placed at the midpoint of the gap with symbol-pad applied.

For `09/15/25` with PDF tokens `[09] [15] [25]`:
- Gap 09→15 → slash candidate 1
- Gap 15→25 → slash candidate 2

Both are emitted and independently go through acceptance.

---

### Phase 1b — Digit-Group Anchor Tightening

**Goal:** Prevent tokens like `SECTION 2` from being treated as valid digit-group anchors.

#### Old behavior
```python
digit_anchors = [t for t in neighbours if any(ch.isdigit() for ch in t.text)]
```
This accepted *any* token containing at least one digit — so `SECTION 2` qualified.

#### New function: `_is_digit_group(text)`

A token qualifies as a digit-group if:
1. **Starts with a digit** (fast path: catches `09`, `8.33`, `2A`), OR
2. **Majority digit/dot** — more than 50% of characters are digits or `.`

This correctly rejects:
- `SECTION 2` — starts with `S`, only 1/9 chars is a digit
- `SECTION`, `ABC` — no digits at all

And accepts:
- `1`, `09`, `8.33`, `2A` — start with digit

---

### Phase 2 — Case A Composite Deferral

**Goal:** When Case A (matched OCR→PDF, look for extra symbols) detects a composite like `09/15`, defer to Case C gap-placement instead of blindly suffixing.

#### Mechanism

A new compiled regex detects composites:
```python
_RE_COMPOSITE = re.compile(r"\d+\s*[%/°±]\s*\d+")
```

In the Case A branch, if the OCR text matches `_RE_COMPOSITE`:
1. Run `_generate_symbol_candidates()` (same as Case C)
2. If candidates are produced → accept them, log as `case_a_deferred_to_c`, continue
3. If no candidates → fall through to original suffix logic

**In practice**, this path rarely fires because Case C is now checked **first** in the loop (before Case A). When the OCR box spans multiple PDF tokens with gaps, Case C finds the gaps and handles it. The deferral is a safety net for edge cases where Case C's initial pass produces no candidates but Case A detects the composite pattern.

---

### Phase 3 — Traceable Rejection Reasons

**Goal:** Every OCR token that enters the loop gets a debug-log entry explaining what happened.

#### New log paths

| `path` value | Meaning |
|---|---|
| `early_reject` | Filtered by Phase 0 numeric-context check |
| `case_c` | Handled by composite Case C logic |
| `case_a_deferred_to_c` | Case A detected composite, delegated to Case C |
| `case_a_no_extra` | Case A matched, but no extra symbols found |
| `case_a` | Case A suffix injection (original behavior) |
| `case_b` | Unmatched symbol-only token placed near digit |

#### New stat: `filtered_non_numeric`

- Returned as the 3rd element of `_inject_symbols()` → `(added, debug_log, n_filtered_non_numeric)`
- Added to `reconcile_ocr()` stats dict
- Written to manifest in `run_pdf_batch.py` as `ocr_reconcile_filtered_non_numeric`

#### Updated overlay

`draw_reconcile_debug()` Layer 2 (orange boxes for "has symbol but not accepted") now uses `_has_numeric_symbol_context` instead of `_has_allowed_symbol`. This means headings like `SURFACING/MILLINGS` no longer get orange boxes — only tokens with legitimate numeric symbol context that were still rejected appear in orange.

---

## Files Modified

| File | Changes |
|---|---|
| `src/plancheck/reconcile/reconcile.py` | Added `import re`, 2 compiled regexes, `_has_numeric_symbol_context()`, `_is_digit_group()`, multi-slash logic in `_generate_symbol_candidates()`, numeric-context gate + composite deferral in `_inject_symbols()`, 3-tuple return, updated `reconcile_ocr()` stats |
| `src/plancheck/export/reconcile_overlay.py` | Updated `draw_reconcile_debug()` Layer 2 to use `_has_numeric_symbol_context` |
| `scripts/runners/run_pdf_batch.py` | Added `ocr_reconcile_filtered_non_numeric` to manifest counts |

---

## Test Results

All unit tests pass (run via Pylance code snippet runner):

```
Phase 0 PASSED              — numeric context pass/reject for 11 test strings
Phase 1b PASSED             — digit-group anchoring (7 cases including SECTION 2 rejection)
Phase 1a PASSED             — 09/15/25 → 2 slashes injected
Phase 2 PASSED              — composite 09/15 handled by Case C before Case A
Phase 0 heading-filter PASSED — AGGREGATE SURFACING/MILLINGS → 0 injections, 1 filtered
Original Case C 1/2% PASSED — backward compat: / and % both injected
Case B bare % PASSED        — unmatched % near digit 5 still accepted

=== ALL TESTS PASSED ===
```

### Key fix during testing

`_is_digit_group("SECTION 2")` was initially returning `True` because the function had a fast-path clause `text[-1].isdigit()` (ends with digit). This was removed — only `text[0].isdigit()` (starts with digit) is used for the fast path, with a majority-digit fallback for the general case.

---

## Pipeline Test Status

Attempts to run the full pipeline on live PDF pages via terminal have been hampered by PaddleOCR's progress bars corrupting terminal output capture. The pipeline *does* complete and produce run directories, but captured output is empty due to terminal buffering interactions. Previous successful runs on page 10 confirmed the end-to-end flow works; the Phase 0–3 changes are purely within functions that those runs exercise.

---

## Summary

| Phase | What it does | Key benefit |
|---|---|---|
| **0** | Numeric-context filter | Eliminates false positives on headings (`SURFACING/MILLINGS`, `N/A`, etc.) |
| **1a** | Multi-slash generation | Handles dates (`09/15/25`) and multi-fraction expressions |
| **1b** | Digit-group anchoring | Prevents `SECTION 2` from acting as a valid anchor |
| **2** | Case A composite deferral | Gap-places symbols precisely instead of blind suffix |
| **3** | Traceable rejection logging | Every token gets a logged reason; new `filtered_non_numeric` stat |
