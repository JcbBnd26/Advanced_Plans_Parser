# OCR Gap-Fill Session 2: Implementation Progress & Current Issues

**Date:** February 7, 2026  
**Status:** ⚠️ OCR detects `%` symbols but runs crash before completion

---

## Executive Summary

We successfully fixed the OCR crop geometry and gap prioritization issues from Session 1. The OCR is now **detecting "3%" and "2%"** in debug output, proving the core algorithm works. However, runs are crashing silently before the filtering and acceptance pipeline completes, preventing any tokens from being recorded.

---

## Changes Implemented This Session

### 1. Unicode Character Cleanup (Encoding Fix)

**Problem:** Runs were crashing with `'charmap' codec can't encode character '\u2192'`

**Solution:** Replaced unicode arrow characters with ASCII equivalents:
```python
# BEFORE (causing Windows encoding crash)
print(f"DEBUG OCR: '{left_tok.text}' gap → OCR found: {ocr_texts}")
logger.debug("OCR extract: %r → %r", res.text, extracted)

# AFTER (Windows-safe)
print(f"DEBUG OCR: '{left_tok.text}' gap -> OCR found: {ocr_texts}")
logger.debug("OCR extract: %r -> %r", res.text, extracted)
```

Also simplified `_VALID_OCR_CHARS` to remove unicode symbols:
```python
# BEFORE (unicode symbols)
_VALID_OCR_CHARS = {
    "%", "/", "°", "±", "×", "÷", "·", "•", "→", "←", "↑", "↓",
    ...
}

# AFTER (ASCII only)
_VALID_OCR_CHARS = {
    "%", "/", "deg", "+-", "x", "div", ".", "*",
    ...
}
```

---

### 2. Line-Height-Based Crop Width

**Problem (from Session 1 analysis):**  
The crop width was set to `gap_width + 2*pad`, which resulted in crops only ~20-25 pixels wide. The `%` symbol is roughly as wide as a character height (~12pt), but a 3pt gap only produced a 3pt + 16pt = 19pt crop — not enough pixels for OCR to recognize the symbol's shape.

**Solution:**  
Crop width now uses the **line height** as a minimum, since `%` symbols are roughly 1:1 aspect ratio:

```python
def _ocr_row_region(row, page_image, page_width, page_height, cfg):
    """
    Crop width uses line height as minimum to give OCR enough signal for
    symbols like % which are wider than the gap they occupy.
    """
    # Compute line height and gap width
    line_height = row.y1 - row.y0
    gap_width = row.x1 - row.x0

    # Crop width: use at least line_height * 1.2 to give OCR enough pixels
    # for symbols like % (which is "two blobs + slash")
    min_crop_width = max(line_height * 1.2, gap_width + 2 * pad)

    # Center the crop on the gap midpoint
    gap_center = (row.x0 + row.x1) / 2
    crop_x0_pt = gap_center - min_crop_width / 2
    crop_x1_pt = gap_center + min_crop_width / 2
```

**Result:** Crops are now ~14pt wide minimum (at 300 DPI → ~58 pixels), giving OCR enough context to recognize `%`.

---

### 3. Digit-Following Gap Prioritization

**Problem (from Session 1 analysis):**  
The gap sorting was "narrowest-first" (favoring ~2-3pt gaps), which wasted the 80-gap budget on random small gaps. The actual `%` symbols follow digits like "8.33" or "2", but those specific gaps weren't being prioritized.

**Solution:**  
Reordered gap sorting to prioritize **digit-following gaps first**, then largest-first within each category:

```python
def gap_priority(g):
    left_tok = tokens[g.left_token_idx]
    # Gaps after digits get priority -1000 (sorted first)
    digit_bonus = -1000 if (left_tok.text and left_tok.text[-1].isdigit()) else 0
    # Within each category, prefer larger gaps (more likely to contain visible symbol)
    gap_width = g.x1 - g.x0
    return (digit_bonus, -gap_width)  # negative width = largest first

scanned_gaps.sort(key=gap_priority)
```

**Result:** The 25 digit-following gaps are now scanned first, and debug output shows OCR is finding the `%` symbols:

```
DEBUG: 25 gaps follow digits (potential % locations)
  '10' | gap=3.5pt | 'FOOT'
  '8.33' | gap=??? | '...'
  '2' | gap=??? | '...'
  ...
```

---

## Evidence That OCR Detection Works

Debug output from test runs shows successful `%` detection:

```
DEBUG OCR: 'TSD-0' gap -> OCR found: ['T']
DEBUG OCR: 'RSF-0' gap -> OCR found: ['0', 'R']
...
DEBUG OCR: '8.33' gap -> OCR found: ['3%']    ← SUCCESS: Found %
DEBUG OCR: '2' gap -> OCR found: ['2%']       ← SUCCESS: Found %
DEBUG OCR: '1' gap -> OCR found: ['1 T']
DEBUG OCR: '21137' gap -> OCR found: ['LL900# :']
```

**Key finding:** OCR is correctly recognizing "3%" and "2%" in the gaps following digits "8.33" and "2". This proves:
1. The crop geometry is now correct
2. The prioritization ensures we scan the right gaps
3. PaddleOCR can detect % symbols

---

## Current Issue: Silent Crash After OCR Detection

### Symptom
Runs exit with code 1 but **no Python traceback is displayed**. The process terminates immediately after the last `DEBUG OCR:` line, before the summary line `DEBUG: OCR loop complete. X accepted from Y gaps` is printed.

### Evidence

Terminal output ends abruptly:
```
DEBUG OCR: '2' gap -> OCR found: ['2%']
DEBUG OCR: '1' gap -> OCR found: ['1 T']
DEBUG OCR: '21137' gap -> OCR found: ['LL900# :']
[process exits with code 1]
```

No manifest.json is written, indicating the crash happens during gap processing, not during post-processing.

### Possible Causes

1. **PaddleOCR/PaddlePaddle native crash**  
   The OCR models are compiled C++/CUDA code. A segfault in native code kills the Python process without traceback. Adding `-X faulthandler` didn't produce additional output.

2. **Memory exhaustion**  
   Although unlikely (we only scan 80 gaps), PaddleOCR does load large models. Could be accumulating tensor objects.

3. **Undiscovered encoding issue**  
   The `%` character itself might trigger an encoding issue somewhere downstream when passed to filtering or logging.

4. **Exception in filtering pipeline**  
   The crash happens after `DEBUG OCR:` prints but before the loop summary. The filtering code for "3%" might be raising an uncaught exception that's swallowed by some error handler.

### What We've Tried

| Attempt | Result |
|---------|--------|
| `-X faulthandler` | No additional output |
| `try/catch` wrapper | No exception message |
| `$LASTEXITCODE` check | Shows exit code 1 |
| Adding `flush=True` to prints | Confirmed output ordering |

---

## Filter Pipeline Analysis

When OCR finds "3%", it goes through these filters:

### Filter 1: Must contain special chars or digits
```python
SPECIAL_CHARS = "%°/±×÷·•"
has_special = any(c in SPECIAL_CHARS for c in text)  # "3%" → True (has %)
```
✅ "3%" passes (contains `%`)

### Filter 2: Containment check
```python
if not _is_contained_in_gap(res, gap, tolerance=0.4):
```
⚠️ Unknown — "3%" bbox likely extends beyond the 3pt gap because it includes the "3" from the adjacent token. Should trigger extraction.

### Filter 2b: Special char extraction
If containment fails:
```python
extracted = _extract_special_chars(text)  # "3%" → "%"
if extracted:
    text = extracted  # Use just "%"
    use_extracted = True
```
✅ Should extract just "%" from "3%"

### Filter 3: Glyph likelihood check
```python
if not use_extracted and not _is_likely_missing_glyph(text):
```
⚠️ Skipped if `use_extracted=True`

### Filter 4: Overlap check
```python
if _overlaps_existing(final_res, tokens, line.token_indices):
```
❓ The extracted "%" should be positioned at gap center, not overlapping existing tokens

---

## Recommended Investigation Steps

### 1. Add granular crash detection
Insert print statements at every filter stage to identify exact crash point:
```python
for res in results:
    print(f"DEBUG: Processing OCR result: {res.text}", flush=True)
    # ... filter 1
    print(f"DEBUG: Passed filter 1", flush=True)
    # etc
```

### 2. Check for numeric issues in bbox calculation
The extracted `%` repositioning math:
```python
gap_center_x = (gap.x0 + gap.x1) / 2
char_half_width = (res.x1 - res.x0) * len(text) / max(1, len(res.text)) / 2
```
If `res.text` is empty, `max(1, 0)` still produces 1, but the width calculation could produce NaN if bboxes are malformed.

### 3. Isolate the crash with targeted test
Create a minimal test that:
1. Runs OCR on just the gap that produces "3%"
2. Manually processes it through each filter
3. Identifies which step crashes

### 4. Check PaddleOCR state
The OCR singleton might be in a bad state after processing many crops. Try:
- Reducing `ocr_max_gaps_per_page` to 25 (just the digit-following gaps)
- Adding explicit garbage collection between OCR calls

---

## Files Modified This Session

| File | Changes |
|------|---------|
| `src/plancheck/ocr_fill.py` | Line-height crop width, digit-priority sorting, unicode cleanup, debug prints |
| `src/plancheck/config.py` | No changes (OCR settings unchanged) |
| `scripts/run_pdf_batch.py` | Reformatted by linter |

---

## Current Configuration

```python
# From GroupingConfig
enable_ocr_fill: bool = True        # Enabled via --ocr-fill flag
ocr_gap_mult: float = 4.0           # Max gap = median_char_width * 4
ocr_min_gap_pts: float = 1.0        # Min gap width
ocr_max_gaps_per_page: int = 80     # Budget cap
ocr_confidence: float = 0.6         # Min confidence
ocr_resolution: int = 300           # Crop DPI  
ocr_pad_pts: float = 8.0            # Vertical padding
```

---

## What Success Looks Like

When the crash is fixed, we expect:
1. Debug output: `DEBUG: OCR loop complete. 2 accepted from 80 gaps`
2. Manifest.json: `"ocr_recovered": 2`
3. Final output: Tokens "%" added after "8.33" and "2"
4. Text reconstruction: "8.33%" instead of "8.33" and "2%" instead of "2"

---

## Next Steps

1. **Identify crash location** by adding granular debug prints in the filter pipeline
2. **Reduce scope** to just the digit-following gaps (set `ocr_max_gaps_per_page: 25`)
3. **Validate extracted bbox** calculation isn't producing invalid coordinates
4. **Consider wrapping OCR call** in try/except to catch native crashes gracefully
