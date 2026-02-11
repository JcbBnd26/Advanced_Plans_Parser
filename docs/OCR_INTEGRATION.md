# OCR Gap-Fill Integration

## Overview

The OCR gap-fill feature is designed to recover missing special characters (%, °, /, ±) from CAD-generated PDFs. These characters are often rendered as vector graphics by SHX fonts rather than embedded in the PDF text layer, causing pdfplumber to miss them during extraction.

---

## Pipeline Order

The processing pipeline runs in the following order:

```
1. PDF Text Extraction (pdfplumber)
   ↓
2. Token Pruning (IoU-based deduplication)
   ↓
3. Row Grouping (build_clusters_v2)
   ↓
4. OCR Gap-Fill (fill_ocr_gaps)  ← NEW
   ↓
5. Block Formation
   ↓
6. Table Detection
   ↓
7. Legend/Abbreviation Extraction
   ↓
8. Overlay Generation
```

**Key Design Decision:** OCR runs AFTER row grouping is established. This allows us to:
- Know which tokens belong to which lines
- Identify gaps BETWEEN consecutive tokens within a line
- Avoid scanning entire rows (which caused performance issues)

---

## Implementation Details

### Location
- **Module:** `src/plancheck/ocr_fill.py`
- **Entry Point:** `fill_ocr_gaps(blocks, tokens, page_image, page_width, page_height, cfg)`
- **Called From:** `scripts/run_pdf_batch.py` line ~230

### Configuration Parameters (in `config.py`)
| Parameter | Default | Description |
|-----------|---------|-------------|
| `ocr_gap_mult` | 4.0 | Gap width threshold = median_char_width × this multiplier |
| `ocr_min_gap_pts` | 1.0 | Minimum gap width in points |
| `ocr_max_gaps_per_page` | 80 | Budget cap on gaps to scan per page |
| `ocr_confidence` | 0.6 | Minimum OCR confidence threshold |
| `ocr_resolution` | 300 | DPI for page image rendering |
| `ocr_pad_pts` | 8.0 | Padding around gap crops in points |

### Gap Detection Algorithm
1. Iterate through all lines in all blocks
2. For each line, sort tokens by x-coordinate
3. Find gaps between consecutive tokens where `gap_width > threshold`
4. Collect gap regions with: `(x0, y0, x1, y1, line_id, page)`
5. Sort gaps by width (narrowest first) and cap at budget limit

### OCR Processing
1. For each gap region, crop the page image with padding
2. Run PaddleOCR on the crop
3. Map OCR bbox coordinates back to PDF points
4. Apply filtering pipeline (see below)
5. Add accepted tokens to the token list

---

## Filtering Pipeline

Each OCR detection goes through multiple filters:

```
OCR Detection
    ↓
Filter 1: Has Special Char or Digit?
    ↓ (reject if no % ° / ± × ÷ · • or digits)
Filter 2: Contained in Gap?
    ↓ (if fails and has special char, try extraction)
Filter 3: Looks Like Missing Glyph?
    ↓ (reject multi-char letter strings like "INE", "BAS")
Filter 4: Overlaps Existing Token?
    ↓ (reject if IoU > 0.15 or >30% coverage)
Accept → Add to token list
```

### Special Character Extraction
When OCR finds a string like `"/ - M"` that fails containment (bbox extends beyond gap), we extract just the special characters:
- `"/ - M"` → `"/"`
- `"2%"` → `"2%"` (if fits) or `"%"` (if extracted)

The extracted character is repositioned to the center of the gap region.

---

## Issues Encountered

### Issue 1: OCR Running at Wrong Stage (RESOLVED)
**Problem:** Original implementation ran OCR BEFORE row grouping, scanning arbitrary rectangular regions that didn't align with actual text lines.

**Symptoms:**
- OCR finding wrong text
- False positives from adjacent content

**Solution:** Moved OCR to run AFTER `build_clusters_v2()` so we can use established line structure.

---

### Issue 2: Full-Row Scanning Caused Massive Slowdown (RESOLVED)
**Problem:** First post-grouping attempt scanned entire row bboxes (8800+ pixels wide).

**Symptoms:**
- PaddleOCR warnings: "Resized image size (8824x163) exceeds max_side_limit of 4000"
- Very slow processing

**Solution:** Switched from full-row scanning to gap-between-tokens scanning. Only scan the small regions between consecutive tokens.

---

### Issue 3: Too Many False Positives (RESOLVED)
**Problem:** OCR detected letter fragments from adjacent tokens like "INE", "BAS", "LA".

**Symptoms:**
- 75 tokens "recovered" but most were garbage
- Text like "INE" (from "LINE"), "BAS" (from "BASE")

**Solution:** Added `_is_likely_missing_glyph()` filter to only accept:
- Single special characters (%, °, /, etc.)
- Short strings containing special chars (1%, 2%, 1/2)
- Reject multi-character letter-only strings

---

### Issue 4: Filter Too Strict - 0 Recoveries (RESOLVED)
**Problem:** After adding glyph filter, nothing passed.

**Symptoms:**
- `ocr_recovered: 0` despite 80 gaps scanned

**Root Cause:** OCR finds `"2%"` but "2" already exists as a PDF token. The overlap check rejected the entire result.

**Solution:** Added special character extraction when containment fails. If OCR finds `"/ - M"` but the bbox extends beyond the gap, extract just `"/"` and position it within the gap.

---

### Issue 5: Overlap Check Using Wrong BBox (RESOLVED)
**Problem:** Overlap check used original OCR bbox (for "/ - M") instead of extracted bbox (for "/").

**Symptoms:**
- Extraction worked: `"/ - M" → "/"`
- But then rejected: `OCR reject (overlap): '/'`

**Solution:** Build the final result with extracted text and repositioned bbox BEFORE running overlap check.

---

### Issue 6: Limited Recovery - Only 1 Token (CURRENT)
**Problem:** After all fixes, only 1 token recovered ("/" symbol).

**Symptoms:**
- `ocr_gaps: 80, ocr_recovered: 1, ocr_tokens_added: 1`
- Missing "%" symbols not being found

**Hypotheses:**
1. **Gap threshold too narrow:** The gap where "%" lives between "2" and next word might be smaller than our 4× median char width threshold
2. **OCR not detecting %:** PaddleOCR may struggle to recognize "%" in small crops
3. **% rendered differently:** The "%" may be rendered as multiple vector paths that OCR sees as noise
4. **This page may not have many missing %:** Need to test on other pages known to have percentage values

**Investigation Needed:**
- Debug what gaps exist near known % locations
- Check if "2%" patterns exist and what OCR sees there
- Try adjusting `ocr_gap_mult` to scan larger gaps

---

## Investigation Methods

### 1. Debug Print Statements
Added print statements in the filtering loop to see:
```
OCR reject (no special): 'INE'
OCR reject (containment): '/ - M' gap=[2052.5,2055.2] ocr=[2044.3,2063.0]
OCR extract: '/ - M' → '/'
OCR accept: '/'
```

### 2. Manifest JSON Analysis
Checked `manifest.json` for metrics:
```json
{
  "ocr_gaps": 80,
  "ocr_recovered": 1,
  "ocr_tokens_added": 1
}
```

### 3. Token Origin Inspection
Queried boxes.json for tokens with `origin: "ocr"`:
```powershell
(Get-Content boxes.json | ConvertFrom-Json) | Where-Object { $_.origin -eq "ocr" }
```

### 4. Gap Geometry Analysis
Examined gap dimensions from debug output:
- Gap `[2052.5, 2055.2]` = 2.7 pts wide
- OCR bbox `[2044.3, 2063.0]` = 18.7 pts wide
- Conclusion: OCR detection extends well beyond the gap → need extraction

---

## Current State

### What's Working
- ✅ OCR runs post-grouping (correct pipeline position)
- ✅ Gap-based scanning (fast, ~80 gaps scanned quickly)
- ✅ Special character extraction (`"/ - M"` → `"/"`)
- ✅ Repositioned bbox within gap
- ✅ No false positives (rejected 99+ letter-fragment detections)
- ✅ 1 valid "/" token recovered

### What Needs Investigation
- ❓ Why aren't "%" symbols being detected?
- ❓ Are there gaps near percentage values?
- ❓ What does OCR see in those gaps?

### Next Steps
1. Add debug logging to show gap locations and what OCR finds
2. Cross-reference with known "%" locations in the PDF
3. Consider adjusting gap threshold or adding targeted "%" detection
4. Test on pages with known percentage values (slopes, grades, etc.)

---

## Code References

### Key Functions

**`fill_ocr_gaps()`** - Main orchestrator
```python
def fill_ocr_gaps(
    blocks: List[BlockCluster],
    tokens: List[GlyphBox],
    page_image: Image.Image,
    page_width: float,
    page_height: float,
    cfg: GroupingConfig,
) -> Tuple[List[GlyphBox], List[RowRegion], List[OcrResult]]:
```

**`_ocr_row_region()`** - Crops and OCRs a gap region
```python
def _ocr_row_region(row: RowRegion, page_image, page_width, page_height, cfg) -> List[OcrResult]:
```

**`_is_likely_missing_glyph()`** - Validates OCR text
```python
def _is_likely_missing_glyph(text: str) -> bool:
    # Returns True for: %, °, /, 1%, 50%, 1/2
    # Returns False for: INE, BAS, LA, etc.
```

**`_extract_special_chars()`** - Extracts just special chars from OCR
```python
def _extract_special_chars(text: str) -> Optional[str]:
    # "/ - M" → "/"
    # "2%" → "2%" or "%"
```

**`_is_contained_in_gap()`** - Checks if OCR bbox fits in gap
```python
def _is_contained_in_gap(res: OcrResult, gap: RowRegion, tolerance: float = 0.4) -> bool:
```

---

## Dependencies

- **PaddleOCR 3.4.0** - OCR engine
- **PaddlePaddle 3.0.0** - ML framework
- **Pillow** - Image manipulation
- **pdfplumber 0.11.0** - PDF text extraction

Environment variable required:
```
PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True
```

---

## Summary

The OCR gap-fill feature is functional but currently has limited effectiveness (1 token recovered). The architecture is sound:
1. Runs at correct pipeline stage (post-grouping)
2. Uses efficient gap-based scanning
3. Has robust filtering to avoid false positives

The main unknown is why "%" symbols aren't being detected. This requires investigation into:
- Whether gaps exist near percentage values
- What OCR is detecting in those regions
- Whether the gap threshold needs adjustment
