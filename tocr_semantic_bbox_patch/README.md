# TOCR Semantic Box Building — Patch Notes

**Date:** 2026-02-17  
**Files changed:** 3 (drop-in replacements for `src/plancheck/`)  
**Backward compatible:** Yes — all existing callers and tests pass unchanged.

---

## What This Fixes

The TOCR stage's semantic bbox building had five problems that caused it to miss headers, over-grow or under-grow regions, and fracture note columns on messy/arbitrary plan layouts.

---

## File: `config.py`

**What changed:** Added 7 new fields to `GroupingConfig`. No existing fields touched.

| Field | Default | Purpose |
|-------|---------|---------|
| `region_growth_max_gap` | `0.0` | Max vertical gap (pts) for region growth. **0 = adaptive** (recommended). |
| `region_gap_adaptive_mult` | `3.0` | Adaptive gap = median_line_spacing × this multiplier. |
| `region_growth_x_tolerance` | `80.0` | Horizontal tolerance (pts) for including blocks in a region. |
| `region_font_size_ratio` | `1.8` | Max font-size deviation ratio before excluding a block. 0 = disabled. |
| `header_large_font_mult` | `1.25` | ALL CAPS text at ≥ this × median font size → detected as header. |
| `header_max_rows` | `3` | Max rows in a block to be considered a header candidate. |
| `notes_column_running_mean` | `True` | Use running-mean x0 clustering (robust) vs consecutive-pair (fragile). |

All validated in `__post_init__`.

---

## File: `grouping.py`

### Fix 1: `mark_headers()` — Large-font headers now detected

**Problem:** The acceptance condition was:
```python
if has_colon or (is_all_caps and is_bold):
```
The `is_large` signal (font size > median × 1.3) was computed but **never used in the condition**. Headers like `GENERAL NOTES` in 14pt (no colon, not bold) were silently missed.

**Fix:** Changed to:
```python
if has_colon or (is_all_caps and is_bold) or (is_all_caps and is_large):
```
The multiplier is now configurable via `header_large_font_mult` (default 1.25, slightly more sensitive than the old hardcoded 1.3). Also added `cfg` parameter for configurability.

### Fix 2: `group_notes_columns()` — Running-mean x0 clustering

**Problem:** x0 clustering compared each block's x0 to the **previous block's** x0. A single indented continuation line (shifted 15pt right) would exceed the tolerance and fracture the column into two.

**Fix:** When `notes_column_running_mean=True` (default), each block's x0 is compared to the **cluster's running mean x0**. A single indent is absorbed because the mean barely moves. Only a sustained shift (an actual different column) triggers a split.

---

## File: `_structural_boxes.py`

### Fix 3: `_grow_region_from_anchor()` — Adaptive vertical gap

**Problem:** Hardcoded `max_gap=40.0` pts. Construction plans with loose spacing (40+ pt gaps between notes) would stop region growth prematurely, leaving notes orphaned outside their semantic region.

**Fix:** When `max_gap=0` (new default), the allowed gap is computed adaptively:
```
effective_gap = median(line_spacings_so_far) × adaptive_gap_mult
```
This scales naturally — tight 12pt-spaced notes get a ~36pt threshold, loose 25pt-spaced notes get ~75pt. Clamped to [20, 200] pts. Setting `max_gap > 0` restores the old hard-cap behavior.

### Fix 4: `_grow_region_from_anchor()` — Font-size continuity filter

**Problem:** No glyph-style check during region growth. A 20pt title-block label sitting below an 8pt notes header would get vacuumed into the notes region because it was spatially close.

**Fix:** Each candidate block's average font size is compared to the region's running median font size. If the ratio exceeds `region_font_size_ratio` (default 1.8, i.e. ±80%), the block is skipped. Set to 0 to disable.

### Fix 5: Parameter plumbing

`create_synthetic_regions()`, `_build_semantic_regions()`, and `detect_semantic_regions()` all had their signatures updated to accept and pass through the new adaptive/font-size parameters. Existing callers with no arguments get the new defaults automatically.

---

## Installation

Drop these 3 files into `src/plancheck/`, replacing the originals:

```
src/plancheck/config.py              ← replace
src/plancheck/grouping.py            ← replace
src/plancheck/_structural_boxes.py   ← replace
```

No other files need to change.
