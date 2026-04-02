# Vector Symbol Recovery — Implementation Plan

## Overview

**Problem:** CAD-generated PDFs (AutoCAD, Revit, MicroStation) render certain symbols as vector graphics instead of text characters. The text layer says `2 34` but the page visually shows `2 3/4"` because the slash is a diagonal line path. TOCR reads the text layer perfectly but can't see the graphics layer. VOCR (Surya) can see it but is too heavy for the payoff.

**Solution:** A lightweight post-TOCR pass that matches orphan vector graphics to gaps in the text token stream and injects recovered symbols. No image processing. No OCR model. Just geometry math on coordinates you already extract.

**Why this works:** The vocabulary of vectorized construction symbols is small (~8-10 characters), their geometric signatures are distinctive, and they always appear in predictable spatial relationships to adjacent text. A slash between two numbers is always a short diagonal line. A percent sign after a number is always two small circles with a diagonal line. These patterns are deterministic — no model confidence, no training data needed.

---

## The Data You Already Have

After TOCR runs, you have:

**Text tokens (GlyphBox list):** Every word from the text layer with exact `(x0, y0, x1, y1)` bounding boxes, font info, and text content. `2`, `34`, `"` as separate tokens with their positions.

**Vector graphics (PageContext):**
- `ctx.lines` — line segments with `x0, y0, x1, y1, linewidth, stroking_color`
- `ctx.rects` — rectangles with the same fields plus `non_stroking_color`
- `ctx.curves` — Bézier curves with `pts` (list of control points) plus color/width

The slash sitting between `2` and `34` is one of those `lines` entries. The percent sign after `7` is a combination of `curves` (the two circles) and a `line` (the diagonal stroke). This data is already in memory — it was extracted during ingest and is sitting in `PageContext` waiting to be used.

---

## Symbol Signatures

Each vectorized symbol has a recognizable geometric fingerprint:

### Slash `/`
- **Geometry:** A single diagonal line segment
- **Size:** Roughly the same height as adjacent text (~6-14 pts typically)
- **Angle:** Approximately 50-75° from horizontal (forward lean)
- **Context:** Sits in a gap between two digit-bearing text tokens
- **Example:** `2` [diagonal line] `34"` → `2 3/4"`

### Percent `%`
- **Geometry:** Two small circles (curves with ~4 control points each forming arcs) plus one diagonal line
- **Size:** Bounding box roughly square, similar height to adjacent text
- **Context:** Immediately after a digit-bearing text token, no text token to the right within the symbol's span
- **Example:** `7` [two circles + diagonal] → `7%`

### Degree `°`
- **Geometry:** One small circle (curve forming a closed arc)
- **Size:** Smaller than text height — roughly 30-50% of adjacent font size
- **Position:** Sits at the top-right of the preceding digit (superscript position)
- **Context:** After a digit token, often followed by a text token (like `F` or `C`)

### Hash/Number `#`
- **Geometry:** Two horizontal lines + two diagonal lines (or sometimes four lines in a tic-tac-toe pattern)
- **Size:** Similar height to adjacent text
- **Context:** Before a digit token (rebar notation: `#4`, `#5`)

### Plus-Minus `±`
- **Geometry:** One horizontal line + one vertical line + one more horizontal line below
- **Size:** Similar to adjacent text height
- **Context:** Between/before digit tokens (tolerances)

### Diameter `Ø`
- **Geometry:** A circle (curve) with a diagonal line through it
- **Size:** Similar to adjacent text height
- **Context:** Before or after a digit token (pipe sizes, rebar)

---

## Architecture — Where It Fits

The recovery pass runs **after TOCR and before grouping**. This is critical because:
- After TOCR: we have the text tokens and know where the gaps are
- Before grouping: the recovered symbols need to be in the token stream so they get grouped into the correct semantic regions
- Inside the existing pipeline: this is a token enrichment step, not a new stage

### Pipeline Position

```
ingest → tocr → [VECTOR SYMBOL RECOVERY] → prune/deskew → vocr_candidates → ...
```

The pass takes the TOCR token list and the PageContext graphics, and returns an augmented token list with injected GlyphBox entries for recovered symbols. The injected tokens have `origin="vector_symbol"` so downstream code can distinguish them from text-layer tokens if needed.

### Integration Point

In `pipeline.py`, function `_run_early_stages()`:

```
_run_ingest_stage(pr, ctx, cfg, resolution)
boxes, page_w, page_h = _run_tocr_stage(pr, ctx, cfg)

# NEW: Vector symbol recovery
boxes = _recover_vector_symbols(boxes, ctx, cfg, page_w, page_h)

boxes, skew = _run_prune_deskew(boxes, cfg, page_w, page_h)
```

One function call. The token list goes in, an enriched token list comes out.

---

## Implementation

### New Module: `src/plancheck/tocr/vector_symbols.py`

This is a single focused module. No external dependencies beyond what's already imported.

#### Core Function: `recover_vector_symbols()`

```
Input:
  - tokens: list[GlyphBox]     — TOCR output
  - lines: list[dict]           — from PageContext
  - rects: list[dict]           — from PageContext  
  - curves: list[dict]          — from PageContext
  - page_num: int
  - cfg: GroupingConfig

Output:
  - list[GlyphBox]              — original tokens + injected symbol tokens
```

**Algorithm overview:**

1. **Filter graphics to candidates.** Most lines/curves on a construction plan are drawing geometry (walls, section lines, borders). Symbol-sized graphics are small — typically under 15 pts in both dimensions. First pass: filter to graphics whose bounding box is within a reasonable size range for text-scale symbols.

2. **Build a spatial index of text tokens.** For each digit-bearing token, record its position. We need fast lookup of "what text is near this graphic?"

3. **For each candidate graphic, attempt classification:**
   - Is it a diagonal line between two digit tokens? → Slash candidate
   - Is it a small circle/arc after a digit token? → Degree or part of percent
   - Is it a cluster of small circles + diagonal near a digit? → Percent candidate
   - Does it match none of the above? → Skip (it's drawing geometry, not a symbol)

4. **Validate each candidate against context rules:**
   - Slash: must have digit tokens on both sides within a reasonable gap
   - Percent: must have a digit token to the left, gap size consistent with character width
   - Degree: must be at superscript height relative to the adjacent digit
   - Hash: must have a digit token to the right

5. **Create GlyphBox entries for accepted candidates** and insert them into the token list at the correct position.

#### Classification Functions

Each symbol type gets its own classifier function. These are pure geometry — no ML, no models.

##### `_classify_as_slash(line, digit_tokens, cfg) -> GlyphBox | None`

A line is a slash candidate when:
- Length is between 4 and 20 pts (roughly text-scale)
- Angle from horizontal is between 40° and 80° (forward lean, not flat or vertical)
- A digit-bearing token exists within `char_width * 2` to the left
- A digit-bearing token exists within `char_width * 2` to the right
- The line's vertical span overlaps the digit tokens' vertical span (same text line)
- No existing text token already occupies this position with a `/` character

##### `_classify_as_percent(graphics_cluster, digit_tokens, cfg) -> GlyphBox | None`

A percent sign is typically 3 graphic elements in close proximity:
- Two small closed curves (the circles) — each with a bounding box under ~5 pts
- One diagonal line between them
- All three fit within a bounding box roughly `1.0-1.5x` the height of adjacent text
- A digit-bearing token exists to the left

This one is trickier because it requires clustering nearby graphics before classifying. The approach: for each small circle-like curve, check if there's another circle and a diagonal line within a `1.5 * font_height` radius. If all three are present in the right arrangement, it's a percent.

##### `_classify_as_degree(curve, digit_tokens, cfg) -> GlyphBox | None`

A degree symbol is a single small closed curve (circle/arc):
- Bounding box is small: both width and height under `0.5 * adjacent_font_size`
- Position is at superscript height: its vertical center is above the midpoint of the adjacent digit token
- A digit-bearing token exists to the left within `char_width * 1.5`

##### `_is_small_circle(curve_dict) -> bool`

Helper that checks if a pdfplumber curve dict looks like a circle or arc:
- Has at least 4 control points (Bézier circle approximation uses 4+ points)
- Bounding box is roughly square (aspect ratio between 0.7 and 1.4)
- Bounding box is small (both dimensions under a configurable threshold)

---

## What Makes This Different From the Existing Symbol Injection

The existing `reconcile/symbol_injection.py` solves a *different problem*. It takes VOCR output (where Surya DID see the symbol as pixels) and figures out where to spatially insert that already-recognized symbol into the TOCR token stream. It knows what the symbol IS because Surya read it. The challenge is placement.

The new module solves the *inverse problem*. Nobody has read the symbol. We're looking at raw vector geometry and asking "is this collection of lines and curves a recognizable symbol?" The challenge is identification, not placement — because once we know it's a slash, the placement is obvious (it's wherever the vector graphic is).

The two systems could coexist — vector recovery runs during TOCR, VOCR reconciliation runs later if VOCR is enabled. But with VOCR disabled (your plan), vector recovery becomes the primary symbol recovery mechanism.

---

## Config Fields

Add to `GroupingConfig` in `pipeline.py`:

| Field | Default | Purpose |
|-------|---------|---------|
| `tocr_vector_symbols_enabled` | `True` | Master toggle |
| `tocr_vector_symbol_max_size` | `20.0` | Max bounding box dimension (pts) for a graphic to be considered symbol-scale |
| `tocr_vector_symbol_slash_angle_min` | `40.0` | Minimum angle from horizontal for slash candidates (degrees) |
| `tocr_vector_symbol_slash_angle_max` | `80.0` | Maximum angle from horizontal |
| `tocr_vector_symbol_proximity` | `2.0` | Max gap (in char widths) between a symbol candidate and its anchor digit token |
| `tocr_vector_symbol_circle_max_aspect` | `1.4` | Max aspect ratio for a curve to qualify as "circular" |

These should rarely need tuning. The defaults cover the vast majority of CAD output. But having them in config means edge cases can be fixed per-project without code changes.

---

## Diagnostics

The function should return metadata about what it found so you can see it working:

```python
{
    "vector_symbols_found": 12,
    "by_type": {"slash": 8, "percent": 3, "degree": 1},
    "candidates_rejected": 4,
    "rejection_reasons": {"no_digit_neighbor": 2, "already_in_text": 1, "angle_out_of_range": 1}
}
```

This goes into the `PageResult.stages["tocr"].counts` dict so the GUI can show it and you can watch the recovery working.

---

## Implementation Order

### Step 1: Build the diagnostic pass first (read-only)
Scan the graphics, classify candidates, but DON'T inject anything. Just log what it finds. Run this on a few plan sets and verify it's correctly identifying the vectorized symbols. This is zero-risk — it doesn't touch the token stream.

### Step 2: Slash recovery only
Start with the simplest and most common case. Diagonal line between two digits. Inject the GlyphBox. Verify that `2 34"` becomes `2 3/4"` in the TOCR output. This is the highest-value single change.

### Step 3: Percent and degree
Add the multi-graphic clustering for percent signs and the small-circle detection for degrees. These are more complex but less common than slashes.

### Step 4: Remaining symbols
Hash, plus-minus, diameter. These are less common in typical plan sets but follow the same pattern.

---

## What We're NOT Changing

- **TOCR extraction** — `tocr/extract.py` is untouched. We're enriching its output, not modifying how it reads the text layer.
- **The reconcile module** — `reconcile/symbol_injection.py` stays as-is. If someone re-enables VOCR, it still works.
- **Grouping and analysis** — Downstream code already handles GlyphBox tokens regardless of origin. An `origin="vector_symbol"` token groups and classifies identically to an `origin="text"` token.
- **The pipeline stages** — We're adding one function call in `_run_early_stages()`, not a new stage. No gating changes needed.

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| False positives (drawing geometry classified as symbol) | Medium | Medium | Size filter + digit-neighbor requirement eliminates most false positives. Diagnostic pass (Step 1) validates before injection. |
| Missed symbols (vectorized in unusual ways) | Medium | Low | Different CAD software may draw symbols differently. Config tunables handle edge cases. Log rejections so you can spot patterns. |
| Percent sign clustering misses components | Medium | Low | Percent is the hardest symbol — three components must cluster. If clustering fails, it just doesn't inject (fails safe). |
| Performance impact | Low | Low | Filtering small graphics from the full list is O(n) on a small list. The existing plan has hundreds of lines/curves per page, not millions. Sub-millisecond. |
| Injected tokens confuse downstream grouping | Low | Medium | `origin="vector_symbol"` tag lets any downstream code filter if needed. But GlyphBox is GlyphBox — the grouping logic doesn't care about origin. |

---

## Success Criteria

You'll know this is working when:

1. Open a CAD-generated plan with scale notations like `3/4" = 1'-0"`
2. Run the pipeline with VOCR disabled
3. The TOCR output shows `3/4" = 1'-0"` instead of `34 = 1-0`
4. The diagnostics show `vector_symbols_found: N` with the correct count
5. Slope percentages like `2% MAX` appear correctly instead of `2 MAX`
6. No false positives on structural drawing lines or border geometry
