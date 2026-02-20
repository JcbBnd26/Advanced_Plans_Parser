# OCR Reconciliation: Current Status & Investigation Log

**Date:** February 10, 2026
**Pipeline version:** Dual-Source OCR Reconciliation (v2)
**Previous approach:** Gap-based OCR fill (v1) — **removed**

---

## What We Did

### Phase 1: Tear-Down of Old OCR Pipeline (Feb 9)

The original OCR pipeline (`ocr_fill.py`) used a **gap-based approach**: after grouping tokens into lines, it found horizontal gaps between consecutive tokens, cropped tiny image regions at each gap, and ran PaddleOCR on each crop individually. This had several fundamental problems documented across four OCR session logs:

- **Dozens of small PaddleOCR calls per page** → intermittent native crashes after ~25 calls
- **Gap geometry was fragile** — symbols rendered inside or overlapping existing glyph areas were invisible to gap detection
- **Limited recovery** — after extensive tuning across 4 sessions, only 1–2 tokens (% and /) were recovered per page
- **Post-grouping insertion** — OCR tokens were added after `build_clusters_v2()`, so they couldn't participate in natural line/span grouping

**Removed files:**
- `src/plancheck/ocr_fill.py` (725 lines)
- `src/plancheck/ocr_fill_old.py` (479 lines)
- `debug_ocr_dump.py`
- `docs/OCR_SLOT_ROUTING_IMPLEMENTATION.md`
- `docs/OCR_SESSION_3_STATUS.md`
- `docs/OCR_SESSION_4_STATUS.md`
- `docs/OCR_CRASH_DIAGNOSIS.md`

**Removed from config:** 8 OCR gap-fill config fields (`enable_ocr_fill`, `ocr_gap_mult`, `ocr_min_gap_pts`, etc.)
**Removed from CLI:** `--ocr-fill`, `--ocr-debug` flags (old versions)

### Phase 2: Build Dual-Source Reconciliation Pipeline (Feb 9)

Implemented the new architecture as `src/plancheck/ocr_reconcile.py` — a 4-stage pipeline:

1. **Stage 1 — Full-page OCR extraction**: Render the entire page at 300 DPI, run PaddleOCR once, convert all detected bboxes from image pixels back to PDF points. Each result becomes a `GlyphBox(origin="ocr_full")`.

2. **Stage 2 — Spatial alignment**: For each OCR token, find the best-matching PDF token by IoU (≥ 0.5) or center proximity (±3pt x, ±2pt y). Classify as `"iou"`, `"center"`, or `"unmatched"`.

3. **Stage 3 — Symbol-only filtering & injection**:
   - **Case A (matched + extra symbol):** OCR token overlaps a PDF token but contains an extra symbol (e.g., OCR=`"2%"` vs PDF=`"2"`) → inject just the `"%"` as a new token positioned after the digit.
   - **Case B (unmatched symbol):** OCR token has no PDF match but contains an allowed symbol, and a digit token exists within 10pt to its left → inject the symbol.
   - **Whitelist:** Only `% / ° ±` are ever injected (configurable).
   - **PDF-wins rule:** If the symbol location already has a PDF token, OCR is ignored.

4. **Stage 4 — Merge**: Accepted OCR tokens are appended to the main `boxes` list **before** `build_clusters_v2()`, so they participate in natural line grouping.

**New files created:**
- `src/plancheck/vocr/engine.py` — shared PaddleOCR singleton
- `src/plancheck/reconcile/reconcile.py` — full reconciliation pipeline
- `src/plancheck/export/reconcile_overlay.py` — debug overlay

**New config fields:** `enable_ocr_reconcile`, `ocr_reconcile_allowed_symbols`, `ocr_reconcile_resolution`, `ocr_reconcile_confidence`, `ocr_reconcile_iou_threshold`, `ocr_reconcile_center_tol_x/y`, `ocr_reconcile_proximity_pts`, `ocr_reconcile_debug`

**New CLI flags:** `--ocr-full-reconcile`, `--ocr-debug`

---

## What Went Right

1. **Clean removal of old pipeline.** No import errors, no broken references. The non-OCR pipeline runs identically to before.

2. **PaddleOCR 3.x API adaptation.** The initial code used the 2.x `.ocr()` API which no longer exists. We discovered PaddleOCR 3.x uses `.predict()` returning generator objects with `.dt_polys`, `.rec_texts`, `.rec_scores` attributes. Fixed on first test run.

3. **PIL → numpy conversion.** PaddleOCR 3.x requires numpy arrays, not PIL images. Added the conversion in `_extract_ocr_tokens`.

4. **Single OCR call per page.** The pipeline makes exactly 1 PaddleOCR call per page instead of dozens of crops. This eliminates the crash-after-25-calls problem entirely.

5. **Pre-grouping insertion.** OCR tokens are now injected into the token list before `build_clusters_v2()`, so they participate naturally in line building and span splitting. No special post-hoc insertion logic needed.

6. **Debug overlay generated.** The `ocr_reconcile.png` overlay was created successfully, providing visual inspection capability.

7. **Run completed without crashes.** The page 3 run finished cleanly (`page 3: done`), no PaddleOCR segfaults or hangs.

---

## What Went Wrong

### Critical Issue: **OCR returned 0 tokens** (`ocr_reconcile_total: 0`)

The manifest shows:
```json
"ocr_reconcile_accepted": 0,
"ocr_reconcile_total": 0
```

PaddleOCR ran successfully (no errors) but detected **zero text tokens** on the entire page. The console log shows:

```
Resized image size (10200x6601) exceeds max_side_limit of 4000. Resizing to fit within limit.
```

This is a PaddleOCR internal message — it resized our 300 DPI render (10200×6601 px for a 34"×22" sheet) down to fit within its 4000px internal limit. At that downscaled resolution, the text on the CAD sheet is likely too small for PaddleOCR's detector to find.

### Secondary Issues

- **No "OCR reconcile:" stats line was printed.** This confirms the early-exit path in `reconcile_ocr()` where `ocr_tokens` is empty — the function returns before printing stats.
- **PowerShell `2>&1 | Tee-Object` interaction.** PowerShell's error stream handling caused exit code 1 and swallowed some output, making debugging harder. Resolved by redirecting with `*>` instead.
- **First run interrupted.** The echo/wait pattern in the terminal interrupted the long-running OCR process. Required re-running with `*>` file redirection.

---

## Current Theories

### Theory 1: Image too large → PaddleOCR downscales below readable threshold

The CAD sheet is 34" × 22" (2448 × 1584 PDF points). At 300 DPI, that's **10200 × 6601 pixels**. PaddleOCR's `max_side_limit=4000` means it internally resizes to ~4000 × 2590 pixels, which is roughly **113 DPI effective**. CAD text at 8–10pt would be ~10–15 pixels tall at that resolution — borderline for OCR detection.

**Test plan:** Try rendering at a lower DPI (150–200) so the image stays under 4000px on the long side, or configure PaddleOCR to raise `max_side_limit`.

### Theory 2: PaddleOCR's internal resize loses the small symbols entirely

Even if the text detector fires on large text (titles, headers), the tiny special characters (`%`, `/`, `°`) at 8pt may be below the detection threshold after downscaling. The original gap-based approach worked because it cropped tight regions at full 300 DPI, keeping the symbols at readable pixel sizes.

**Test plan:** After fixing the resolution issue, check whether PaddleOCR detects the large text tokens (note titles, note text) even if it misses symbols. If it sees large text but not symbols, we may need a **hybrid approach** — full-page OCR at a resolution that fits, plus targeted high-res crops in regions where symbols are expected.

### Theory 3: PaddleOCR 3.x predict() result format may differ from expectations

The result parsing uses `getattr(page_result, "dt_polys", None)` etc. If the attribute names changed in this version, we'd silently get `None` and skip all results. However, the fact that no error was thrown suggests the attributes exist but the lists are empty.

**Test plan:** Add diagnostic logging to print `type(page_result)`, `dir(page_result)`, and lengths of `dt_polys`/`rec_texts`/`rec_scores` even when they're empty.

### Theory 4: numpy array color format

`page_image` from pdfplumber's `.to_image().original` is RGB, but PaddleOCR may expect BGR (OpenCV convention). If the channels are swapped, detection could fail silently.

**Test plan:** Try `img_array = np.array(page_image)[:, :, ::-1]` (RGB→BGR) or check PaddleOCR's expected input format.

---

## Detective Work Done

| Date | Action | Result |
|------|--------|--------|
| Feb 9 | Removed old `ocr_fill.py` pipeline entirely | Clean removal, no broken imports |
| Feb 9 | Built `ocr_reconcile.py` with 4-stage pipeline | Compiles, imports clean, `--help` shows new flags |
| Feb 9 | First test run — PaddleOCR 3.x API error | `.ocr(cls=False)` fails → switched to `.predict()` API |
| Feb 9 | Second test run — image resize warning, 0 tokens | PaddleOCR ran successfully but found nothing |
| Feb 9 | Checked manifest | Confirmed `ocr_reconcile_total: 0` — no tokens extracted at all |

---

## Planned Next Steps

### Immediate (diagnostic)

1. **Add verbose logging** to `_extract_ocr_tokens()` — print `type(page_result)`, attribute names, and poly/text/score list lengths regardless of whether they're empty. This confirms whether PaddleOCR is finding-but-filtering vs finding-nothing.

2. **Test with a standalone script** — run PaddleOCR directly on the 300 DPI render outside the pipeline to inspect raw output. Dump all detected regions to see if any text is found at all.

3. **Try lower render DPI** — render at 150 DPI so the image is ~5100 × 3300 px, closer to the `max_side_limit=4000`. Or render at 115 DPI (the effective DPI after resize) to avoid the lossy resize step.

4. **Try raising `max_side_limit`** — check if PaddleOCR supports a config to increase this limit, keeping full 300 DPI.

5. **Test RGB vs BGR** — try channel reversal to rule out color format mismatch.

### After diagnostics resolve the 0-token issue

6. **Inspect reconciliation stats** — once OCR finds tokens, review how many contain allowed symbols, how many match PDF tokens, and how many are injected. Compare overlay.

7. **Run on pages 2–5** — broader test across multiple pages with known missing symbols.

8. **Compare boxes.json baseline vs reconcile** — diff to confirm only `origin:"ocr"` tokens are added.

9. **Evaluate hybrid approach** — if full-page OCR finds text but misses tiny symbols, consider a two-pass strategy: full-page for large text validation + targeted crops for symbol recovery.

---

## Architecture Reference

```
PDF Text Layer (pdfplumber)        Full-Page OCR (PaddleOCR)
       │                                    │
       ▼                                    ▼
  List[GlyphBox]                      List[GlyphBox]
  origin="text"                       origin="ocr_full"
       │                                    │
       └──────────┐        ┌────────────────┘
                  ▼        ▼
            Spatial Alignment
           (IoU / center proximity)
                     │
                     ▼
            Symbol-Only Filter
         (whitelist: % / ° ±)
         (context: digit nearby)
                     │
                     ▼
              Inject Tokens
            origin="ocr" (add-only)
                     │
                     ▼
              NMS Re-prune
                     │
                     ▼
           build_clusters_v2()
                     │
                     ▼
          Rest of pipeline unchanged
```

### Key Files

| File | Purpose |
|------|---------|
| `src/plancheck/vocr/engine.py` | Shared PaddleOCR singleton (`_get_ocr()`) |
| `src/plancheck/reconcile/reconcile.py` | 4-stage reconciliation pipeline |
| `src/plancheck/export/reconcile_overlay.py` | Debug overlay rendering |
| `src/plancheck/config.py` | `GroupingConfig` with OCR reconcile fields |
| `scripts/runners/run_pdf_batch.py` | Pipeline entry point, `--ocr-full-reconcile` flag |

### Config Defaults

| Setting | Default | Purpose |
|---------|---------|---------|
| `enable_ocr_reconcile` | `False` | Master switch |
| `ocr_reconcile_allowed_symbols` | `"%/°±"` | Symbol whitelist |
| `ocr_reconcile_resolution` | `300` | Render DPI for OCR image |
| `ocr_reconcile_confidence` | `0.6` | Min PaddleOCR confidence |
| `ocr_reconcile_iou_threshold` | `0.5` | IoU for spatial match |
| `ocr_reconcile_center_tol_x` | `3.0` | Center proximity (pts, x) |
| `ocr_reconcile_center_tol_y` | `2.0` | Center proximity (pts, y) |
| `ocr_reconcile_proximity_pts` | `10.0` | Digit-neighbour distance (Case B) |
| `ocr_reconcile_debug` | `False` | Force debug overlay |
