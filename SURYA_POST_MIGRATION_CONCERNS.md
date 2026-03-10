# Surya OCR Migration — Post-Migration Concerns

## Context

This document assumes all issues from `SURYA_MIGRATION_ISSUES.md` have been resolved. These are deeper concerns found during a second-pass analysis of how Surya integrates with the existing pipeline architecture, preprocessing, threading model, and config surface.

---

## 1. VOCRPP Grayscale Preprocessing May Hurt Surya Accuracy

**Risk Level:** Medium — potential accuracy regression, not a crash

**Files:**
- `src/plancheck/vocrpp/preprocess.py`
- `src/plancheck/vocr/extract.py` (line 155)
- `src/plancheck/config/subconfigs.py` (line 73: `vocrpp_grayscale: bool = True`)

**Problem:**

The VOCRPP preprocessing pipeline defaults to `vocrpp_grayscale=True`. When enabled (along with CLAHE or binarization), the output is a PIL image in mode `"L"` (single-channel grayscale). Before passing to Surya, `extract.py` line 155 converts non-RGB images to RGB:

```python
if page_image.mode != "RGB":
    page_image = page_image.convert("RGB")
```

This won't crash — but it triplicates the gray channel into a fake RGB image. PaddleOCR handled grayscale-ish input fine because its detection pipeline was designed for it. Surya's transformer-based models were trained on color document images. Feeding them triplicated grayscale may degrade detection and recognition accuracy because:

- Color contrast cues between text and background are lost
- CLAHE + binarization can destroy subtle gradients that Surya's attention layers use
- Surya likely applies its own internal normalization and expects real color data

**Recommended Fix:**

This needs empirical testing, not a code change yet. Run the same set of test pages through the pipeline with these configurations and compare token counts, confidence scores, and symbol detection rates:

1. `enable_ocr_preprocess=True, vocrpp_grayscale=True` (current default)
2. `enable_ocr_preprocess=True, vocrpp_grayscale=False` (color + CLAHE only)
3. `enable_ocr_preprocess=False` (raw color image, no preprocessing)

If option 2 or 3 produces better results, update the default in `subconfigs.py`. If grayscale still helps (some architectural plans are grayscale-scanned), consider making the default conditional or adding documentation about when to toggle it.

---

## 2. Surya Version Pin Has No Upper Bound

**Risk Level:** Medium — future breakage on upgrade

**File:** `pyproject.toml` (line 24)

**Current:**
```toml
ocr = [
    "surya-ocr>=0.6.0",
]
```

**Problem:**

The Surya backend code depends on specific API patterns from Surya 0.6.x:

- Imports `DetectionPredictor` from `surya.detection` and `RecognitionPredictor` from `surya.recognition`
- Calls `self._rec_predictor([pil_image], [self._languages], self._det_predictor)` with positional args
- Reads `page_result.text_lines` and iterates `text_line.bbox`, `text_line.text`, `text_line.confidence`

Surya is actively developed and pre-1.0. Any of these could change in a future minor release:
- The predictor classes could be renamed or moved
- The call signature could change (keyword args, different parameter order)
- Result objects could rename `text_lines` to `lines` or restructure `bbox`

With `>=0.6.0` and no ceiling, a `pip install --upgrade` could silently break the OCR pipeline.

**Recommended Fix:**

Pin to the tested minor version range:

```toml
ocr = [
    "surya-ocr>=0.6.0,<0.7.0",
]
```

Adjust the upper bound to match whichever version you actually test against. When upgrading Surya in the future, update the pin and verify the API contract in `surya.py` still holds.

---

## 3. Backend Cache Is Not Thread-Safe

**Risk Level:** Low today, Medium if parallelism is added later

**File:** `src/plancheck/vocr/backends/base.py` (lines 85–160)

**Problem:**

The `_backend_cache` is a plain `OrderedDict` at module scope with no synchronization:

```python
_backend_cache: OrderedDict[tuple, OCRBackend] = OrderedDict()
```

`get_ocr_backend()` performs a non-atomic read-then-write sequence:

1. Check if `key in _backend_cache` (line 132)
2. If not, create a new backend (line 139–143)
3. Insert into cache (line 148)
4. Evict oldest if over limit (line 151–156)

The current pipeline processes pages sequentially, and the GUI worker runs one pipeline at a time in a single background thread, so this is safe **today**. But:

- The batch runner (`run_pdf_batch.py`) could be extended to process pages in parallel
- The heartbeat wrapper already spawns threads that access the backend
- Future work could add multi-page parallelism

If two threads call `get_ocr_backend()` simultaneously with the same config, both could pass the cache check, both create a new backend (doubling GPU memory usage), and race on insertion/eviction.

**Recommended Fix:**

Add a `threading.Lock` around the cache operations. This is cheap insurance — zero performance cost in the single-threaded case:

```python
import threading

_cache_lock = threading.Lock()
_backend_cache: OrderedDict[tuple, OCRBackend] = OrderedDict()

def get_ocr_backend(cfg=None):
    key = _backend_key(cfg)
    with _cache_lock:
        if key in _backend_cache:
            _backend_cache.move_to_end(key)
            return _backend_cache[key]

    # Create backend outside lock (slow operation)
    backend = _create_backend(key, cfg)

    with _cache_lock:
        # Double-check after acquiring lock
        if key in _backend_cache:
            _backend_cache.move_to_end(key)
            return _backend_cache[key]
        _backend_cache[key] = backend
        while len(_backend_cache) > _MAX_CACHE:
            evicted_key, _ = _backend_cache.popitem(last=False)
            log.info("Evicted OCR backend %s", evicted_key)

    return backend
```

Note: The backend creation itself should happen **outside** the lock to avoid blocking other threads during model loading.

---

## 4. Heartbeat Wrapper Adds Unnecessary Threading Complexity

**Risk Level:** Low — works but adds indirection

**File:** `src/plancheck/vocr/extract.py` (lines 206–234)

**Problem:**

`_run_ocr_with_heartbeat()` spawns a new daemon thread for every OCR call (once per tile, or once for single-pass). It was designed to prevent terminal/CI timeout by printing periodic heartbeat logs during long-running PaddleOCR calls. It also served as a safety valve — if PaddleOCR deadlocked (the whole reason for this migration), the main thread wouldn't hang forever.

With Surya:
- Surya is thread-safe, so the deadlock protection is unnecessary
- Surya inference is generally faster than PaddleOCR, reducing the need for heartbeat logging
- The wrapper creates a thread-within-a-thread pattern (GUI worker thread → heartbeat thread → OCR), which adds stack complexity and makes debugging harder
- Exception handling via `exc_box` with `sys.exc_info()` is fragile — if the thread is interrupted or the GC collects the traceback early, re-raising can fail

**Recommended Fix:**

Replace the heartbeat wrapper with a direct call and optional periodic logging. Two options:

**Option A — Remove entirely (simplest):**

Replace all `_run_ocr_with_heartbeat(func, *args)` calls with direct `func(*args)` calls. If heartbeat logging is still desired for CI, log before and after each tile instead.

**Option B — Make it configurable (conservative):**

Add a config field `vocr_use_heartbeat: bool = False` (default off). When enabled, keep the current wrapper. When disabled (default), call directly. This lets you re-enable it if Surya turns out to be slow on large images.

Either way, the exception re-raise pattern should be simplified. The current `exc_value.with_traceback(exc_tb)` approach can fail if the traceback object has been garbage collected.

---

## 5. `vocr_max_det_skew` Config Field Is Orphaned

**Risk Level:** Low — dead config, no runtime impact

**Files:**
- `src/plancheck/config/subconfigs.py` (line 51)
- `src/plancheck/config/pipeline.py` (line 97)

**Problem:**

`vocr_max_det_skew: float = 0.0` is defined in both config dataclasses but is never read anywhere in the codebase:

```bash
$ grep -rn "vocr_max_det_skew" --include="*.py" src/
src/plancheck/config/subconfigs.py:51:    vocr_max_det_skew: float = 0.0
src/plancheck/config/pipeline.py:97:    vocr_max_det_skew: float = 0.0
```

No code in `extract.py`, `targeted.py`, `surya.py`, or anywhere else references `cfg.vocr_max_det_skew`. This was likely a PaddleOCR-specific knob for filtering detected text boxes by skew angle. Surya handles skew internally through its transformer architecture.

**Recommended Fix:**

Remove the field from both `subconfigs.py` and `pipeline.py`. If a user has persisted config files (JSON/TOML) that include this field, the `GroupingConfig` dataclass will raise `TypeError` on construction. If config is ever loaded from files, add a note to migration docs or add a `__post_init__` filter that pops unknown fields with a deprecation warning.

---

## 6. Targeted OCR Runs One Surya Inference Per Patch (Performance)

**Risk Level:** Medium — correctness is fine, but performance may be poor

**File:** `src/plancheck/vocr/targeted.py` (lines 130–136)

**Problem:**

The targeted VOCR path iterates over each candidate and runs a separate `backend.predict()` per crop:

```python
for cand in candidates:
    crop = _crop_patch(page_image, cand, page_width, page_height)
    if crop is None:
        cand.outcome = "miss"
        continue
    hits = _ocr_crop(backend, crop, min_conf)
```

Each `_ocr_crop` call invokes `backend.predict(crop_array)`, which in turn calls Surya's recognition predictor. For PaddleOCR, per-call overhead was low because inference was a lightweight CNN forward pass. Surya's transformer architecture has significantly higher per-call overhead due to:

- Attention computation scales with image size
- Model warmup and memory allocation per call
- No batching benefit from single-image calls

If a page has 50–100 candidates (the config allows up to `vocr_cand_max_candidates=200`), that's 50–200 separate Surya inference passes. This could be 5–20x slower than necessary.

Surya's API natively supports batching — the predictor accepts a **list** of images:

```python
predictions = self._rec_predictor(
    [pil_image_1, pil_image_2, ...],  # batch of images
    [languages, languages, ...],       # per-image languages
    self._det_predictor,
)
```

**Recommended Fix:**

Add a `predict_batch` method to the `OCRBackend` base class and `SuryaOCRBackend`:

**Step 1 — Add to `base.py`:**

```python
class OCRBackend(ABC):
    @abstractmethod
    def predict(self, image: np.ndarray) -> List[TextBox]:
        ...

    def predict_batch(self, images: List[np.ndarray]) -> List[List[TextBox]]:
        """Run OCR on multiple images. Default: sequential fallback."""
        return [self.predict(img) for img in images]
```

**Step 2 — Override in `surya.py`:**

Implement `predict_batch` to pass all images to Surya in a single call, then split the results back out per-image.

**Step 3 — Refactor `targeted.py`:**

Collect all crops first, call `backend.predict_batch(all_crops)` once, then map results back to candidates.

This is a performance optimization, not a correctness fix. It can be deferred, but it will become important when processing multi-page documents with many candidates.

---

## 7. Stale PaddleX Comment in targeted.py

**Risk Level:** Cosmetic

**File:** `src/plancheck/vocr/targeted.py` (line 56)

**Current:**
```python
# PaddleX text detection expects H×W×C.  If the page image is grayscale
# (mode 'L') or palette-based, np.array(crop) becomes 2D and crashes.
```

**Fix:**

Update the comment to be backend-agnostic:

```python
# OCR backends expect H×W×3 (RGB).  If the page image is grayscale
# (mode 'L') or palette-based, np.array(crop) becomes 2D and crashes.
```

---

## Summary Checklist

| # | Risk | File(s) | Action |
|---|------|---------|--------|
| 1 | Medium | `vocrpp/preprocess.py`, `subconfigs.py` | Test Surya accuracy with/without grayscale preprocessing |
| 2 | Medium | `pyproject.toml` | Add upper bound to surya-ocr version pin |
| 3 | Low→Medium | `backends/base.py` | Add `threading.Lock` to backend cache |
| 4 | Low | `vocr/extract.py` | Simplify or remove heartbeat thread wrapper |
| 5 | Low | `config/subconfigs.py`, `config/pipeline.py` | Remove orphaned `vocr_max_det_skew` field |
| 6 | Medium | `vocr/targeted.py`, `backends/base.py`, `backends/surya.py` | Add batch predict for targeted OCR performance |
| 7 | Cosmetic | `vocr/targeted.py` | Update stale PaddleX comment |
