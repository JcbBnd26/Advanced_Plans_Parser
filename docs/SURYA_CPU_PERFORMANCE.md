# Surya OCR CPU Performance Problem

## Summary

Surya OCR 0.17.1 is **functionally correct** — detection and recognition both
produce valid results. However, running on CPU against full-page architectural
drawings at standard DPI is **prohibitively slow**, consuming multiple GB of RAM
and requiring hours per page.

---

## Environment

| Component       | Version / Value                        |
|-----------------|----------------------------------------|
| Python          | 3.12.10                                |
| surya-ocr       | 0.17.1                                 |
| transformers    | 4.57.6 (downgraded from 5.3.0)         |
| torch           | 2.10.0                                 |
| Device          | CPU (no CUDA GPU available)            |
| OS              | Windows                                |

---

## Issues Resolved

### 1. OpenBLAS Memory Allocation Crash

**Symptom:** `DetectionPredictor(device='cpu')` immediately crashes with:
```
OpenBLAS error: Memory allocation still failed after 10 retries, giving up.
```

**Cause:** Multiple BLAS-linked ML stacks can install conflicting thread pool
defaults in the same Windows environment. That can trigger an OpenBLAS memory
contention crash during predictor initialization.

**Fix:** Set BLAS thread environment variables to 1 before model construction.
This is baked into `SuryaOCRBackend._ensure_initialized()`:
```python
for var in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(var, "1")
```

**Status:** ✅ Fixed. `setdefault` ensures user-specified values are respected.

### 2. Transformers 5.x Incompatibility

**Symptom:** `FoundationPredictor(device='cpu')` crashes with:
```
AttributeError: 'SuryaDecoderConfig' object has no attribute 'pad_token_id'
```

**Cause:** `transformers 5.3.0` removed `pad_token_id` from the default
`PretrainedConfig`. Surya 0.17.1 was built against transformers 4.x.

**Fix:** Downgrade to `transformers==4.57.6`.

**Status:** ✅ Fixed. Pin should be added to `pyproject.toml` (see
[Action Items](#action-items)).

---

## Verified Working

All three predictor classes initialize successfully with the above fixes:

| Predictor             | Init Time  | Notes                              |
|-----------------------|------------|------------------------------------|
| DetectionPredictor    | ~4s        | After models are cached locally    |
| FoundationPredictor   | ~77s       | Large transformer model            |
| RecognitionPredictor  | ~0s        | Thin wrapper around Foundation     |

First-run import takes ~230s due to model download from HuggingFace Hub.
Subsequent runs use the cached models.

---

## The Performance Problem

### Benchmark: Full-Page OCR on Architectural Drawing

**Input:** Page 2 of `IFC Operations Facilities McClain County - Drawings 25_0915.pdf`

| Metric                 | @ 150 DPI         | @ 300 DPI (pipeline default) |
|------------------------|--------------------|------------------------------|
| Image size             | 3301 × 5100 px    | ~6600 × 10200 px             |
| Megapixels             | 16.8 MP            | ~67 MP                       |
| Detection time         | 4 min 6 sec        | 3 min 34 sec                 |
| Text regions found     | 331                | 124 batches (est. 300+ lines)|
| Recognition rate       | ~3 items/min       | ~0.4 batches/min             |
| Est. total time        | 1.5–2 hours        | 2.5–5+ hours                 |
| Peak memory            | ~3.5 GB            | ~2.9 GB                      |

### Why It's Slow

1. **Transformer models on CPU.** Surya uses ViT (Vision Transformer) for
   detection and a seq2seq transformer for recognition. These architectures
   are designed for GPU — on CPU, each forward pass through the model is
   orders of magnitude slower.

2. **Large input images.** Architectural drawings at 150–300 DPI produce
   images of 17–67 megapixels. Surya must tile these into overlapping chunks,
   run detection on each, then merge results. More tiles = more forward passes.

3. **Many text regions.** A typical architectural sheet has 200–400 individual
   text annotations (dimensions, labels, notes, title block entries). Each
   detected text line requires a separate recognition forward pass.

4. **Single-threaded BLAS.** The OpenBLAS fix forces `NUM_THREADS=1`, which
   prevents the crash but also prevents any BLAS-level parallelism. This
   further slows matrix operations in the transformer layers.

---

## How the Pipeline Should Use VOCR

The pipeline has a **targeted VOCR mode** that avoids full-page scans:

```
TOCR (pdfplumber) → Candidate Detection → Targeted VOCR → Reconciliation
```

1. **TOCR** extracts all text from the PDF text layer (instant, no OCR needed)
2. **Candidate detection** runs 5 tiers of anomaly checks on TOCR tokens to
   find areas where text extraction failed (encoding errors, spatial gaps,
   suspicious patterns)
3. **Targeted VOCR** crops small patches (~100–500 px) around each candidate
   and runs OCR only on those patches
4. **Reconciliation** merges VOCR results back into the TOCR token stream

In targeted mode, Surya would process maybe **5–20 small crops** instead of
the entire page, reducing CPU time from hours to seconds/minutes.

**The problem:** The pipeline fell through to **full-page VOCR** because no
candidates were detected on this page (the PDF has a clean text layer). The
full-page fallback is the worst-case path.

---

## Action Items

### Short-Term (Required)

- [ ] **Pin transformers version** in `pyproject.toml`:
  ```toml
  transformers>=4.45.0,<5.0.0
  ```

- [ ] **Remove unused OCR/ML packages** from the environment when possible.
  A leaner environment reduces the chance of BLAS/runtime conflicts and may
  eventually let us relax the `NUM_THREADS=1` workaround.

### Medium-Term (Performance)

- [ ] **Disable full-page VOCR fallback on CPU.** When `vocr_device == "cpu"`,
  only run targeted VOCR on detected candidates. Skip the expensive full-page
  scan. Add a config flag: `vocr_allow_fullpage_cpu = False`.

- [ ] **Reduce OCR resolution for CPU.** When running on CPU, default to
  72–100 DPI instead of 300 DPI. The image becomes ~9–16x smaller. Add a
  config option: `vocr_cpu_resolution = 100`.

- [ ] **Add image downscaling in the backend.** Before passing to Surya,
  optionally resize the image to a maximum dimension (e.g., 2048 px on the
  long side). This caps inference time regardless of input resolution.

### Long-Term (Architecture)

- [ ] **GPU support.** On machines with CUDA, Surya runs 50–100x faster.
  Detection: seconds. Recognition: seconds. The `vocr_device` config already
  supports `"gpu"` — this just needs a machine with a GPU.

- [ ] **Async/subprocess OCR.** Run Surya in a separate process to avoid
  blocking the pipeline and to isolate memory usage. The existing
  `subprocess_client.py` / `subprocess_worker.py` infrastructure can be
  adapted.

- [ ] **Batch targeted patches.** Instead of running OCR on each candidate
  crop individually, batch all crops into a single `predict_batch()` call.
  Surya's batching amortizes the per-image overhead.

---

## Key Files

| File | Purpose |
|------|---------|
| `src/plancheck/vocr/backends/surya.py` | Surya OCR backend (OpenBLAS fix, lazy init, predict/predict_batch) |
| `src/plancheck/vocr/backends/base.py` | OCR backend ABC, TextBox dataclass, `get_ocr_backend()` factory |
| `src/plancheck/vocr/targeted.py` | Targeted VOCR — crops candidate patches, runs OCR on each |
| `src/plancheck/vocr/extract.py` | Full-page VOCR — scans entire page image |
| `src/plancheck/vocr/candidates/api.py` | 5-tier candidate detection from TOCR anomalies |
| `src/plancheck/config/pipeline.py` | Pipeline config (vocr_device, vocr_resolution, etc.) |
| `src/plancheck/config/subconfigs.py` | Sub-configs (VOCRConfig with backend, device, languages) |
