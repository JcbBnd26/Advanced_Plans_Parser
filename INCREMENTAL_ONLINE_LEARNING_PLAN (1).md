# Incremental Online Learning — Implementation Plan

## Overview

**Goal:** Transform the ML Trainer workflow from "batch → review all pages → retrain at threshold" into "batch → sequential page-by-page review with live model updates between each page."

**Motto:** Aim small, miss small. One page at a time, each one better than the last.

### The Workflow (Before vs. After)

**Current workflow:**
1. Upload PDF
2. Run full batch pipeline (all pages)
3. Open ML Trainer, review page 1 → correct → next
4. Review page 2 → correct → next (same mistakes repeat)
5. Repeat 60 times, correcting the same patterns over and over
6. After 50 corrections accumulate, retrain triggers
7. Model finally improves, but you already did the work

**New workflow:**
1. Upload PDF
2. Run full batch pipeline (all pages) — **Phase 1: Batch** (unchanged)
3. Open ML Trainer, review page 1 → correct grouping, fix labels
4. Hit Next → **micro-retrain fires** (~1-2s) → page 2 re-predicts with updated model
5. Review page 2 → mostly correct now → light touch-ups → Next
6. By page 4-5, rubber-stamping most detections
7. Optional: re-run batch to validate the model learned — **Phase 3: Validation**

**Why this works:** Construction plans are extremely repetitive by design. The same title block, notes column, legend, and revision block appear on 80% of pages. Fix it once, the model propagates it forward.

---

## Architecture — Three Phases

### Phase 1: Batch (No Changes)

`run_document()` in `pipeline.py` runs exactly as it does today. All pages get the full treatment: ingest → TOCR → VOCR candidates → VOCR → reconcile → grouping → analysis → checks → export.

This is the "survey your traverse" step. You want the full picture before you start refining.

**Why we don't touch this:** The batch pipeline does things that only work at document scale — GNN cross-page refinement, candidate detection patterns, memory-efficient VOCR batching (load Surya once, not per page). All of that stays.

**What this produces:** Every page's detections, features, bounding boxes, and initial ML predictions are stored in the SQLite `corrections.db`. This is the baseline the review phase works from.

### Phase 2: Sequential Refinement (The Core Change)

This is where the new behavior lives. The loop looks like:

```
for each page in document (sequential):
    1. Load existing detections from DB (already there from batch)
    2. Re-predict labels using CURRENT model state (lightweight — features already cached)
    3. Display page with updated predictions
    4. User reviews: fix grouping, accept/correct/reject labels
    5. User hits "Next Page"
    6. Save all corrections to CorrectionStore
    7. Micro-retrain: export JSONL → train GBM → save model → reload classifier
    8. Advance to next page → go to step 1
```

**Critical detail:** Only the NEXT page gets re-predicted after each retrain. Pages further ahead keep their batch predictions until you reach them. This means every page benefits from ALL corrections accumulated before it, not just a premature re-prediction from a baby model.

### Phase 3: Validation Re-Run (Optional, No Code Changes Needed)

After finishing review, the user can re-run the batch pipeline on the same document. This is a "trust but verify" step — it tells you whether the model generalized from your corrections or just memorized them in sequence.

This already works today with no changes. The user just clicks "Run Pipeline" again.

---

## Implementation — File-by-File Breakdown

### Change 1: New Module — `src/plancheck/corrections/micro_retrain.py`

**What it does:** A lightweight retrain function purpose-built for the page-by-page loop. Distinct from `auto_retrain()` which is designed for threshold-triggered bulk retraining.

**Why a new module instead of modifying `retrain_trigger.py`:**
- `auto_retrain()` has rollback logic, threshold checks, and Stage-2 training that we don't want during a live session
- The micro-retrain needs to be fast and unconditional — no threshold, no rollback, just "train on what we have and go"
- Keeps the existing retrain path untouched for startup checks and background retraining

**Key behaviors:**
- Exports full JSONL from the corrections store (ALL corrections, not just the latest — avoids catastrophic forgetting)
- Trains the GBM classifier (same `train_classifier()` from `training_loop.py`)
- Skips rollback check (during active review session)
- Skips Stage-2 subtype training (too slow for inter-page cadence; save for session end)
- Returns metrics for the status bar
- Total target time: < 2 seconds

**Key design decision — why we use ALL corrections, not just the latest:**
If we only trained on corrections from the current session, the model would "forget" everything it learned from previous documents. The full JSONL export from `CorrectionStore.export_training_jsonl()` includes ALL historical corrections. This is what prevents catastrophic forgetting. The function already exists and works correctly — we just call it more frequently.

### Change 2: New Module — `src/plancheck/corrections/page_repredict.py`

**What it does:** Re-runs ML predictions on a single page's detections using the freshly-retrained model. This is the "lightweight re-score" that replaces running the entire pipeline.

**Why this is fast:** The expensive work (TOCR, VOCR, grouping, feature extraction) already happened in Phase 1. All the features are sitting in the SQLite `detections` table. This function:
1. Loads all detections for a specific (doc_id, page) from the DB
2. Loads the freshly-trained classifier
3. Runs `clf.predict()` on each detection's cached features
4. Writes updated labels and confidence scores back to the DB
5. Skips any detections the user already corrected (respects `corrected_det_ids`)

**Why not just call `_apply_ml_feedback()`:** That function in `ml_feedback.py` is close to what we need, but it does too much — it also applies prior corrections by bbox IoU, handles drift detection, sets up image extractors and text embedders. For the inter-page re-prediction, we only need the "re-score with current model" part. A focused function is safer and faster.

**Performance target:** < 500ms for a page with 20 detections. The bottleneck is `clf.predict()` calls, which are microseconds each for a GBM.

### Change 3: Modify `scripts/gui/tab_annotation.py` — Session Mode

**What it does:** Introduces a "Training Session" mode that wraps the existing page-by-page review with the micro-retrain loop.

**UI changes (minimal):**
- New toggle or button: "Start Training Session" / "End Training Session"
- Status bar addition during session: shows model update status between pages
  - Example: `"Model updated: F1 0.84 → 0.88 | +6 corrections | 2.1s"`
- Page navigation behavior changes during an active session (see below)

**Why a session mode vs. always-on:**
- The user might want to browse pages without triggering retrains (reviewing previous work, jumping around)
- Session mode signals intent: "I am actively training the model right now"
- Provides a clean start/end boundary for disabling rollback and deferring Stage-2 training

### Change 4: Modify `scripts/gui/pdf_loader.py` — `_on_next_page()`

**What changes:** During an active training session, the "Next Page" action triggers the micro-retrain + re-predict loop before advancing.

**The new flow inside `_on_next_page()` when session is active:**

```
1. Gather all corrections made on current page (already saved to DB individually)
2. Check: were any corrections actually made? (skip retrain if user just accepted everything)
3. Show status: "Updating model..."
4. Call micro_retrain() in background thread via PipelineWorker
5. On completion: call page_repredict() for next page only
6. Advance page index
7. Call _navigate_to_page() which loads the now-updated detections from DB
8. Show status: "Model updated: F1 0.88 | Page X ready"
```

**Threading consideration:** The micro-retrain must run in the background thread (via existing `PipelineWorker`) so the GUI doesn't freeze. But it should block page advancement — the user shouldn't see the next page until re-prediction is complete. The existing worker's `on_done` callback handles this pattern already.

**Important subtlety:** If the user made ZERO corrections on the current page (just clicked through), skip the retrain entirely. No corrections = no new signal = retrain would produce the identical model. This keeps the cadence fast when the model is already performing well.

### Change 5: Modify `scripts/gui/mixins/model_training.py` — Session Lifecycle

**What it does:** Manages the training session lifecycle: start, end, and the deferred work that happens when a session closes.

**On session start:**
- Record the current model state (F1, accuracy) as the session baseline
- Set a flag that disables rollback checks in `micro_retrain()`
- Reset a session correction counter

**On session end (user clicks "End Training Session" or closes the document):**
- Run Stage-2 subtype training if enough subtype corrections accumulated
- Run a single rollback check: compare current model F1 to session-start baseline
- If F1 regressed (rare, but possible if corrections were inconsistent), offer to restore the pre-session model from backup
- Update drift statistics
- Log session summary: pages reviewed, corrections made, F1 delta

**Why defer Stage-2 to session end:** The subtype classifier (`TitleSubtypeClassifier`) trains on a subset of labels and needs more data to be useful. Running it after every page would often fail the minimum-examples check and waste time. One training at session end is sufficient.

### Change 6: Modify `src/plancheck/corrections/retrain_trigger.py` — Awareness of Sessions

**What changes:** Small modification so that `check_retrain_needed()` and `auto_retrain()` are aware that a session may be in progress.

**The concern:** If the user is mid-session doing page-by-page micro-retrains, we don't want the startup check or a background retrain to fire simultaneously and corrupt the model file. 

**Solution:** A simple lock file or flag in the corrections DB:
- `micro_retrain()` sets `session_active = True` in a metadata table before training
- `auto_retrain()` checks this flag and skips if a session is active
- Session end clears the flag

This is defensive programming — unlikely to cause issues in practice since both run in the same process, but important for robustness.

---

## Correction Weighting Strategy

### Grouping Corrections vs. Label Corrections

Based on our discussion about focusing on semantic grouping over labels, corrections should carry different weight in the training data:

**High-value corrections (grouping changes):**
- User merges two detections → this creates a new training example with corrected features (the merged bbox changes zone, aspect ratio, text density, etc.)
- User splits a detection → creates two new examples, each with correct features
- User reshapes a bbox → corrected spatial features propagate to the model

**Standard corrections (label changes):**
- User relabels a detection → label changes but features stay the same
- This is the bread and butter of training data — and the type the model gets good at fast

**The key insight:** Grouping corrections implicitly improve label predictions because they fix the *features* the model trains on. A mis-grouped notes column might have features that look like a legend (wrong aspect ratio, wrong zone). Fixing the grouping fixes the features, and the model naturally classifies it correctly. This is why "focus on grouping, trust the machine on labels" works.

---

## Performance Budget

Target per-page cadence during a training session:

| Step | Target Time | Notes |
|------|-------------|-------|
| User review + corrections | Variable | The human-speed part |
| Save corrections to DB | < 100ms | Already fast (SQLite WAL mode) |
| Export JSONL | < 500ms | Full history, ~1000 rows typical |
| GBM training | < 1s | 200 estimators, ~1000 rows, sub-second |
| Model save (joblib) | < 200ms | ~2.7 MB pickle file |
| Classifier reload | < 200ms | `_reload_classifiers()` already exists |
| Re-predict next page | < 500ms | Load features from DB + predict |
| **Total machine time** | **< 2.5s** | Feels like a natural page turn |

These are conservative estimates. The current `element_classifier.pkl` is 2.7 MB and the `training_data.jsonl` is 338 KB (~1000 rows). GBM training on this scale is well under 1 second on any modern CPU.

---

## Implementation Order

Aim small, miss small. Each step is independently testable:

### Step 1: `micro_retrain.py`
Build the lightweight retrain function. Test it standalone: call it, verify it produces a valid model, verify it uses full JSONL history. This is pure backend — no GUI changes needed to validate.

### Step 2: `page_repredict.py`
Build the single-page re-prediction function. Test it standalone: run batch pipeline on a test PDF, manually insert a correction, call re-predict, verify labels changed. Again, pure backend.

### Step 3: Wire into `_on_next_page()`
Minimal GUI change: when Next is clicked, call micro_retrain → page_repredict → navigate. No session mode yet — just always do it. Test with a real plan set and verify the "watch it learn" experience works.

### Step 4: Add session mode
Add the Start/End session UI, rollback protection, deferred Stage-2 training, and the session lock. This is polish on top of a working core.

### Step 5: Status bar feedback
Show F1 progression, correction count, timing. This is the "watch it learn in real time" UX that makes the whole thing satisfying to use.

---

## What We're NOT Changing

Explicitly listing what stays untouched, because scope discipline matters:

- **`run_document()` in `pipeline.py`** — The batch pipeline is unchanged. It still processes all pages with the full stage sequence.
- **`_apply_ml_feedback()` in `ml_feedback.py`** — The existing feedback loop still runs during batch processing. The new `page_repredict()` is a parallel path for the incremental case.
- **`auto_retrain()` in `retrain_trigger.py`** — The threshold-based retrain still works for non-session scenarios (startup checks, background retraining). Only change: respects the session-active flag.
- **`training_loop.py`** — The core `train_classifier()` function is called by both `auto_retrain()` and the new `micro_retrain()`. No changes needed to the training logic itself.
- **The corrections database schema** — No schema migrations. The existing tables support everything we need. At most, one metadata row for the session-active flag.
- **VOCR / TOCR / grouping** — These only run during the batch phase. The incremental phase never touches them.

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Catastrophic forgetting (model forgets old patterns) | Low | High | Micro-retrain always uses full JSONL history, never just session data |
| Model overfits to current document style | Medium | Medium | Validation re-run (Phase 3) catches this; also mitigated by growing correction history across documents |
| Retrain takes too long (> 5s) | Low | Medium | GBM on <5000 rows is fast; monitor and cap estimators if needed |
| Race condition with background retrain | Low | High | Session-active flag prevents concurrent retrains |
| User expects labels to be perfect after 2 pages | Medium | Low | Status bar shows F1 progression so expectations are calibrated; tooltips explain the learning curve |
| Corrections DB grows large over many sessions | Low | Low | Existing `purge_all_stale_detections()` handles this; JSONL export already filters to training-relevant data |

---

## Success Criteria

You'll know this is working when:

1. **Pages 1-3:** You're actively correcting 5-10 detections per page (normal for a new document)
2. **Pages 4-6:** You're correcting 1-2 detections per page (the repeating patterns are learned)
3. **Pages 7+:** You're mostly hitting "Accept All" and only fixing genuinely novel elements
4. **Status bar shows:** F1 climbing from ~0.70 to ~0.90+ over the first 10 pages
5. **Total session time:** A 60-page plan set takes 15-20 minutes to review instead of 60+
