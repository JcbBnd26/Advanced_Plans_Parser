# Alpha Completion Path
## Advanced Plans Parser — GUI-to-ML Learning Loop

**Date:** March 6, 2026  
**Goal:** A system that can successfully start learning from user input via the GUI  
**Philosophy:** "Aim small, Miss small"

---

## Executive Assessment

You have built **74,000 lines of sophisticated infrastructure** — a full PDF parsing pipeline, ML classifiers, GNN architecture, correction storage, experiment tracking, and a Tkinter GUI. The architecture is sound and the individual pieces work.

**The gap:** These pieces don't fully connect yet. The GUI and the ML pipeline talk to the same database through *different doors*. The learning loop exists in code but hasn't been activated end-to-end.

**Alpha Definition:** User opens GUI → loads PDF → sees detections → makes corrections → clicks Train → model improves → next PDF shows better predictions.

That's the full loop. Right now, roughly 70% of the wiring is done. Alpha completion means closing the remaining 30%.

---

## Current State Summary

| Component | Status | Notes |
|-----------|--------|-------|
| PDF Pipeline | ✅ Working | 9 stages, well-tested |
| Detection Storage | ✅ Working | SQLite with proper schema |
| Correction Storage | ✅ Working | Types: relabel, delete, add, accept, reshape |
| Feature Extraction | ✅ Working | 42 features, v6 schema |
| Training Set Builder | ✅ Working | Stratified splits, negative examples |
| Element Classifier | ✅ Working | GradientBoosting with calibration |
| GUI Annotation Tab | ⚠️ Partial | Can make corrections, training button exists |
| GUI ↔ ML Sync | ❌ Gap | GUI reimplements ML logic, bypasses APIs |
| Training Feedback Loop | ❌ Gap | No visual confirmation learning works |
| Cold Start Bootstrap | ❌ Gap | Need 50+ corrections before anything trains |

---

## The Five Milestones to Alpha

### Milestone 1: Fix the Wiring
**Goal:** GUI calls the ML pipeline's APIs instead of reimplementing them  
**Effort:** Small (mostly deletions and redirects)  
**Risk:** Low

**What's wrong now:**  
The `model_training.py` mixin manually calls `store.build_training_set()`, `store.export_training_jsonl()`, creates a classifier, and calls `.train()`. Meanwhile, `retrain_trigger.auto_retrain()` does all of this *plus* F1 rollback protection, drift stats updates, and experiment recording.

**The fix:**  
Replace the body of `_on_train_model()` with a call to `auto_retrain()`. The GUI thread wrapper stays the same — `auto_retrain()` just runs inside it.

**Specific changes:**
1. In `scripts/gui/mixins/model_training.py`, replace lines 68-154 with a clean call to `auto_retrain(threshold=0)` (threshold=0 forces training when user explicitly clicks)
2. Wire up `ExperimentTracker` instance on GUI state object
3. Replace three duplicate history/comparison implementations with tracker calls

**Why this matters:**  
Until this is fixed, the GUI and CLI/programmatic paths behave differently. Training via GUI won't get rollback protection, won't update drift stats, won't record experiments properly.

---

### Milestone 2: Fix the Two Bugs
**Goal:** The ML system behaves correctly when it does run  
**Effort:** Tiny (two one-line fixes)  
**Risk:** Low

**Bug 1: Retrain rollback doesn't actually restore the model**

In `retrain_trigger.py`, when F1 regresses, the code sets `rolled_back = True` and returns — but the new (worse) model has already been saved to disk. The backup exists (`model_path.with_suffix(".pkl.bak")`) but isn't restored.

**Fix location:** `src/plancheck/corrections/retrain_trigger.py`, around line 198-206  
**The fix:** The code that restores from backup exists but is inside a conditional that may not execute. Ensure backup restoration happens whenever `curr_f1 < prev_f1`.

**Bug 2: Feature cache ignores feature version**

In `CorrectionStore`, `cache_features()` stores vectors with a `feature_version` column, but `get_cached_features()` doesn't filter on it. If you upgrade the feature schema, stale vectors get returned.

**Fix location:** `src/plancheck/corrections/feature_cache.py`  
**The fix:** Add `AND feature_version = ?` to the cache lookup query and pass `FEATURE_VERSION` from `classifier.py`.

---

### Milestone 3: Enable Cold-Start Bootstrap
**Goal:** Training can begin without 50 manual corrections  
**Effort:** Small (mechanism already exists, just needs activation)  
**Risk:** Low

**The problem:**  
The classifier needs training data. The `auto_retrain()` threshold is 50 corrections. A new user would need to manually correct 50 detections before the ML system does anything. That's a terrible first-run experience.

**The solution already exists:**  
`training_data.py` has `generate_pseudo_labels()` which creates training examples from high-confidence (≥95%) rule-based detections. This bootstraps the classifier without human annotation.

**The changes:**
1. Add a "Bootstrap Training Data" button to the GUI (or make it automatic on first run)
2. Call `store.generate_pseudo_labels(confidence_threshold=0.95, max_per_label=500)`
3. Then call `auto_retrain(threshold=0)` to train on the pseudo-labels

**Expected result:** User loads their first PDF, clicks "Bootstrap", and now has a working classifier that improves with each correction.

---

### Milestone 4: Close the Visual Feedback Loop
**Goal:** User can SEE that training worked  
**Effort:** Medium (requires one chart implementation)  
**Risk:** Low

**The problem:**  
User makes corrections. User clicks Train. Status says "Model trained — acc 87% F1 84%". User has no idea if that's good, bad, improving, or stuck.

**The minimum viable feedback:**  
One chart: **F1 Over Time**

**Implementation:**
- X-axis: Training run number (or date)
- Y-axis: F1 weighted
- Data source: `ExperimentTracker.list_experiments()` → plot `f1_weighted` for each run

**Where it goes:**  
The Diagnostics tab already has matplotlib wired up for the calibration diagram. Add a "Training Progress" section using the same `FigureCanvasTkAgg` pattern.

**Why this matters:**  
This is the moment when the system becomes *believable*. User makes 10 corrections, trains, sees F1 go from 0.72 to 0.76. Makes 10 more corrections, trains, sees 0.76 to 0.79. That's the "magic" — watching the system learn from their input.

---

### Milestone 5: Call `startup_check()` on GUI Launch
**Goal:** System improves automatically in the background  
**Effort:** Tiny (one line)  
**Risk:** Low

**The change:**  
In `scripts/gui/gui.py`, after the store is opened, call:

```python
from plancheck.corrections.retrain_trigger import startup_check
result = startup_check(cfg)
if result and result.retrained:
    # optionally show toast notification
```

**Why this matters:**  
This is the hands-free learning path. User accumulates 50 corrections over several sessions. Next time they launch the app, it automatically retrains without them clicking anything. The model just *gets better*.

---

## The Alpha Completion Checklist

| # | Task | File(s) | Lines Changed | Prerequisite |
|---|------|---------|---------------|--------------|
| 1 | Replace `_on_train_model()` with `auto_retrain()` call | `scripts/gui/mixins/model_training.py` | ~20 | None |
| 2 | Fix rollback bug (restore backup model) | `src/plancheck/corrections/retrain_trigger.py` | ~3 | None |
| 3 | Fix feature cache version filter | `src/plancheck/corrections/feature_cache.py` | ~2 | None |
| 4 | Add "Bootstrap" button to GUI | `scripts/gui/tab_annotation.py` or `tab_database.py` | ~15 | None |
| 5 | Wire `startup_check()` to GUI launch | `scripts/gui/gui.py` | ~5 | #1 |
| 6 | Create `ExperimentTracker` instance on GUI state | `scripts/gui/gui.py` | ~3 | None |
| 7 | Replace history/comparison calls with tracker | `scripts/gui/mixins/model_training.py`, `tab_diagnostics.py`, `tab_database.py` | ~30 | #6 |
| 8 | Add F1 Over Time chart | `scripts/gui/tab_diagnostics.py` | ~50 | #6, #7 |

**Total estimated changes:** ~130 lines across 7 files

---

## Recommended Order of Implementation

**Phase A: Foundation Fixes (Day 1)**
1. Fix Bug #1 (rollback restoration)
2. Fix Bug #2 (feature cache version)
3. Replace `_on_train_model()` with `auto_retrain()` call

**Phase B: Bootstrap Path (Day 2)**
4. Add Bootstrap button to GUI
5. Test: Fresh install → Bootstrap → Train → Verify model exists

**Phase C: Visibility (Day 3)**
6. Create ExperimentTracker on GUI state
7. Replace duplicate history/comparison code
8. Add F1 Over Time chart

**Phase D: Automation (Day 4)**
9. Wire startup_check() to launch
10. End-to-end test: Make corrections → Close app → Reopen → Verify auto-retrain

---

## Definition of Done: Alpha Complete

The following workflow must succeed without errors:

1. **Cold start:** User launches app for the first time. Clicks "Bootstrap Training Data". Model trains on pseudo-labels. Status shows F1 score.

2. **First corrections:** User loads a PDF. Pipeline runs. Detections appear. User corrects 5 detections (relabel, delete, add). Clicks Train. F1 improves visibly on chart.

3. **Second PDF:** User loads a different PDF. Some detections are now better because the model learned from the first PDF's corrections.

4. **Auto-retrain:** User makes 50 cumulative corrections. Closes app. Reopens app. System auto-retrains in background. Toast notification appears.

5. **Rollback protection:** User trains. F1 regresses for some reason. System automatically rolls back to previous model. Status shows "rolled back" message.

When all five scenarios work, Alpha is complete.

---

## What Alpha is NOT

- **Not performance-optimized.** Memory leaks on large docs, no rate limiting — those are Beta concerns.
- **Not beautiful.** The F1 chart doesn't need to be pretty. It needs to exist.
- **Not comprehensive.** GNN, image embeddings, text embeddings, drift detection — all deferred. The simple GradientBoosting classifier is enough for Alpha.
- **Not deployed.** This is local-only. Server deployment is Beta or later.

Alpha is: **Proof that the learning loop works.**

---

## Post-Alpha: The Path to Beta

Once Alpha is complete and validated, the next priorities are:

1. **Memory management** — Large document processing without OOM
2. **Error recovery** — Checkpoint/resume for failed runs
3. **More charts** — Per-class F1 heatmap, corrections-per-document trend
4. **Drift warnings in GUI** — Surface `PageResult.drift_warnings` to user
5. **Resource limits** — Max file size, max pages, concurrent processing caps

But those are for after you can watch the system learn.

---

## Summary

**Alpha = The learning loop works end-to-end.**

Five milestones. ~130 lines of code changes. Four days of careful work.

The architecture is already solid. The pieces exist. You're not building new infrastructure — you're connecting existing wires and adding one visibility chart.

Aim small, miss small. Get the loop working first. Everything else follows.
