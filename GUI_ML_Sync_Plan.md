# GUI ↔ ML Pipeline Sync & Training Visualization Plan

**Date:** March 5, 2026
**Goal:** Close the gap between the GUI and the ML pipeline APIs, and add visual training progress tracking.

---

## The Problem: Two Parallel Paths

The GUI and the ML pipeline both talk to the same data, but they do it through different doors. Here's the map of what's happening:

### What the GUI does directly (bypassing ML APIs)

| GUI Location | What It Does | ML API That Already Does This |
|---|---|---|
| `model_training.py` → `_on_train_model()` | Calls `store.build_training_set()`, `store.export_training_jsonl()`, creates classifier, calls `.train()`, calls `store.save_training_run()` | `retrain_trigger.auto_retrain()` does all of this plus F1 rollback and drift stats update |
| `model_training.py` → `_on_show_training_history()` | Calls `store.get_training_history()`, formats text table manually | `ExperimentTracker.list_experiments()` returns structured `ExperimentSummary` objects |
| `tab_diagnostics.py` → `_refresh_training_runs()` | Calls `store.get_training_history()`, formats combo box entries | `ExperimentTracker.list_experiments()` |
| `tab_diagnostics.py` → `_compare_runs()` | Calls `store.get_training_history()`, manually computes deltas | `ExperimentTracker.compare_experiments()` returns `ExperimentComparison` |
| `tab_database.py` → `_populate_overview()` | Calls `store.count_corrections_since_last_train()`, `store.should_retrain()` | `retrain_trigger.check_retrain_needed()` |
| `tab_database.py` → Training History section | Calls `store.get_training_history()`, builds label grid | `ExperimentTracker.list_experiments()` |
| `model_training.py` → `_refresh_stats()` | Runs raw SQL (`SELECT COUNT(*) FROM detections`) | `store.get_db_overview()` already does this |

### What the ML pipeline has that the GUI doesn't use at all

| ML API | What It Does | GUI Status |
|---|---|---|
| `retrain_trigger.auto_retrain()` | Full retrain with F1 rollback, drift stats update, experiment recording | Not called — GUI reimplements a simpler version without rollback |
| `retrain_trigger.startup_check()` | Check-and-retrain on app launch | Not called |
| `ExperimentTracker` | Structured experiment listing, comparison, best run, CSV export | Not imported anywhere in GUI |
| `DriftDetector` | Detect when production data deviates from training distribution | Not imported anywhere in GUI |
| `auto_retrain_candidate_classifier()` | Retrain the VOCR candidate hit/miss classifier | Not called |
| `classifier.calibration_curve()` | Compute calibration curves | Called in diagnostics (this one IS wired up) |
| `classifier.predict_negative_probability()` | P(false positive) for a detection | Not called |

---

## The Sync Plan

The principle: the GUI should call the ML pipeline's high-level APIs, not reimplement their logic. When the GUI needs something the API doesn't expose, the fix is to add it to the API — not to write raw SQL in a tkinter callback.

### Change 1: Replace GUI training with `auto_retrain()`

**Current:** `model_training.py` → `_on_train_model()` manually calls `build_training_set`, `export_training_jsonl`, `ElementClassifier().train()`, `save_training_run` in sequence.

**Proposed:** Replace the body of `_on_train_model()` with a call to `auto_retrain()`. This gives the GUI the same F1 rollback protection, drift stats update, and experiment recording that the programmatic path gets.

The GUI thread wrapper stays the same — `auto_retrain()` just runs inside it. The status label updates to reflect whether the retrain was accepted or rolled back.

### Change 2: Wire up `ExperimentTracker` for all history/comparison views

**Current:** Three separate places format training history from raw `store.get_training_history()` dicts.

**Proposed:** Create one `ExperimentTracker` instance (probably on the GUI state object) and use it everywhere:
- `_on_show_training_history()` → `tracker.list_experiments()`
- `_refresh_training_runs()` → `tracker.list_experiments()`
- `_compare_runs()` → `tracker.compare_experiments()`
- Database tab Training History section → `tracker.list_experiments()`

The formatting code in the GUI stays — it just consumes `ExperimentSummary` objects instead of raw dicts. Single source of truth for what "training history" means.

### Change 3: Call `startup_check()` on GUI launch

**Current:** The GUI doesn't check for retrain eligibility on startup.

**Proposed:** In `gui.py` initialization, after the store is opened, call `startup_check(cfg)`. If it returns a `RetrainResult` with `retrained=True`, show a toast/status update. If it returns `rolled_back=True`, warn the user. This is one line of code.

### Change 4: Surface drift warnings in the overlay/annotation tab

**Current:** `DriftDetector` exists, `pipeline.py` computes drift warnings and stores them on `PageResult.drift_warnings`, but the GUI doesn't display them.

**Proposed:** When drift warnings exist on a page result, show a small indicator in the annotation tab (e.g., an amber label "Drift detected on N features"). This tells the user "this page looks different from what the model was trained on — corrections here are extra valuable."

### Change 5: Replace raw SQL in `_refresh_stats()` with `store.get_db_overview()`

**Current:** `model_training.py` runs raw `SELECT COUNT(*)` queries against the connection.

**Proposed:** Use `store.get_db_overview()` which returns all those counts in a dict. Already exists, just not called here.

---

## Training Progress Visualization

You already have matplotlib wired into the diagnostics tab for the calibration reliability diagram. The same pattern works for training progress charts. Here's what would be most useful:

### Chart 1: F1 Over Time (the "is it getting better?" chart)

This is the most important chart. It answers the question "is my model improving as I add corrections?"

**X-axis:** Training run date (or run number)
**Y-axis:** F1 weighted (primary line), F1 macro (secondary line)
**Data source:** `tracker.list_experiments()` → plot `f1_weighted` and `f1_macro` for each run

This shows the learning curve. Early on it'll be noisy. As corrections accumulate, it should trend upward. Plateaus tell you the model has learned what it can from the current feature set — that's when to add new features or enable embeddings.

### Chart 2: Corrections Needed Per Plan Set (the "is it working?" chart)

This is the chart that tells you the self-training loop is succeeding.

**X-axis:** Document (or batch) in chronological order
**Y-axis:** Number of corrections made
**Data source:** `store.get_corrections_for_page()` grouped by document, ordered by `corrected_at`

A downward trend means the model is learning. If it flattens, you've hit the ceiling of the current approach.

### Chart 3: Per-Class F1 Heatmap (the "what's still broken?" chart)

**Rows:** Training runs (most recent at top)
**Columns:** Element types (header, notes, legend, abbreviations, etc.)
**Color:** F1 score (red → yellow → green)
**Data source:** `tracker.list_experiments()` → each run has `per_class` metrics

This shows at a glance which element types the model handles well and which still need work. If "legend" is always red, you know you need more legend corrections.

### Chart 4: Confidence Distribution (the "how sure is it?" chart)

**X-axis:** Confidence bins (0.0-0.1, 0.1-0.2, ..., 0.9-1.0)
**Y-axis:** Count of detections in each bin
**Color:** Correct predictions (green bars) vs incorrect (red bars)
**Data source:** Holdout predictions from the latest training run (already stored in `metrics["holdout_predictions"]`)

This tells you whether the model's confidence scores are trustworthy. If high-confidence predictions are usually right and low-confidence ones are usually wrong, the active learning queue will work well.

### Where to put them

The Diagnostics tab already has the calibration diagram section with matplotlib. Add a new collapsible section called "Training Progress" with Charts 1 and 3 side by side, and Chart 2 below. Chart 4 can go next to the existing calibration diagram since they're related.

All four charts use the same matplotlib → `FigureCanvasTkAgg` pattern that `_draw_reliability_diagram()` already uses. The data is all accessible through `ExperimentTracker` and `CorrectionStore` — no new infrastructure needed.

---

## Implementation Order

1. **Wire up `ExperimentTracker`** — Create the instance on GUI state, replace the three history/comparison call sites. This is the foundation for everything else.

2. **Replace `_on_train_model()` with `auto_retrain()`** — Gets rollback protection for free. Small change, high safety value.

3. **Add Chart 1 (F1 Over Time)** — The most useful single visualization. Uses the experiment data you just wired up.

4. **Add Chart 3 (Per-Class Heatmap)** — Tells you where to focus your correction effort.

5. **Call `startup_check()` on launch** — One line.

6. **Add Chart 2 (Corrections Per Document)** — Motivational chart that shows progress.

7. **Surface drift warnings** — Small UI indicator, connects the drift system to the user.

8. **Add Chart 4 (Confidence Distribution)** — Useful once the model has been trained a few times.

The first two items are cleanup that makes everything else easier. Charts 1 and 3 give you the visibility you need to manage the training process intelligently. The rest is polish.
