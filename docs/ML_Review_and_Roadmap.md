# ML Functionality Review & Roadmap

**Date:** March 5, 2026
**Scope:** Full review of the ML stack — element classifier, GNN, candidate classifier, drift detection, feature engineering, image/text embeddings, experiment tracking, and retraining pipeline.

---

## What You've Built

Let me lay out what's here before talking about what's next, because there's more ML infrastructure than most projects at this stage would have:

**5-level VOCR candidate confidence system:** Rule-based triggers (Level 0) → adaptive per-method hit rates (Level 1) → ML candidate classifier for hit/miss filtering (Level 2) → per-PDF-producer stats (Level 3) → GNN-based cross-page priors (Level 4). That's a genuinely sophisticated confidence stack.

**Element classifier** with GradientBoosting, optional ensemble (GBM + HistGBM + optional LightGBM/XGBoost), isotonic calibration, negative-class handling for false positives, and a 37-feature schema versioned across 5 iterations.

**GNN for cross-page refinement** — a 2-layer Graph Attention Network that operates on the document graph to capture relationships a single-page classifier can't see.

**Drift detection** with percentile bounds, per-feature and per-page granularity.

**Auto-retrain pipeline** with threshold-based triggering, F1 regression rollback, experiment tracking, and DB-backed training history.

**Optional multimodal features** — CNN image embeddings (ResNet-18 via timm) and sentence-transformer text embeddings (MiniLM), both with lazy loading and graceful degradation when dependencies aren't installed.

The architecture is good. Everything is optional, everything degrades gracefully, and the training/inference/persistence boundaries are clean. The question now is: where's the leverage?

---

## What's Actually Working vs. What's Scaffolding

Here's the honest assessment. A lot of this ML code is *infrastructure waiting for data*. That's not a criticism — you built the pipes before turning on the water, which is the right order. But it means the priorities should be about getting value from what's already wired up, not adding more layers.

### The element classifier needs corrections to train

The entire `ElementClassifier` → `CorrectionStore` → `build_training_set` → `train` → `auto_retrain` loop is well-built, but it's powered by user corrections. Until enough annotations accumulate (the threshold is 50 corrections before retrain triggers), the classifier doesn't exist. The rule-based pipeline is doing all the real work right now.

**What this means:** The most impactful thing for ML quality isn't a code change — it's making the annotation workflow in the GUI as frictionless as possible. Every correction is a training example. Every deleted false positive is a negative example. The faster a user can correct a detection, the faster the classifier starts earning its keep.

### The VOCR candidate classifier needs outcomes to train

Same story. `CandidateClassifier` trains on `candidate_outcomes` rows, requiring at least 100 (default `min_rows`). These outcomes are only populated after targeted VOCR runs and reconciliation verifies hits/misses. Until a user has processed enough pages with VOCR enabled, this classifier sits idle.

### The GNN is the most ambitious and least likely to pay off soon

The GNN requires `torch` + `torch_geometric` (heavy dependencies), operates at the document level (needs multi-page runs), and refines labels that the single-page pipeline already assigns. For ODOT construction plans where each page type is fairly self-contained, the cross-page context may not change many predictions. The GNN candidate prior (Level 4) adds another layer on top of that.

This isn't wasted work — it's future capability. But if you're deciding where to invest time, this is the lowest-ROI area right now.

---

## Concrete Improvements Worth Making

### 1. Feature Engineering: Add Structural/Relational Features

Your current 37-feature schema is solid for individual block classification, but it's missing features about *relationships between blocks* — which is exactly what distinguishes notes from standalone text in construction plans.

Features to consider adding:

- **`is_below_header`** — binary: is there a header-labeled block directly above within block_gap distance? This is the single strongest signal for notes blocks.
- **`header_distance_pts`** — continuous: vertical distance to the nearest header block above. Notes directly under a header are almost always actual notes.
- **`column_position_index`** — ordinal: which visual column is this block in (0 = leftmost)? Notes columns tend to cluster in specific page positions.
- **`sibling_count`** — how many other blocks share the same header/column? A block with 15 siblings under "GENERAL NOTES" is very different from an isolated block.
- **`starts_with_note_number`** — you already have `has_period_after_num` but a direct "matches `_NOTE_BROAD_RE`" feature would be stronger.

These are cheap to compute (the data is already available in the pipeline) and would give the classifier much better discrimination between notes, headers, and standalone text blocks.

### 2. Training Data Bootstrap: Seed from Rule-Based Detections

Right now the classifier waits for human corrections. But your rule-based pipeline already makes good detections — the whole point of the classifier is to refine them. You could bootstrap initial training data by:

- Running the pipeline on a batch of PDFs with the rule-based engine
- Treating high-confidence rule-based detections as "pseudo-labels" (not corrections, just noisy training data)
- Training an initial classifier on those pseudo-labels
- Using disagreements between rule-based and ML predictions to prioritize which detections to show the user for correction

This gives the classifier a head start without requiring any manual annotation. The quality won't be as good as human-corrected data, but it gets the feedback loop spinning.

### 3. The Retrain Rollback Has a Gap

In `auto_retrain()`, the F1 regression check compares against the previous run's F1:

```python
if curr_f1 < prev_f1:
    result.rolled_back = True
    return result
```

But it doesn't actually *restore the old model file*. It sets `rolled_back = True` and returns, but the new (worse) model has already been written to disk by `train_classifier()`. If the application loads the model after a rollback, it gets the regressed model.

**Fix:** Either restore the model from the snapshot before overwriting, or save the new model to a temporary path and only move it into place after the F1 check passes.

### 4. Calibration Is Training on Training Data

In `training_loop.py`, the isotonic calibration fits on `X_train`:

```python
cal = CalibratedClassifierCV(estimator=clf, method="isotonic", cv=cv_folds)
cal.fit(X_train, y_train)
```

This uses cross-validation internally so it's not pure leakage, but calibrating on the training set gives optimistic calibration curves. The validation set (`X_val`) exists and would be a better choice. Using `cv="prefit"` with `X_val` (like the candidate classifier does) would give more honest calibration.

### 5. Drift Detection Threshold Is Static

The drift detector uses fixed 1st/99th percentile bounds. This works for detecting gross distribution shifts, but construction plans have high natural variance — a drainage plan looks very different from an electrical plan. Feature distributions that are "in range" globally might be completely wrong for a specific plan type.

A more useful approach would be per-document-type or per-producer drift detection, since you already track `producer_id`. The infrastructure for this exists (Level 3 producer stats for VOCR candidates), but drift detection doesn't use it yet.

### 6. Feature Cache Isn't Keyed on Feature Version

In `CorrectionStore`, `cache_features` stores vectors with a `feature_version` column, but the `get_cached_features` query doesn't filter on it. If the feature schema changes (you're on version 5), stale cached vectors from version 4 could be returned and fed into a version-5 model, producing silent misclassification.

**Fix:** Add `AND feature_version = ?` to the cache lookup query.

---

## What to Prioritize (In Order)

1. **Fix the retrain rollback gap** — This is a bug. The worse model gets persisted even when rollback is flagged. Quick fix.

2. **Fix the feature cache version filter** — Also a bug. Stale features could silently corrupt predictions. One line in the SQL query.

3. **Add relational features to the feature schema** — Biggest bang for model quality. The data is already computed in the pipeline; you just need to pass it into `featurize()`.

4. **Bootstrap training data from rule-based detections** — Gets the feedback loop started without waiting for 50 manual corrections.

5. **Calibrate on validation data, not training data** — Small code change, meaningfully better confidence scores.

6. **Per-producer drift detection** — Leverage the producer stats infrastructure you already have.

Items 1-2 are bugs to fix. Items 3-4 are the highest-leverage improvements for model quality. Items 5-6 are refinements that make the existing system more trustworthy.

The GNN, image embeddings, and text embeddings are all well-built and ready for when the data volume justifies them. Don't invest more time there until the element classifier is trained and running on real corrections. Get the simple model working well first, then layer on complexity.

---

## ML Grade: B+

| Area | Grade | Notes |
|---|---|---|
| Architecture | A | Clean separation, graceful degradation, everything optional |
| Feature Engineering | B | Solid base schema, missing relational features |
| Training Pipeline | B+ | Well-structured, two bugs (rollback, calibration) |
| Inference | A- | Batch support, feature version guarding, negative class handling |
| Experiment Tracking | A | Self-contained, DB-backed, CSV export, comparison tools |
| Data Strategy | C+ | Infrastructure is there but no bootstrap path for cold start |
| Drift Detection | B | Working but static thresholds, no per-type granularity |
| GNN | B | Well-implemented but premature for current data volume |

The ML stack is well-engineered infrastructure. The gaps aren't in the code quality — they're in the data strategy and the two specific bugs. Fix those and you have a system that will improve itself as people use it.
