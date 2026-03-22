# PlanCheck ML System — Code Review & Grade

**Project:** Advanced Plans Parser (`plancheck`)
**Scope:** ML learner subsystem — classifier pipeline, training loop, hierarchical routing, drift detection, active learning, experiment tracking, GNN, and supporting modules
**Date:** March 22, 2026

---

## Overall Grade: B+

This is a legitimately well-architected ML system for a desktop application. The architecture is sound, the feature engineering is thoughtful, and the operational concerns (rollback, drift detection, active learning, experiment tracking) show full-lifecycle thinking — not just "train a model and ship it."

The B+ reflects a few structural issues (calibration data leak, unintegrated GNN, no cross-validation) that are fixable and expected at this stage of development. None are "the building is on fire" — they're "here's where to reinforce it next."

---

## Strengths (A-tier)

### Hierarchical Two-Stage Architecture

Stage 1 does coarse family classification. Only when it's confident the element is a "title" does it route to Stage 2 for subtype refinement (page_title, detail_title, section_title, etc). The fallback chain is:

> Stage 1 → Stage 2 → LLM tiebreaker → argmax

Every step has a graceful degradation path. If Stage 2's model file doesn't exist, it falls back to the Stage 1 label. If the LLM isn't available, it falls back to Stage 2's best guess. This is how production ML systems handle cascading uncertainty.

**Key files:** `hierarchical_classifier.py`, `classifier.py`, `subtype_classifier.py`

### Feature Engineering (42 features, 6 categories)

The features aren't generic — they're tuned to classifying elements on architectural/engineering plan sheets:

- **Font metrics** (3): median, max, min font size
- **Text properties** (9): all-caps ratio, bold detection, token/row counts, word-level stats
- **Spatial position** (7): bbox fractions, center position, aspect ratio
- **Content flags** (4): digit detection, colon presence, numbered-list patterns
- **Keyword patterns** (7): regex-matched indicators for notes, headers, legends, abbreviations, revisions, title blocks, details
- **OCR confidence** (2): mean and min token confidence
- **Relational** (5, added in v6): is_below_header, header_distance, sibling_count, column_position, note_number detection

The v6 relational features show the system is evolving based on what's learned from real data.

**Key file:** `features.py`

### Graceful Degradation Everywhere

The image features module (ResNet-18 via timm), text embeddings module (MiniLM-L6-v2), and GNN module all follow the same pattern:

1. Check if the dependency is installed
2. Return zeros if not
3. Let the rest of the pipeline proceed

This means the app works on a bare install and gets smarter as optional packages are added. The feature vector encoding handles this transparently — the classifier adapts to whatever dimensionality it was trained with.

**Key files:** `image_features.py`, `text_embeddings.py`, `gnn/model.py`

### Isotonic Calibration with FrozenEstimator

Raw GBM probabilities are notoriously overconfident. The system wraps the trained model with `CalibratedClassifierCV` using `FrozenEstimator` (sklearn ≥ 1.6), which learns a calibration mapping without cloning or retraining the underlying model. This is a detail most ML engineers get wrong — they accidentally retrain during calibration.

**Key file:** `training_loop.py` (`_calibrate` function)

### Auto-Rollback on F1 Regression

The retrain trigger backs up the existing model before training, compares the new model's weighted F1 to the previous run, and automatically rolls back if performance regressed. This prevents "we shipped a worse model" disasters.

**Key file:** `retrain_trigger.py`

### Self-Contained Experiment Tracking

Everything is stored in the same SQLite database that holds corrections — zero external dependencies, no MLflow server to maintain. Training runs record metrics, hyperparameters, feature versions, holdout predictions, and per-class breakdowns. The tracker supports listing, comparing, ranking, and CSV export.

**Key files:** `experiment_tracker.py`, `training_runs.py`

---

## Issues Found

### 1. Calibration/Evaluation Data Contamination (Medium Severity)

**Location:** `training_loop.py`, lines 179–206

When validation data exists, it's used for **both** isotonic calibration fitting **and** evaluation metric computation. The code acknowledges this with a log warning but proceeds anyway.

**Impact:** Reported F1 and accuracy numbers will look better than real-world performance. Since the auto-rollback decision (`retrain_trigger.py`) compares F1 between runs, optimistic metrics could cause acceptance of a model that should have been rolled back.

**Fix:** Use the existing 3-way split properly. The `training_data.py` already produces a "test" split (10% of data), but it's never used for final evaluation. Route: train split → fit model, val split → fit calibration, test split → compute metrics.

### 2. Negative-Class Confidence Scaling Interacts with Calibration

**Location:** `classifier.py`, line 369

When the model includes a `__negative__` class (trained from deleted detections), the best real-class confidence is scaled by `(1.0 - p_negative)`. This is applied **after** calibration, which effectively un-calibrates the output probabilities.

**Impact:** Low severity in practice, but confidence values for elements near deletion-like regions won't be well-calibrated. This could cause unexpected behavior at threshold boundaries.

**Note:** The intent is correct — suppress confidence on false-positive-like regions. The implementation just needs to be aware that post-calibration scaling breaks the calibration guarantee.

### 3. Feature Vector Truncation Is a Silent Band-Aid

**Location:** Multiple prediction methods in `classifier.py` and `subtype_classifier.py`

Pattern: `if x.shape[1] > self._n_features_in: x = x[:, :self._n_features_in]`

This silently truncates new features when running an older model. A model trained on v4 features will ignore v5 and v6 features without any warning to the user.

**Impact:** Users may not realize they're running a stale model that can't take advantage of newer features. Particularly problematic when v6 relational features (is_below_header, sibling_count) were added specifically to improve classification accuracy.

**Fix:** Log a warning when truncation occurs, and surface it in the GUI so the user knows retraining would unlock new capabilities.

### 4. GNN Is Architecturally Sound but Disconnected

**Location:** `analysis/gnn/model.py`, `analysis/gnn/graph.py`

The GAT (Graph Attention Network) — 2-layer, multi-head attention, early stopping, save/load — is well-implemented. But it's not wired into the main `classify_element` pipeline in `hierarchical_classifier.py`.

**Impact:** The GNN exists as a standalone component that can be trained and tested, but doesn't contribute to production classification. It's a solid foundation waiting to be plugged in.

**Status:** Expected — this is future work, not a bug.

### 5. No Cross-Validation in Main Training Loop

**Location:** `training_loop.py`

The training loop does a single train/val split and trains once. For the data volumes likely in play (hundreds to low thousands of examples), variance from a single split can be significant.

**Impact:** A class with 8 examples might have all 8 in train and 0 in val by luck of the MD5 hash. The stratification logic tries to prevent this, but with small classes it's fragile. Reported metrics may not be representative.

**Fix:** Consider k-fold cross-validation for metric reporting (even if the final model is trained on all data). This becomes more important as the label count grows.

### 6. Ensemble Sample Weights — Verify All Estimators Receive Them

**Location:** `training_loop.py`, line 166

When using `VotingClassifier`, `sample_weight` is passed to `clf.fit()` and sklearn routes it to each sub-estimator. However, `HistGradientBoostingClassifier`, `LGBMClassifier`, and `XGBClassifier` handle sample weights through different internal paths.

**Impact:** If any ensemble member silently ignores the balanced weights, class-imbalance handling becomes inconsistent across the ensemble. This likely works correctly (sklearn's VotingClassifier handles routing), but it's worth explicit verification.

---

## Supporting Components — Assessment

| Component | File(s) | Assessment |
|-----------|---------|------------|
| **Drift Detection** | `drift_detection.py` | Reasonable lightweight approach using p1/p99 percentile bounds. Treats features independently — won't catch correlated drift. Adequate for a monitoring signal. |
| **Active Learning** | `active_learning.py` | Standard uncertainty-based page ranking (1 - max_proba). Clean implementation. Nothing remarkable, nothing wrong. |
| **Feature Cache** | `feature_cache.py` | SQLite-backed cache with version-aware invalidation. Practical performance optimization. |
| **Pseudo-Labels** | `training_data.py` | Cold-start bootstrap from high-confidence (≥0.95) rule-based detections, capped per class. Solid bootstrapping strategy. |
| **Candidate Classifier** | `candidate_classifier.py` | Secondary classifier for hit/miss prediction on candidate regions. Separate from the main element classifier. |
| **Custom Metrics** | `metrics.py` | Lightweight precision/recall/F1/confusion matrix without sklearn dependency. Keeps GUI startup fast. Correct implementation. |
| **Image Features** | `image_features.py` | ResNet-18 via timm, headless pooling → 512-d embedding. Lazy loading, batch support, graceful fallback. |
| **Text Embeddings** | `text_embeddings.py` | MiniLM-L6-v2 via sentence-transformers → 384-d embedding. Lazy loading, batch support, graceful fallback. |
| **GNN** | `gnn/model.py`, `gnn/graph.py` | 2-layer GAT with early stopping. Well-implemented but not yet integrated into the production pipeline. |

---

## Recommended Priority Order

1. **Fix the calibration/evaluation data leak** — Use the test split for metrics. Highest impact on trustworthy model selection.
2. **Add feature-truncation warnings** — Surface when a stale model is ignoring newer features.
3. **Integrate the GNN into the classification pipeline** — The foundation is ready; the wiring is the remaining work.
4. **Add cross-validation for metric reporting** — Especially important as label diversity grows.
5. **Verify ensemble sample-weight propagation** — Quick check, prevents silent class-imbalance bugs.

---

*"Aim small, miss small."*
