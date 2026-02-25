# Advanced Plans Parser — ML System Assessment & Upgrade Roadmap

## Current System: Where You Stand

Your system sits at the **intermediate tier** of ML maturity — above basic but with significant room to push toward advanced. Here's a breakdown of what you've built and where it scores across the key dimensions:

### What you've done well (Intermediate/Strong areas)

**Pipeline architecture (Strong).** Your 9-stage pipeline (`ingest → tocr ‖ vocrpp → vocr → reconcile → grouping → analysis → checks → export`) is cleanly factored, with a canonical gating system, per-stage timing and error reporting, and parallel execution of TOCR + VOCRPP. This is well above basic — most "basic" ML systems are a single script with no stage isolation or observability.

**Feature engineering (Solid Intermediate).** You have 35+ handcrafted features across three generations (v1 numeric/positional → v2 text-content/keyword patterns → v3 discriminative features like text_density, margin distance, line width variance). The label registry with regex-based keyword patterns is thoughtful. Features are versioned and well-documented.

**Human-in-the-loop feedback (Above Average).** The `CorrectionStore` + `_apply_ml_feedback` loop is the standout architectural decision. You have: a SQLite store for detections/corrections/training examples, prior-correction carry-forward via IoU spatial matching, ML relabelling when the model disagrees with confidence ≥ 0.80, and active learning that ranks pages by uncertainty for efficient annotation. This is a genuine feedback loop — many production systems lack this entirely.

**Model training pipeline (Functional).** GradientBoostingClassifier with balanced class weights, train/val split, confusion matrix, per-class precision/recall/F1, JSONL export, COCO/VOC annotation export, and training run auditing in the database.

**Testing (Strong).** ~1,499 tests covering every module is excellent for a project this size.

### Where it's still basic

**Model complexity.** A single `GradientBoostingClassifier` with 200 trees and `max_depth=3` is a reasonable starting point, but it's limited. It only sees hand-engineered features — it cannot learn spatial relationships, visual patterns, or textual semantics beyond your regex keyword detectors.

**No visual understanding.** The classifier never sees the actual image. It makes decisions from derived features (bbox fractions, font metrics, keyword flags). For plan-sheet analysis — where the visual layout, line weights, hatching, and graphical context carry enormous semantic meaning — this is a major blind spot.

**No language understanding.** Text features are shallow: keyword regex matches, capitalization ratios, word counts. The model has no ability to understand what the text means, which matters enormously for distinguishing, e.g., a revision description from a general note that happens to mention "revision."

**Static rule-based analysis.** Your `analysis/` module (legends, abbreviations, revisions, structural boxes, zoning) is entirely heuristic. These regex and geometry rules are brittle — they work for the plan-sheet formats you've seen but will fail on new layouts.

**No confidence calibration.** The raw `predict_proba` output from GradientBoosting is used directly as confidence, but these probabilities are notoriously poorly calibrated. Your 0.80 threshold for ML relabelling may be too aggressive or too conservative depending on the actual calibration curve.

**No model versioning or A/B testing.** Training runs are recorded in the database, but there's no mechanism to compare model versions on the same data, roll back, or serve multiple models side-by-side.

**Single-page processing.** The ML model operates per-detection in isolation. It doesn't see the full page context, relationships between detections, or cross-page patterns (though your rule-based cross-page checks partially address this).

---

## Complexity Rating: 7/10

> **Updated 2026-02-24** — Phases 1–4 complete. 1,499 tests passing, 3 skipped.

| Dimension | Current Level | Score |
|---|---|---|
| Pipeline architecture | 9-stage + drift-detection pass + feature cache | 8/10 |
| Feature engineering | 51+ features (3 gen tabular + 512-d vision + 384-d embeddings) | 7/10 |
| Model sophistication | Calibrated 3-model ensemble (HistGBM / LightGBM / XGBoost) | 6/10 |
| Visual understanding | CNN region-crop features (ResNet-18 / EfficientNet-B0) | 5/10 |
| Language understanding | Sentence-transformer embeddings + optional LLM checks | 6/10 |
| Feedback loop | Active learning + auto-retrain triggers + drift monitoring | 8/10 |
| MLOps & monitoring | Drift detection, experiment tracking, feature cache, auto-retrain | 7/10 |
| Cross-page reasoning | Rules + Graph Attention Network | 5/10 |

---

## Upgrade Roadmap: From Intermediate to Sophisticated

The following upgrades are ordered by **impact-to-effort ratio** — highest bang-for-buck first. Each one is designed to improve accuracy and automation while keeping the system usable (preserving your GUI annotation workflow, your pipeline architecture, and your human-in-the-loop paradigm).

### Phase 1: Quick Wins (1–2 weeks each) ✅ Completed 2026-02-20

#### 1.1 — Confidence Calibration ✅

**Problem:** GBM probabilities don't reflect true accuracy. Your 0.80 threshold is arbitrary.

**Solution:** Add isotonic regression or Platt scaling as a post-hoc calibration step after training. Scikit-learn provides `CalibratedClassifierCV` for this.

**Implementation:**
- After training the GBM, wrap it with `CalibratedClassifierCV(method='isotonic', cv=5)`
- Store the calibrated model alongside the raw one
- Add a reliability diagram to your diagnostics tab
- Your 0.80 threshold will then have a meaningful interpretation: "this label is correct 80% of the time"

**Usability impact:** Users see more trustworthy confidence scores; the colored dots in your GUI become genuinely informative.

#### 1.2 — Model Versioning & Comparison ✅

**Problem:** You can't compare model v1 vs v2 on the same data.

**Solution:** Extend `training_runs` to store holdout predictions and add a comparison view.

**Implementation:**
- At training time, persist `(detection_id, y_true, y_pred, confidence)` for every validation example
- Add a `compare_runs(run_a, run_b)` function that computes per-class deltas in F1/precision/recall
- Add a diagnostics panel in the GUI showing improvement/regression per class
- Add automatic rollback: if the new model's F1_weighted is lower than the previous run, keep the old model

**Usability impact:** Users can retrain with confidence, knowing regressions are caught.

#### 1.3 — Ensemble with LightGBM + XGBoost ✅

**Problem:** A single GBM is brittle on small data.

**Solution:** Train a 3-model ensemble (GBM, LightGBM, XGBoost) with soft voting.

**Implementation:**
- Add `lightgbm` and `xgboost` to requirements
- Train all three with the same data and features
- Predict via soft-vote averaging of `predict_proba`
- The ensemble is more robust, especially on minority classes

**Usability impact:** Invisible to users — same API, better predictions. Expect 2–5% F1 improvement.

---

### Phase 2: Visual Intelligence (2–4 weeks each) ✅ Completed 2026-02-21

#### 2.1 — Region-Level Image Feature Extraction ✅

**Problem:** The classifier never sees the image, so it can't distinguish visually distinct elements that share similar text/positional features.

**Solution:** For each detection bbox, crop the rendered page image and extract CNN features.

**Implementation:**
- At detection time (in `_apply_ml_feedback` or during featurization), crop the background image to each detection's bbox
- Run the crop through a pre-trained CNN backbone (e.g., ResNet-18 or EfficientNet-B0) with the classification head removed
- Take the 512-d feature vector and append it to your existing 45-feature vector
- Retrain the GBM/ensemble on the concatenated features (tabular + visual)

This is a "visual bag of features" approach — it's simple, doesn't require end-to-end deep learning training, and works with small annotation sets. You already render `background_image` in the ingest stage, so the image is available.

**Usability impact:** Significantly better at distinguishing legends (which have graphical symbols) from abbreviation tables (which are text-only), title blocks (which have border lines) from general notes, etc.

#### 2.2 — Layout-Aware Detection with a Pre-trained Document AI Model ✅

**Problem:** Your rule-based structural box detection and zoning are brittle.

**Solution:** Replace or augment the geometry rules in `analysis/structural_boxes.py` and `analysis/zoning.py` with a fine-tuned document layout model.

**Implementation:**
- Use a pre-trained model like **LayoutLMv3** (Microsoft) or **DocTR** (Mindee) as a layout detector
- Fine-tune on your corrections database — your COCO/VOC export already provides the annotations
- The model takes the page image + OCR text and outputs bounding boxes with class labels
- Run this as a parallel "ML zoning" path alongside your existing rule-based path
- Compare outputs; let the human resolve disagreements (which feeds back into training)

**Usability impact:** The system learns to recognize new plan-sheet layouts without new regex rules. This is the single highest-impact upgrade for handling diverse plan formats.

#### 2.3 — OCR Confidence-Weighted Tokens ✅

**Problem:** All tokens are treated equally, but some OCR extractions are unreliable.

**Solution:** Propagate PaddleOCR confidence scores through the pipeline and weight them in downstream decisions.

**Implementation:**
- You already extract `ocr_confs` in the VOCR stage but don't propagate them fully
- Add a `confidence` field to `GlyphBox`
- In reconciliation, prefer higher-confidence tokens when merging
- In featurization, add `mean_token_confidence` and `min_token_confidence` as features
- In the semantic checks, lower severity when findings are based on low-confidence tokens

**Usability impact:** Fewer false positives from bad OCR reads; users see confidence indicators on individual tokens.

---

### Phase 3: Semantic Intelligence (3–6 weeks each) ✅ Completed 2026-02-22

#### 3.1 — Text Embeddings for Classification ✅

**Problem:** Keyword regex matching is too shallow to understand text meaning.

**Solution:** Replace the 7 `kw_*_pattern` binary features with dense text embeddings.

**Implementation:**
- Use a lightweight sentence transformer (e.g., `all-MiniLM-L6-v2` — 80MB, fast CPU inference)
- For each block, concatenate the text content and compute a 384-d embedding
- Append to the feature vector (replacing or supplementing the keyword flags)
- The embedding captures semantic similarity: "GENERAL NOTES" and "CONSTRUCTION NOTES" will be close in embedding space even though they match different regex patterns

**Usability impact:** Better classification of ambiguous headers and non-standard naming conventions. Particularly valuable for plans from different firms that use different terminology.

#### 3.2 — LLM-Assisted Semantic Checks ✅

**Problem:** Your semantic checks are deterministic rules. They can detect "note #3 is missing" but can't understand whether a note's content actually addresses a code requirement.

**Solution:** Add an optional LLM pass for content-level checks.

**Implementation:**
- After the rule-based checks stage, add an optional `llm_checks` stage
- For each notes column, send the extracted text to a local LLM (e.g., Llama 3.1 8B via Ollama, or Claude API) with a prompt like: "Given these construction notes, identify any that are vague, contradictory, or reference non-existent details."
- Return findings as `CheckResult` with `severity="llm_suggestion"` and `check_id="LLM_*"`
- Gate behind `cfg.enable_llm_checks` so it's off by default

**Usability impact:** Catches content-level issues that rules cannot. Important: frame these as "suggestions" not "errors" since LLMs can hallucinate. Let the user accept/dismiss them, which feeds back into prompt refinement.

#### 3.3 — Cross-Page Graph Neural Network ✅

**Problem:** Your cross-page checks are independent rules. They don't learn from corrections.

**Solution:** Model the document as a graph where pages are nodes, shared entities (abbreviations, revision numbers, sheet references) are edges, and a GNN predicts inconsistencies.

**Implementation:**
- Build a document graph: nodes = (page, region), edges = shared abbreviation codes, revision numbers, sheet references, legend symbols
- Use features from your existing featurizer as node features
- Train a simple Graph Attention Network (GAT) to predict `is_inconsistent` per edge
- This learns patterns like "when abbreviation X appears on structural sheets but not on electrical sheets, it's usually an error"

**Usability impact:** Catches subtle cross-page inconsistencies that rules miss. Most valuable for large plan sets (50+ pages).

---

### Phase 4: Production ML Infrastructure ✅ Completed 2026-02-24

#### 4.1 — Data Drift Detection ✅

Add monitoring to detect when new PDFs are statistically different from the training set:

- Compute feature distributions from training data (means, stdevs, quantiles)
- At inference time, flag pages where features fall outside the expected distribution
- Surface warnings in the GUI: "This plan-sheet layout is unusual — predictions may be less reliable"

#### 4.2 — Automated Retraining Triggers ✅

Instead of manual `python scripts/train_model.py`, add:

- A threshold: retrain when `n_corrections_since_last_train > 50`
- A scheduler: retrain weekly if any new corrections exist
- A validation gate: only deploy the new model if F1_weighted improves

#### 4.3 — Feature Store ✅

As features grow more expensive (CNN crops, text embeddings), compute them once and cache:

- Add a `feature_cache` table: `(detection_id, feature_version, features_json)`
- Version features so stale caches are invalidated when you add new feature extractors
- This avoids re-computing embeddings and CNN features on every retrain

#### 4.4 — Experiment Tracking ✅

Lightweight custom tracker (no external dependencies) to track:

- Hyperparameter configurations
- Feature sets used
- Training/validation curves
- Model artifacts
- A/B comparison dashboards

---

## Priority Recommendation

If I were to pick the **three upgrades with the highest impact for your use case**, they would be:

1. **Phase 2.2 — Layout-Aware Detection** (LayoutLMv3 fine-tuning). This is the game-changer. Your corrections database already has the annotations. A pre-trained document AI model will learn plan-sheet layouts far better than hand-tuned geometry rules, and it generalizes to new formats you've never seen.

2. **Phase 3.1 — Text Embeddings** (Sentence transformers). This is low-effort, high-reward. Swap 7 binary features for a 384-d semantic vector and your classifier immediately understands "KEYNOTES" ≈ "GENERAL NOTES" without adding new regex patterns.

3. **Phase 1.1 — Confidence Calibration**. This is a one-day change that makes your entire system more trustworthy. Until confidence scores are calibrated, your ML relabelling threshold, your active learning ranking, and your GUI confidence dots are all unreliable.

These three together would push your system from a **4/10 to roughly a 7/10** in ML sophistication while maintaining the clean, usable architecture you've already built.

> **Status:** All three priorities above — plus every other phase on this roadmap — are now complete. The system scores **7/10** as of 2026-02-24 with 1,499 tests passing.
