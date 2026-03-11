# Advanced Plans Parser
## Hierarchical Two-Stage Classification — ML Architecture Plan & Agent Analysis Guide

---

## Executive Summary

This document is a step-by-step analysis guide for an LLM agent to evaluate, design, and validate a Hierarchical Two-Stage Classification system for the Advanced Plans Parser. It is grounded in the existing codebase — specifically `ElementClassifier`, the corrections database, the features pipeline, and the existing LLM checks infrastructure.

> **Core Problem:** Many plan elements share visual structure but differ in meaning. A "title" above a floor plan is functionally different from a title above a legend, a detail bubble, or a graph. A flat label system either conflates them (loses precision) or starves each sub-type of training data (loses accuracy). The hierarchical approach solves both problems.

The plan is organized into four phases:

1. **Phase 1 — Audit:** Understand what you have
2. **Phase 2 — Design:** Build the family taxonomy
3. **Phase 3 — Build:** Implement Stage 1 and Stage 2 classifiers
4. **Phase 4 — Wire:** Connect the LLM tiebreaker

---

## Phase 1 — Audit the Current State

Before designing anything new, the agent must understand what the existing data and model actually know. This phase produces three deliverables: a label consolidation map, a data sufficiency report, and a feature relevance assessment.

### 1.1  Analyze the Corrections Database

The agent should query the corrections database and answer the following questions:

1. How many corrections exist per `corrected_element_type`? Flag any label with fewer than 20 corrections as data-insufficient.
2. Are there any typo or non-canonical labels? (e.g., the known `lable` typo.) List them.
3. What is the ratio of the largest class to the smallest class? A ratio above 20:1 signals severe imbalance.
4. How many of the 18 training runs show zero F1 for a given class? Each zero-F1 class is a candidate for family consolidation.

> **Current known state:** `notes_column` (366) is 366x larger than `standard_detail` (1). Eight out of 14 classes show zero F1 in the most recent run. This is the imbalance problem the hierarchy directly solves.

---

### 1.2  Map Every Label to a Structural Family

The agent should build a consolidation table. For every label in the corrections database and `label_registry.json`, assign it to one of the proposed Stage 1 families below. The families are chosen based on behavioral role in the pipeline — elements in the same family are processed similarly downstream.

| Stage 1 Family | Current Labels That Map Here | Why One Family | Downstream Behavior |
|---|---|---|---|
| `title` | header, misc_title, plan_title, page_title, detail_title, map_title, graph_title, box_title, callokie_logo* | All are short, prominent text identifying a drawing region or section | Extract as section identifier; feed to Stage 2 for sub-type |
| `notes` | notes_column, notes_block | Both are numbered/bulleted construction requirement text | Feed to LLM checks; run code compliance analysis |
| `legend` | legend | Symbol-to-meaning lookup tables | Parse symbol-description pairs |
| `title_block` | title_block, county, name_block | Formal sheet-border metadata blocks | Extract project/sheet/date fields via title_block.py |
| `reference` | standard_detail, revision | Both point outside the current sheet to other content | Resolve cross-sheet references; build revision history |
| `abbreviations` | abbreviations | Two-column code lookup tables | Build abbreviation dictionary for text normalization |
| `negative` | \_\_negative\_\_ | False positive — not a real element | Suppress; do not pass downstream |

\* `callokie_logo` has 1 example. If it is a visual mark (not text-based), it may belong in a future `graphic_mark` family rather than `title`.

---

### 1.3  Assess Feature Relevance Per Family

The existing feature set in `classifier.py` has 42+ features covering position, font, text content, and relational context. The agent should evaluate which features are most discriminative at the family level vs. the subtype level.

| Feature Group | Stage 1 Value (family) | Stage 2 Value (subtype) |
|---|---|---|
| Position: `x_frac`, `y_frac`, `zone` | **HIGH** — title_block lives bottom-right; notes fills middle-left | **MEDIUM** — detail titles cluster near bubbles; page titles are top-center |
| Font: `is_all_caps`, `is_bold`, `font_size_pt` | **HIGH** — title families are large/bold; notes are small/regular | **LOW** — most subtypes share font characteristics within family |
| Text content: `kw_*` keyword patterns | **MEDIUM** — existing patterns cover notes/legend/revision well | **HIGH** — "FLOOR PLAN" vs "SITE PLAN" vs "KEYNOTE" is pure text |
| Text content: actual text string | **NOT USED** (GBM doesn't read raw text) | **CRITICAL** for LLM tiebreaker — this is the semantic signal |
| Relational: `is_below_header`, `sibling_count` | **LOW** at Stage 1 (families differ in bigger ways) | **HIGH** — a title below a north arrow is a map title; below a bubble is a detail title |
| Spatial: `neighbor_count`, `width_frac` | **MEDIUM** — notes columns are wide and have many neighbors | **HIGH** — detail titles are small/isolated; plan titles are wide/prominent |

> **Key insight:** Stage 1 is a spatial and structural decision — it uses position, size, and font. Stage 2 is a semantic decision — it uses text content, relative context, and potentially the LLM. This matches what the two layers of the system are already good at.

---

## Phase 2 — Design the Family Taxonomy

This phase produces the authoritative label map that will govern all future annotation and training. The agent's job is to finalize the Stage 1 families and define the Stage 2 subtypes for each family that has known variation.

### 2.1  Define Stage 2 Subtypes for the `title` Family

The `title` family is where the most work is needed. These are the known title subtypes observed in architectural/engineering plans:

| Subtype Label | Identifying Characteristics | Key Feature Signal | Example Text |
|---|---|---|---|
| `page_title` | Large, top-center or top-left of sheet; spans full width; often all-caps | `y_frac < 0.15`, `width_frac > 0.4`, `is_all_caps` | FLOOR PLAN, SITE PLAN, ROOF PLAN |
| `plan_title` | Inside or above a specific drawing view; smaller than page title; names the view | `width_frac` 0.1–0.3, near drawing region in `zone=drawing` | FIRST FLOOR PLAN, NORTH ELEVATION |
| `detail_title` | Below or beside a circled detail bubble; small bounding box; alphanumeric ID nearby | small `height_frac`, neighbor with circle graphic, `zone=details` | WINDOW HEAD DETAIL, SILL DETAIL 3/A5 |
| `section_title` | Arrow-terminated section cut label; references a sheet number; typically A-A or 1-1 format | `contains_digit`, `has_period_after_num`, near section arrow graphic | SECTION A-A, SECTION 1/A4 |
| `graph_title` | Above a data graph or schedule table; short; may have underline or border | above tabular content, width matches table width | DOOR SCHEDULE, WINDOW SCHEDULE |
| `map_title` | Above or beside a vicinity/site map; often includes scale or compass reference nearby | near north arrow graphic, near scale bar | VICINITY MAP, SITE MAP |
| `box_title` | Inside a bordered box, not the title block; department/agency/project banner | `is_boxed=True` (from misc_titles.py), `zone=border` or `page` | DEPT OF TRANSPORTATION, CITY OF NORMAN |

---

### 2.2  Define Stage 2 Subtypes for Other Families

Not every Stage 1 family needs a Stage 2. The agent should evaluate which families have meaningful internal variation worth distinguishing:

| Family | Needs Stage 2? | Reasoning |
|---|---|---|
| `notes` | Yes — later | `notes_column` vs `notes_block` already exist. When data is sufficient, train a subtype classifier. For now, keep both as-is since they already have enough examples. |
| `reference` | Yes — later | `standard_detail` vs `revision` are already distinct labels with different downstream behavior. These can be the two subtypes once the family has enough combined data. |
| `title_block` | Maybe | `county` and `name_block` are rare sub-elements. Treat as `title_block` for Stage 1. Only add Stage 2 if downstream extraction needs to distinguish them. |
| `legend` | No | Legends don't have meaningful subtypes yet. Single class is fine. |
| `abbreviations` | No | Abbreviation tables are structurally uniform. Single class is fine. |
| `negative` | No | False positives are false positives. No subtype value. |

---

### 2.3  Define the Annotation Protocol

The agent should produce a clear annotation protocol to give the human annotator. This is the single most important output of Phase 2 — bad annotations make everything downstream worse.

#### Rules for Annotating in the GUI

1. **Always annotate to the Stage 1 family label first.** If the element is clearly a title of any kind, label it `title`. Do not try to label it `page_title` or `detail_title` yet — that comes in Stage 2.
2. **Add a note in the correction record when the subtype is visually obvious.** Example: `label='title'`, `notes='plan_title — above floor plan drawing'`. This seeds Stage 2 without requiring a second annotation pass.
3. **If an element has a typo label** (like `lable`), correct it to the canonical Stage 1 family label immediately.
4. **Target 50+ examples per Stage 1 family** before starting Stage 2 annotation. The current gap families are: `reference` (5 total), title-subtypes (currently all `misc_title` or `header`), `legend` (14).
5. **Never annotate ambiguous elements under pressure.** If you cannot determine the family, skip it. A skipped element is better than a mislabeled one.

---

## Phase 3 — Build the Two-Stage Classifier

This phase describes exactly what needs to be built, where it lives in the codebase, and how it connects to the existing infrastructure. The agent should produce implementation specifications, not code.

### 3.1  Stage 1 — Family Classifier (Modify Existing)

Stage 1 is a modification of the existing `ElementClassifier`, not a replacement. The work here is taxonomy remapping, not new code.

#### What Changes

- The label set in `corrections.db` gets consolidated. All `misc_title`, `header`, `plan_title`, `callokie_logo` corrections are remapped to `title`. All `county`, `name_block` corrections are remapped to `title_block`. All `standard_detail`, `revision` corrections are remapped to `reference`.
- `label_registry.json` gets updated to reflect the 7 Stage 1 families with new `text_patterns` for each.
- The `ZONE_VALUES` and keyword feature patterns in `classifier.py` get a new entry: `kw_title_pattern` replaces the existing fragmented title-related patterns.
- The retrain trigger threshold stays at 50 corrections. This is still appropriate for Stage 1.

#### What Does NOT Change

- `ElementClassifier` class — same class, same training loop, same GBM architecture.
- The corrections database schema — same tables. Only the label values change.
- The training pipeline — `train_model.py` runs exactly as before.
- Feature extraction in `features.py` — no structural changes needed.

> **Expected outcome after Stage 1 retraining:** weighted F1 should jump from 0.71 to above 0.85. The reason is simple arithmetic — you're consolidating 14 sparse classes into 7 well-populated ones. Each family will have 50–200+ examples instead of 1–366.

---

### 3.2  Stage 2 — Subtype Classifier (New Class)

Stage 2 is a new classifier that only runs when Stage 1 returns `title` (and later other families as they mature). It is architecturally identical to Stage 1 — same GBM, same training loop — but trained on a filtered subset of data.

#### New File: `src/plancheck/corrections/subtype_classifier.py`

This file mirrors `classifier.py` with these differences:

- **Class name:** `TitleSubtypeClassifier` (initially; later can be generalized to `SubtypeClassifier` with a `family=` parameter).
- **Model path:** `data/title_subtype_classifier.pkl`
- **Training data source:** Only `training_examples` rows where Stage 1 family = `title`. Filter applied in `build_training_set()` via a new `family=` parameter.
- **Feature set:** Same base features PLUS three new relational features that are only meaningful for title disambiguation: `near_north_arrow` (bool), `near_detail_bubble` (bool), `near_section_arrow` (bool). These require new entries in `features.py`.
- **Label set:** The 7 subtype labels defined in Phase 2.1: `page_title`, `plan_title`, `detail_title`, `section_title`, `graph_title`, `map_title`, `box_title`.

#### Training Data Bootstrap Strategy

Stage 2 has a cold start problem — you don't have 50+ labeled subtypes yet. The agent should specify this bootstrap sequence:

1. **Mine the `notes` field** of existing `title`-family corrections for subtype hints. Any correction with notes containing "floor plan", "site plan", "elevation" etc. can be auto-labeled as `plan_title`.
2. **Run pseudo-labeling** on high-confidence Stage 1 `title` detections using the spatial rules from Phase 2.1. Elements with `y_frac < 0.15` and `width_frac > 0.4` get pseudo-labeled `page_title`. Elements in `zone=details` get pseudo-labeled `detail_title`. These are pseudo-labels, not ground truth — they bootstrap, not finalize.
3. **Begin targeted annotation:** in the GUI, after Stage 1 stabilizes above 0.85 F1, switch annotation focus to subtype labeling of all `title` elements.
4. **Do not train Stage 2** until you have at least 20 examples per subtype. With 7 subtypes, that means 140+ title subtype annotations minimum.

---

### 3.3  The Prediction Interface

The agent should specify how the two stages connect at prediction time. This is the contract that the pipeline will use:

| Input / Condition | Output / Action |
|---|---|
| Any element enters prediction | Run Stage 1 (`ElementClassifier.predict`). Returns `(family_label, confidence)`. |
| Stage 1 confidence >= 0.7, family != `title` | Return `family_label` as final prediction. Done. |
| Stage 1 confidence < 0.7, any family | Flag for human review. Return `family_label` with `low_confidence=True`. |
| Stage 1 family == `title`, confidence >= 0.7 | Run Stage 2 (`TitleSubtypeClassifier.predict`). Returns `(subtype_label, subtype_confidence)`. |
| Stage 2 `subtype_confidence` >= 0.6 | Return `subtype_label` as final prediction. |
| Stage 2 `subtype_confidence` < 0.6 | Pass to LLM tiebreaker (Phase 4). Return LLM result or `subtype_label` if LLM unavailable. |

This interface should be implemented as a new function: `classify_element(feature_dict, text) -> ClassificationResult` in a new file `src/plancheck/corrections/hierarchical_classifier.py`.

---

## Phase 4 — Wire the LLM Tiebreaker

The LLM is not the first-line classifier. It is the last resort for cases where the two GBM stages are uncertain. This is the correct use of LLM budget — targeted, not blanket.

### 4.1  When the LLM Gets Invoked

The LLM tiebreaker is invoked when Stage 2 subtype confidence is below 0.6. This corresponds to cases where the element text contains the actual semantic signal that distinguishes subtypes — and the GBM cannot read raw text. Examples:

- `FLOOR PLAN` — could be `page_title` or `plan_title` depending on context
- `KEYNOTE LEGEND` — `title` or `legend` family? Text resolves it.
- `SCALE: 1/4" = 1'-0"` — `title_block` field or standalone reference?
- `VICINITY MAP` — `map_title`, but only if you read the words

---

### 4.2  The LLM Prompt Design

The agent should produce a prompt template specifically for title subtype disambiguation. This is different from the existing `llm_checks.py` prompt, which is focused on code compliance. The new prompt is focused on classification:

**System prompt:**
```
You are analyzing elements extracted from an architectural or engineering plan sheet.
Your task is to classify a text element into exactly one subtype category based on its
text content, position on the sheet, and nearby context.

You must return ONLY a JSON object with two keys:
  "subtype": one of: page_title, plan_title, detail_title, section_title, graph_title, map_title, box_title
  "confidence": float 0.0–1.0

Do not explain your reasoning. Do not return any other text.
```

**User message should include:**
- The element's text content (the actual words)
- Its normalized position: `zone`, `x_frac`, `y_frac`
- The top 2 candidates from Stage 2 with their confidence scores
- Any nearby element types (from neighbor context)

---

### 4.3  Integration Point in the Codebase

The existing LLM infrastructure in `src/plancheck/llm/` and `src/plancheck/checks/llm_checks.py` already supports Anthropic, OpenAI, and Ollama backends. The tiebreaker should:

1. **Reuse `LLMClient`** from `plancheck.llm.client` — no new client code needed.
2. **Be implemented as a new function:** `llm_classify_title_subtype(text, features, candidates) -> (subtype_label, confidence)` in `src/plancheck/checks/llm_checks.py`.
3. **Be gated by `is_llm_available()`** — if no LLM is configured, fall back to the Stage 2 argmax prediction.
4. **Log every LLM call and result** to the corrections database for future analysis. This data is valuable for improving Stage 2 over time.
5. **Be wrapped with a cost guard** — the existing `docs/LLM_BUDGET.md` should be consulted and a per-run token limit enforced.

> **Health metric:** The LLM should handle fewer than 5% of all classifications in steady state. If the LLM is being invoked on more than 10% of elements, Stage 2 is not trained sufficiently. Use LLM invocation rate as a health metric for Stage 2 quality.

---

## Phase 5 — Validation & Success Criteria

The agent should define what "done" looks like at each stage. These are the acceptance criteria for each phase.

| Milestone | Success Metric | Measurement Method | Blocker If Failed |
|---|---|---|---|
| Stage 1 remapped | Zero zero-F1 classes in training run | Run `train_model.py`, inspect `per_class_json` | Consolidation mapping is wrong — revisit Phase 2.1 |
| Stage 1 trained | Weighted F1 >= 0.85 on validation set | Training run metrics in corrections DB | Need more annotations on weak families |
| Stage 2 data ready | >=20 labeled examples per title subtype (140+ total) | Query corrections DB with `family='title'` filter | Need targeted annotation sprint |
| Stage 2 trained | Per-subtype F1 >= 0.70 for all subtypes with >=10 val examples | New training run for subtype classifier | Need more subtype-specific features |
| LLM wired | LLM invocation rate < 10% of title elements on test set | Count LLM calls in pipeline run logs | Stage 2 not confident enough — add features or data |
| System integrated | End-to-end pipeline run produces correct family + subtype for all 7 known title types on held-out test PDFs | Manual review of overlay output on 3 test documents | Integration bug in `hierarchical_classifier.py` routing |

---

### Sequencing — Aim Small, Miss Small

The hierarchy approach directly applies the project's core motto. Here is the specific sequencing that minimizes risk at each step:

1. **Fix the typo label first.** One SQL UPDATE. Zero risk.
2. **Consolidate labels in the database before writing any new code.** Validate that Stage 1 F1 improves with remapping. If it does not, the consolidation map is wrong — fix the map, not the code.
3. **Train Stage 1 to 0.85+ F1 before writing `SubtypeClassifier`.** Do not proceed to Stage 2 until Stage 1 is stable.
4. **Bootstrap Stage 2 with pseudo-labels and spatial rules.** Validate pseudo-label quality before training. Discard any pseudo-label batch with estimated accuracy below 80%.
5. **Wire the LLM tiebreaker last**, after Stage 2 is trained. Never use the LLM as a crutch to compensate for insufficient training data.

---

## Appendix — Files and Their Roles

Quick reference for the agent to locate relevant code during analysis:

| File Path | Role in This Plan |
|---|---|
| `src/plancheck/corrections/classifier.py` | Stage 1 classifier — `ElementClassifier` class. Train, predict, calibrate. |
| `src/plancheck/corrections/training_loop.py` | Core training loop used by both Stage 1 and future Stage 2. |
| `src/plancheck/corrections/features.py` | Feature extraction. Add `near_north_arrow`, `near_detail_bubble`, `near_section_arrow` here for Stage 2. |
| `data/label_registry.json` | Label definitions and `text_patterns`. Update Stage 1 family patterns here. |
| `data/corrections.db` | Source of truth for training data. Remapping happens via SQL UPDATE on `corrected_element_type`. |
| `scripts/train_model.py` | CLI training entry point. Runs Stage 1 training. Will need a `--stage2` flag added for Stage 2. |
| `src/plancheck/checks/llm_checks.py` | Add `llm_classify_title_subtype()` here. Reuses `LLMClient` from `plancheck.llm.client`. |
| `src/plancheck/analysis/misc_titles.py` | Detects boxed titles (`box_title` subtype). The `is_boxed` and `box_bbox` fields are Stage 2 features. |
| `src/plancheck/corrections/retrain_trigger.py` | Startup retrain check. Will need to trigger both Stage 1 and Stage 2 retraining. |
| `src/plancheck/corrections/drift_detection.py` | Fit drift detectors on both stages independently after training. |
| `NEW: src/plancheck/corrections/subtype_classifier.py` | Stage 2 classifier (to be created in Phase 3). |
| `NEW: src/plancheck/corrections/hierarchical_classifier.py` | Routing logic that chains Stage 1 → Stage 2 → LLM (to be created in Phase 3). |

---

> **Final Note for the Agent:** The most important outputs of this analysis are (1) the completed consolidation table from Phase 2.1 with actual correction counts per proposed family, and (2) the annotation protocol from Phase 2.3. Everything else depends on those two documents being correct. Produce those first before evaluating any code changes.
