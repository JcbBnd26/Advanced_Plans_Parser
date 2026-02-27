# Advanced Plans Parser — Execution Plan (Revised)

**Created:** 2026-02-26  
**Basis:** Revised from `ML_ROADMAP_REORDERED.md` with tighter scoping, realistic timelines, and prerequisite tasks surfaced.

**Core thesis:** Build the Query Engine first, test on real plans, then decide the next direction. Explore before you optimize.

---

## Guiding Principles

1. **Explore-first.** The ML foundation (7/10 extraction) is done. Don't polish it further until new capabilities reveal what matters.
2. **Prove value in days, not months.** Each phase starts with the smallest possible working version and gates investment behind evidence.
3. **Measure before you claim.** No capability ships without an evaluation harness that can produce a number.
4. **Price your dependencies.** Every external API call has a cost and latency budget attached before it enters the critical path.
5. **Narrow scope, widen later.** Start with 3 classes, not 13. Start with 50 questions, not 500. Expand only when the narrow version proves out.

---

## Current State (as of 2026-02-24)

| Dimension | Score | Notes |
|---|---|---|
| Pipeline architecture | 8/10 | 9-stage, drift-detection, feature cache |
| Feature engineering | 7/10 | 51+ features, 3 gen tabular + vision + embeddings |
| Model sophistication | 6/10 | Calibrated 3-model ensemble |
| Feedback loop | 8/10 | Active learning + auto-retrain + drift monitoring |
| MLOps & monitoring | 7/10 | Drift detection, experiment tracking, feature cache |
| Semantic understanding | 2/10 | None — this is the gap |
| Object recognition | 2/10 | CNN features only, no true detection |
| Multi-document reasoning | 2/10 | Cross-page checks only |

**Bottom line:** 7/10 extraction engine, 2/10 understanding engine. The work ahead is understanding.

---

## Phase 0: Prerequisites (Week 1)

These are blockers that must be resolved before any new capability work begins.

### 0.1 — PageResult Serialization (2–3 days)

**Problem:** The Query Engine needs to operate on saved runs. Currently there is no way to deserialize a `PageResult` from disk — the CLI script has a `TODO` for this.

**Deliverables:**
- `PageResult.to_dict()` and `PageResult.from_dict()` round-trip methods
- Serialization to JSON (or MessagePack for size) in `artifacts/` during pipeline export
- Loader function: `load_run(run_dir) -> List[PageResult]`
- Unit tests confirming round-trip fidelity for all nested structures (title blocks, notes, legends, revisions, blocks, regions)

**Acceptance criteria:**
- `PageResult.from_dict(pr.to_dict()) == pr` for every field
- Saved runs in `runs/` can be reloaded without re-processing the PDF

### 0.2 — LLM Cost & Latency Budget (1 day)

**Problem:** Priorities 1.1–1.3 depend on LLM API calls. Without a cost model, we can't scope the compliance checker or know if the architecture is viable at scale.

**Deliverables:**
- Measure average context size (tokens) for a 10-page plan
- Calculate cost-per-query for Claude Sonnet, GPT-4 Turbo, and Llama 3.1 8B (local)
- Measure round-trip latency for each provider
- Estimate cost of running full compliance check (N requirements × cost-per-query)
- Document results in a table

**Expected output (example):**

| Provider | Avg Tokens/Query | Cost/Query | Latency | Compliance (100 reqs) |
|---|---|---|---|---|
| Claude Sonnet | ~8,000 | ~$0.04 | ~3s | ~$4.00, ~5 min |
| GPT-4 Turbo | ~8,000 | ~$0.05 | ~4s | ~$5.00, ~7 min |
| Llama 3.1 8B (local) | ~8,000 | $0 | ~8s | $0, ~13 min |

**Decision gate:** If local LLM quality is unacceptable on 10 test questions, the product has a hard dependency on paid APIs. Surface this as a business constraint.

### 0.3 — Privacy & Confidentiality Policy (1 day, non-technical)

**Problem:** Sending architectural plan content to external LLM APIs may violate NDAs or client confidentiality agreements.

**Deliverables:**
- Document which data leaves the machine for each LLM provider
- Determine if Anthropic/OpenAI data-retention policies are acceptable
- If not: local-only mode (Ollama) becomes mandatory, which gates quality expectations
- Create a `cfg.llm_policy` setting: `"local_only" | "cloud_allowed" | "cloud_with_consent"`

---

## Phase 1: Prove Semantic Understanding (Weeks 2–5)

### 1.1 — Document Query Engine (Weeks 2–3)

Build exactly as specified in the roadmap, with these additions:

**Week 2: Core engine + evaluation harness**

- Implement `DocumentQueryEngine` class (index builder, semantic search, LLM query)
- Support 3 providers: Anthropic, OpenAI, Ollama
- CLI interactive script (`scripts/query/interactive_query.py`)
- **Evaluation harness** (critical addition):
  - Create `tests/query/question_bank.json` with 50 ground-truth Q&A pairs across available test plans
  - Automated scoring: LLM-as-judge comparing generated answer vs ground truth
  - Metrics: accuracy (%), average relevance score (1–5), citation correctness rate
  - Script: `scripts/diagnostics/run_query_eval.py`

**Week 3: GUI integration + prompt refinement**

- Add `QueryTab` to GUI (`scripts/gui/tab_query.py`)
- Chat interface with history, clear, export
- Prompt optimization based on evaluation results
- Add response caching (cache by query hash + page content hash)
- Error handling: API timeouts, rate limits, malformed responses, empty results

**Milestone at end of Week 3:**
- Query engine answers 50 test questions with measured accuracy
- Non-technical user can use the GUI tab
- Cost-per-session is documented

**Decision gate:** Is measured accuracy >70%? Is it faster than manual search? If yes → continue. If no → diagnose failure modes before proceeding.

### 1.2 — Compliance Assistant (Weeks 4–5)

> **Reframed from "Compliance Checker" to "Compliance Assistant."**  
> This is advisory — it suggests areas to review, not pass/fail determinations.  
> Building codes require nuanced interpretation (occupancy type, sprinkler status, etc.) that an automated system cannot reliably adjudicate.

**Week 4: Core checker**

- Implement `ComplianceAssistant` class (not `ComplianceChecker`)
- Load requirements from structured JSON knowledge base
- Batch LLM calls where possible (group related requirements into single prompts to reduce cost/latency)
- Output `Finding` objects with severity: `"review"` (not `"error"`), `"note"`, `"info"`
- All findings explicitly labeled: *"This is an automated suggestion. Professional review required."*

**Week 5: Integration + evaluation**

- GUI panel showing findings grouped by code section
- Findings link to source pages and relevant text
- Evaluation: run against 5 known plans with manually verified compliance status
- Measure false-positive rate (target: <30%) and recall (target: >60%)

**Scope limits:**
- Start with ONE code: IBC 2021 egress requirements only (~20 requirements)
- Do NOT attempt full IBC coverage — there are thousands of provisions
- Do NOT make pass/fail claims — always "review suggested"

**Decision gate:** Is the false-positive rate tolerable? Do real reviewers find the suggestions useful?

### 1.3 — Structured Entity Extraction (Weeks 4–5, parallel with 1.2)

- Implement `EntityExtractor` using the Query Engine with structured prompts
- Use structured output patterns (JSON schema enforcement) rather than hoping the LLM returns valid JSON
- Start with 3 entity types only: **Materials**, **Dimensions**, **Equipment**
- Export to CSV/JSON for consumption by external tools
- Evaluate extraction accuracy on 5 test plans with manually verified entities

---

## Phase 2: Visual Object Recognition (Weeks 6–11)

### 2.1 — Symbol Detection, Narrow Scope (Weeks 6–9)

> **Reduced from 13 classes to 4 high-frequency classes:** door, window, outlet, column.  
> Expand only after proving >0.7 mAP on these 4.

**Week 6: Data preparation**

- Export YOLO dataset from `CorrectionStore` using existing annotation data
- Audit exported labels for quality (bounding box accuracy, class correctness)
- Supplement with manual labeling if <200 instances per class
- Train/val split (80/20, stratified by class)

**Week 7: Initial training**

- YOLOv8n (nano) as baseline — fast iteration
- Train on exported dataset, evaluate mAP
- Analyze failure modes: too small? overlapping? misclassified?
- If mAP <0.5: label more data, focusing on failure modes

**Week 8: Iteration**

- Active learning loop: use model predictions to suggest labels → human corrects
- Retrain with expanded dataset
- Try YOLOv8s (small) if nano plateaus
- Multi-scale training for small symbols (outlets) vs large symbols (doors)

**Week 9: Pipeline integration**

- `SymbolDetector` class integrated into pipeline as optional stage
- Gated behind `cfg.enable_symbol_detection`
- Results added to `PageResult.detected_symbols`
- Overlay visualization for detected symbols
- Test on 5 unseen plans

**Milestone:** >0.7 mAP on validation set for 4 classes, <5% false positives.

**Decision gate:** Does symbol detection add value beyond what the extraction pipeline already provides? Are users interested in symbol counts/locations?

### 2.2 — Dimension Association (Weeks 10–11)

- Find text near detected objects using OCR + spatial proximity
- Parse dimension strings (regex for `3'-6"`, `W14x22`, `#5 @ 12" o.c.`, etc.)
- Associate dimension with nearest detected symbol
- Build structured database linking objects to their dimensions
- Evaluate on 5 test plans

---

## Phase 3: Multi-Document Intelligence (Weeks 12–15)

### 3.1 — Revision Comparison (Weeks 12–13)

- Implement `RevisionComparer` with spatial IoU matching + text similarity
- Page matching via title block fingerprinting
- Change classification: added, deleted, modified
- Visual diff overlay (red/green/yellow)
- Evaluate on 3 real revision pairs
- Target: correctly identifies >80% of actual changes

### 3.2 — Cross-Reference Validation (Weeks 14–15)

- Regex + LLM extraction of reference patterns ("See Detail 7/A-5", "Refer to Sheet E-3")
- Validate target existence in document set
- Report broken references with source location
- Add to existing `checks` stage
- Target: finds 95%+ of broken references

---

## What NOT To Do (Yet)

### Don't optimize the existing ML
The 7/10 extraction score is sufficient. Avoid:
- Hyperparameter tuning campaigns
- Additional feature engineering
- Ensemble architecture experiments
- Model compression / inference optimization

### Don't harden infrastructure
Save for post-exploration:
- Checkpoint/resume for long runs
- Process pooling / parallelism
- Structured logging overhaul
- Production monitoring/metrics dashboards

### Don't build deployment infrastructure
- No FastAPI web service
- No Docker containers
- No cloud deployment
- No multi-tenancy

Ship the desktop tool. Get real feedback. Then harden.

---

## Realistic Timeline

| Phase | Scope | Optimistic | Realistic | Depends On |
|---|---|---|---|---|
| **Phase 0** | Prerequisites | 1 week | 1–2 weeks | Nothing |
| **Phase 1.1** | Query Engine | 2 weeks | 3 weeks | Phase 0 |
| **Phase 1.2** | Compliance Assistant | 2 weeks | 3 weeks | Phase 1.1 |
| **Phase 1.3** | Entity Extraction | 2 weeks | 2–3 weeks | Phase 1.1 (parallel with 1.2) |
| **Phase 2.1** | Symbol Detection (4 classes) | 4 weeks | 5–6 weeks | Phase 0 |
| **Phase 2.2** | Dimension Association | 2 weeks | 2–3 weeks | Phase 2.1 |
| **Phase 3.1** | Revision Comparison | 2 weeks | 3 weeks | Phase 0 |
| **Phase 3.2** | Cross-Reference Validation | 2 weeks | 2 weeks | Phase 1.1 |

**Total optimistic:** 14 weeks  
**Total realistic:** 20–26 weeks  

Phases 1.2/1.3 can run in parallel. Phase 2 can overlap with Phase 1 if a second person is available.

---

## Success Metrics

### Query Engine
| Metric | Target | How Measured |
|---|---|---|
| Accuracy on 50-question bank | >70% | LLM-as-judge + human spot-check |
| Citation correctness | >80% | Manual verification of page/note references |
| Latency per query | <10s (cloud), <20s (local) | Timed in evaluation harness |
| Cost per 10-page session | <$1.00 | Token counting |

### Compliance Assistant
| Metric | Target | How Measured |
|---|---|---|
| False positive rate | <30% | Compared to professional review on 5 plans |
| Recall (real issues found) | >60% | Same comparison |
| User trust | Advisory framing accepted | Qualitative feedback |

### Symbol Detection
| Metric | Target | How Measured |
|---|---|---|
| mAP@0.5 (4 classes) | >0.7 | YOLOv8 validation metrics |
| False positive rate | <5% | Manual inspection of 100 predictions |
| Per-class AP | >0.6 each | No class left behind |

### Revision Comparison
| Metric | Target | How Measured |
|---|---|---|
| Change detection recall | >80% | Compared to manual diff on 3 revision pairs |
| False change rate | <15% | Same comparison |

### Overall
| Metric | Target |
|---|---|
| You use it on real projects | Yes/No |
| Someone else finds it valuable | Yes/No |
| Saves >1 hour per document | Timed comparison |

---

## Risk Register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Local LLM quality too low for technical AEC content | High | Blocks offline use | Test Ollama early (Phase 0.2); fall back to cloud with consent |
| LLM costs too high for compliance at scale | Medium | Limits compliance scope | Batch prompts; cache responses; limit to 20 requirements initially |
| Plan content under NDA sent to cloud APIs | High | Legal/business risk | Phase 0.3 privacy policy; `cfg.llm_policy` gating |
| Symbol detection needs >1,000 labels per class | Medium | Delays Phase 2 | Start with 4 classes; use active learning; label failure modes |
| Query Engine accuracy <70% | Low-Medium | Undermines thesis | Prompt engineering; better context building; more aggressive chunking |
| Timeline slippage (2x estimated) | High | Delays all phases | Gate each phase; cut scope rather than extend timeline |
| PageResult serialization reveals missing data | Low | Blocks Phase 0.1 | Audit all `PageResult` fields early; add `__eq__` for round-trip testing |

---

## Immediate Next Steps

1. **This week:** Implement PageResult serialization (Phase 0.1)
2. **This week:** Run LLM cost/latency benchmarks (Phase 0.2)
3. **This week:** Draft privacy policy for plan data (Phase 0.3)
4. **Next week:** Begin Query Engine core implementation (Phase 1.1)
5. **Next week:** Start building the 50-question evaluation bank from existing test plans

---

## Projected Capability Scores After Execution

| Capability | Current | After Phase 1 | After Phase 2 | After Phase 3 |
|---|---|---|---|---|
| Structure Extraction | 7/10 | 7/10 | 7/10 | 7/10 |
| Semantic Understanding | 2/10 | 6/10 | 6/10 | 7/10 |
| Object Recognition | 2/10 | 2/10 | 6/10 | 6/10 |
| Multi-Document | 2/10 | 2/10 | 2/10 | 5/10 |
| **Overall** | **5/10** | **5.5/10** | **6/10** | **6.5/10** |

> Note: These are conservative estimates. The original roadmap projected 7.5/10 overall — that's achievable but assumes everything works on the first try. Plan for 6.5/10 and be pleasantly surprised.
