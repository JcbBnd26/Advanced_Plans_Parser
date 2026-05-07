# PlanCheck — Development Roadmap
*Generated from planning session — May 2026*

---

## Guiding Principles

- **Geometry first, semantics second.** Nail spatial grouping before optimizing labels. Bad grouping permanently corrupts training features. Bad labels can be corrected over time.
- **Four signals, one system.** Yes, Is Group, Edit Auto Group, and Not Group all feed the same grouper. They are degrees of the same signal, not separate systems. The classifier is a future pipeline not touched in this roadmap.
- **The flywheel.** Run pipeline → machine gets some things wrong → human corrects → model improves → repeat. Every correction session makes the next one faster.
- **Aim small, miss small.** Incremental, independently testable steps. The app should work after every phase.
- **Diagnostic first.** For risky new passes, implement as read-only inspection before adding any mutation or injection logic.

---

## Training Signal Architecture

There is one system right now — the grouper. All four signals feed it. They are degrees of the same feedback, not separate pipelines.

The classifier is a future pipeline. It does not exist yet in this roadmap. The label field in storage sits empty until that day comes.

### The Four Signals

**Yes** — the machine got it right. Positive reinforcement. The grouper did exactly what it should have done here. Keep doing this.

**Is Group** — remember this pattern. This is a meaningful semantic unit worth storing as a fingerprint. Triggers the full Is Group snapshot capture. This is how the pattern library gets built.

**Edit Auto Group** — the machine was close but not precise. The human refines the boundary by adding, removing, splitting, or merging boxes. The delta between what the machine produced and what the human corrected it to is the richest training signal of the four. Most correction work lives here.

**Not Group** — these words have no business being together. Strong negative signal. Reserved for when the machine is fundamentally wrong about a relationship, not just imprecise. Used sparingly — this is the sledgehammer, not the scalpel.

### Signal Hierarchy
- **Yes** and **Is Group** are both positive but different in intent — Yes rewards the grouper, Is Group builds the pattern library
- **Edit Auto Group** is the precision instrument — the delta is what the model learns from most
- **Not Group** is the blunt instrument — used for egregiously wrong associations only

---

## The Is Group Snapshot — What Gets Captured

Every time a human marks Is Group, the pipeline captures a full snapshot. This record must be self-contained — it should tell the full story of that group without referencing anything else.

### Bucket 1 — The Boxes Themselves
- Bounding box coordinates (x, y, width, height) for each box
- Text content of each box
- Font size of each box
- Text alignment (left, center, right)
- Position relative to the group's own center (not the page)
- Which box is the group root / anchor

### Bucket 2 — The Group's Geometry
- Total bounding box of the group
- Number of boxes in the group
- Dominant axis (horizontal vs vertical)
- Density — how tightly packed the boxes are
- Aspect ratio of the overall group shape
- Whether any vector geometry is nearby (lines, circles, arrows)
- **Normalized geometry** — scaled so group size doesn't matter, only the shape of the relationship between boxes. This is what allows the model to recognize the same pattern at different scales across different plan sheets.

### Bucket 3 — Page Context
- Page number
- Project ID
- Proximity to page edges
- Proximity to title block region
- Nearby groups and their rough characteristics
- Zone of the page (margin, field, corner, center)

### Bucket 4 — Session Provenance
- Timestamp
- Session ID
- Source PDF filename
- Whether the record was human-captured or machine-suggested

### Empty Label Field
Each record includes a label field that is empty for now. This field gets populated in Phase 4 when semantic labeling begins. The schema is designed for the future without forcing it yet.

### is_verified Flag
Boolean. Human-captured records are verified by definition. When the model later starts auto-suggesting groups, those records come in as unverified until a human confirms them. Same table, same structure, different trust level.

---

## Storage Design

### Two Separate Stores

**Example Store**
Every Is Group capture lives here. Raw fingerprints. The training pattern library.
- One row per captured group
- Full snapshot stored as JSON columns for flexibility
- Empty label field for future use
- is_verified boolean

**Correction Store** *(extends existing CorrectionStore)*
Yes/No grouper feedback lives here.
- Records the machine's grouping at time of judgment
- Records the human's verdict (Yes/No)
- For No verdicts: records the correction type (too many boxes, too few boxes, wrong boxes)
- The delta between machine grouping and corrected grouping is stored — this is valuable training signal

### JSON Columns
Spatial data stored as JSON inside SQLite columns. This allows new properties to be added as they are discovered without migrating the database schema. Early stage projects need this flexibility.

---

## Tab Structure

The UI has four primary tabs. Each tab has one job.

| Tab | Purpose | Audience |
|-----|---------|----------|
| **Pipeline** | Monitoring station for live pipeline runs | All users |
| **Grouper** | Learn sessions and group editing | Builder / Developer |
| **Labeler** | Correction and semantic labeling of pipeline output | All users |
| **Settings** | All backend controls | Builder / Developer |

The Grouper tab is a builder tool. It can be hidden behind a developer mode toggle later without touching the Labeler at all.

---

## Phase 1 — Grouper Tab & Learn Session

*This is the immediate next work. Everything downstream depends on good grouping data.*

### What Is NOT Changing
- The existing pipeline
- The existing Labeler tab
- The existing CorrectionStore
- The existing canvas rendering engine

---

### 1A — Grouper Tab Shell
Create the Grouper tab in the existing UI framework. Three modes accessible from a toolbar inside the tab:
- **Learn Session** — clean canvas, build groups from scratch
- **Edit Mode** — machine groupings shown, tweak them
- **Inspect Mode** — single click to reveal group membership, read only

---

### 1B — Learn Session UI

**Canvas State**
- All word-level boxes rendered as grey outlines
- No pre-existing group boundaries shown
- Clean slate — teaching mode

**Build Gesture — Assembling a Group**
- Hold Shift + left-click words one at a time
- Each clicked word stays highlighted as Shift is held
- Release Shift when all desired words are selected
- Left-click any one of the highlighted words
- Popup appears with a single option: **Group**
- Click Group — all highlighted words become a confirmed group
- Green bounding box renders around the group
- Group stays actively selected (highlighted green)

**Inspect Gesture — Revealing Group Membership**
- Single click any word
- If the word belongs to a group → entire group highlights green together
- If the word is ungrouped → word highlights in a neutral color (orphan state, no action taken)

**Clear Gesture**
- Right-click on blank space or any unselected element
- All selections clear, canvas returns to grey state

**Session Controls (top control bar)**
- **Show Machine Groups** toggle — ON shows machine grouping boundaries, OFF shows raw word boxes only. This is the primary tool for evaluating whether training is working.
- **Save & Next** — saves all groups on current page, advances to next page
- **Clear Groupings** — wipes all groups on current page only. Requires confirmation dialog before executing.
- **Abort Session** — ends session immediately, keeps all data from pages already saved

**Session Data**
- Auto-saves per page on Save & Next
- Each saved page gets a session record attached: timestamp, project ID, PDF filename, page number, group count
- Abort Session preserves all previously saved pages

---

### 1C — Edit Mode UI

Machine groupings are rendered on canvas. Human can adjust boundaries.

**Adding a word to a group**
- Shift+click an ungrouped word
- Left-click any word already in the target group
- Popup offers: **Add to Group**

**Removing a word from a group**
- Single click a word inside a group to inspect (whole group highlights)
- Shift+click the specific word to remove
- Popup offers: **Remove from Group**

**Splitting a group**
- Shift+click the words to break out
- Popup offers: **Split into New Group**

**Merging two groups**
- Click one group to select it
- Shift+click any word in a second group
- Popup offers: **Merge Groups**

**What Edit Mode Produces**
Every edit generates a delta record — what the machine had, what the human changed. The delta is stored alongside the corrected result. This delta is valuable training signal because it shows the model exactly where its judgment was wrong.

---

### 1D — Is Group Capture Function

This function fires when a human confirms a group in Learn Session or Edit Mode.

- Triggered on Group confirmation or Is Group gesture
- Pulls full snapshot at pipeline time — TOCR-enriched data, spatial relationships already calculated, group boundaries already resolved
- Captures all three data buckets plus provenance
- Writes one record to the Example Store
- Simultaneously writes a Yes signal to the Correction Store

The capture happens at pipeline time deliberately — this is the rich version of the fingerprint with all processed data included, not just raw PDF coordinates.

---

### 1E — Storage Implementation

**Example Store** — new table
```
example_id        TEXT PRIMARY KEY
session_id        TEXT
project_id        TEXT
pdf_filename      TEXT
page_number       INTEGER
timestamp         TEXT
boxes             JSON
group_geometry    JSON
normalized_geometry JSON
page_context      JSON
label             TEXT     -- empty for now
is_verified       BOOLEAN
```

**Correction Store** — extend existing
Records all four signals against machine output.
```
signal             TEXT     -- yes | is_group | edit | not_group
machine_grouping   JSON     -- what the machine had
corrected_grouping JSON     -- what the human produced (null for Yes and Not Group)
delta              JSON     -- difference between machine and corrected (edit signal only)
```

---

### Phase 1 Test Criteria
- Can open Grouper tab without breaking existing tabs
- Can enter Learn Session on a loaded PDF page
- Can Shift+click to select multiple words
- Can confirm a group and see green bounding box
- Can single-click a grouped word and see full group highlight
- Can single-click an ungrouped word and see neutral highlight
- Can right-click to clear all selections
- Can toggle Show Machine Groups on and off
- Can Save & Next and see page advance
- Can Clear Groupings with confirmation dialog
- Can Abort Session and verify previously saved pages are intact
- Can enter Edit Mode and add/remove/split/merge groups
- Every confirmed group writes a record to Example Store
- Every Yes / Is Group / Edit / Not Group writes a record to Correction Store with correct signal type

---

## Phase 2 — Seed The Pattern Library

*Manual labor. The most valuable thing you can do for the model after Phase 1 ships.*

- Run Learn Sessions on 20-30 real plan pages minimum
- Capture as many meaningful Is Group examples as possible
- Do not label anything yet — pure fingerprint collection
- Use Edit Mode to correct any machine groupings that are close but wrong
- Note words that appear to be persistent orphans across multiple pages — log them but do not act yet
- Target variety — different plan types, different firms, different disciplines (architectural, structural, mechanical)

### Goal
A pattern library rich enough that the GNN has real geometric examples to learn from. Quantity and variety both matter.

---

## Phase 3 — Train The Grouper

- Feed captured examples into the GNN
- Let the model start making grouping suggestions on new pages
- Use Yes/No signals in the Grouper tab to refine
- Use the Show Machine Groups toggle to directly compare machine judgment vs reality on the same page
- Measure improvement session over session

### The Progress Indicator
If you flip Show Machine Groups on after ten sessions and the boundaries look better than they did before — the flywheel is spinning. That is your measurable milestone.

---

## Phase 4 — Labeler Tab & Semantic Classification

*This is where the app starts reading plans in the full sense.*

- Build the correction and labeling interface in the Labeler tab
- Define the semantic label taxonomy (10-15 labels to start — Sheet Number, Room Label, Detail Reference, Scale, General Note, Revision, etc.)
- Start adding semantic labels to confirmed groups
- Feed labeled examples back into the classifier
- The GNN already has geometric intuition from Phase 3 — labels now become names for patterns it already recognizes

### Label Taxonomy Note
Before building Phase 4, sit down with 5-10 real plan sheets and write down every distinct type of text element worth caring about. That list becomes the label registry. Start with 10-15 maximum. The list can grow over time.

---

## Phase 5 — UI Restructure

*This is where the app becomes a product, not just a tool.*

Based on ML_NOTES.md:

- Replace the 86-checkbox TOCR progress indicator with a per-page progress bar — small box per sheet that fills as pages finish TOCR
- Restructure the Pipeline tab as a monitoring station for the live process
- Implement standard menu bar: File, Edit, View, Settings
- Move Adv ML Runtime, Optional ML Features, LLM Runtime into Settings dialogs — they pop up in their own windows when activated
- Sort remaining tabs — remove or combine as appropriate
- Functions that don't need a full tab go into dropdown menus
- Organic feel in the UI
- Separate business-facing features from backend/builder features
- Hide Grouper tab and other builder tools behind a developer mode toggle
- Remember: some features are for plan grading purposes later in development

---

## Phase 6 — Orphan Logic

*Refinement. Only makes sense after Phase 3 is solid.*

By this point the trained grouper has a strong sense of what valid groups look like. Orphan logic becomes tractable.

- Revisit persistently ungrouped words across many pages
- Distinguish between: words that are orphans by design (dimension ticks, watermarks, border text) vs words the grouper is failing to claim
- Build the null group concept — a valid semantic state meaning "this word stands alone intentionally"
- Build a rule-based pass that attempts to rehome orphaned words into nearby groups
- All rehoming suggestions require human verification before writing to training data
- Teaching the machine that "alone" is sometimes the correct answer is as important as teaching it what groups look like

---

## Open Items / Decisions Not Yet Made

- Exact visual styling of the Grouper canvas (colors, box weights, fonts)
- Whether Edit Mode and Learn Session share the same canvas or are separate views
- Keyboard shortcut assignments for Y / N / G gestures in pipeline correction flow
- Label taxonomy (to be decided before Phase 4 by reviewing real plan sheets)
- Developer mode toggle — what it shows/hides exactly
- Plan grading feature scope (noted for later, not yet designed)

---

*"Aim small, miss small."*
