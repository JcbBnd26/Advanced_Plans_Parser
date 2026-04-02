# Project System — Implementation Plan

## Overview

**Goal:** Each project gets its own folder containing its database, label registry, trained model, and settings. A "New Project" dialog creates the folder and lets you choose which classes to use. An "Open Project" dialog points the app at an existing project folder.

**Core insight:** Every path in the app already derives from `DATA_DIR` in `constants.py`, and `GroupingConfig` already carries all paths as strings. A "project" is just a folder that replaces `DATA_DIR`. The plumbing is 80% there.

---

## What a Project Folder Looks Like

```
projects/
  ODOT_Highway_Plans/
    project.json              ← Project metadata + config overrides
    label_registry.json       ← Which classes this project uses
    corrections.db            ← Training data, detections, corrections
    element_classifier.pkl    ← Trained Stage-1 model (created by training)
    title_subtype_classifier.pkl  ← Stage-2 model (created by training)
    training_data.jsonl       ← Exported training data (created by training)
    training_data_stage2.jsonl
    drift_stats.json          ← Drift reference (created by training)
    snapshots/                ← Safety backups
```

Nothing in this folder exists at creation time except `project.json` and `label_registry.json`. Everything else gets created naturally by the pipeline and training workflow.

---

## `project.json` Schema

This file stores project metadata and any config overrides that differ from defaults. It is intentionally minimal — the goal is to capture project-level decisions, not to duplicate every config field.

```json
{
    "version": "1.0",
    "name": "ODOT Highway Plans",
    "description": "Oregon DOT highway plan sets — civil, structural, traffic",
    "created_at": "2026-03-30T18:00:00Z",

    "config_overrides": {
        "ml_retrain_threshold": 30,
        "ml_hierarchical_enabled": true,
        "ml_relabel_confidence": 0.75
    }
}
```

**Why `config_overrides` and not a full config dump:** The full `GroupingConfig` has ~60 fields. Most are fine at defaults. Storing only the overrides means:
- You see immediately what's special about this project
- Default improvements in future app versions automatically apply
- The file stays readable and editable by hand

When loading a project, the app starts with a fresh `GroupingConfig()`, applies overrides from `project.json`, then repoints all paths to the project folder. The override layer sits on top of defaults — it doesn't replace them.

---

## Implementation — File-by-File Breakdown

### Change 1: New Module — `src/plancheck/config/project.py`

**What it does:** Pure-logic module for project folder operations. No GUI code, no tkinter.

**Functions:**

#### `create_project(project_dir, name, label_defs, config_overrides=None)`

Creates a new project folder with its initial files.

Steps:
1. Create the directory (and `snapshots/` subdirectory)
2. Write `project.json` with name, timestamp, config overrides
3. Write `label_registry.json` with the chosen label definitions
4. Create an empty `corrections.db` by instantiating `CorrectionStore(project_dir / "corrections.db")` and closing it — this ensures the schema is initialized

Returns the `Path` to the project directory.

#### `load_project(project_dir) -> dict`

Reads `project.json` and returns the parsed dict. Validates version and required fields. Raises `FileNotFoundError` if `project.json` is missing (not a valid project folder).

#### `build_project_config(project_dir, overrides=None) -> GroupingConfig`

The key function. Builds a `GroupingConfig` with all paths pointing at the project folder.

Steps:
1. Start with default `GroupingConfig()`
2. Override these path fields to point at project_dir:
   - `ml_model_path` → `{project_dir}/element_classifier.pkl`
   - `ml_stage2_model_path` → `{project_dir}/title_subtype_classifier.pkl`
   - `ml_gnn_model_path` → `{project_dir}/document_gnn.pt`
   - `ml_drift_stats_path` → `{project_dir}/drift_stats.json`
3. Apply any config overrides from `project.json`
4. Apply any additional runtime overrides passed in
5. Return the configured `GroupingConfig`

**Why this function matters:** This is the single place where "project directory" translates into "all paths point the right way." Every other part of the app just uses the resulting `GroupingConfig` as usual. No other code needs to know about projects.

#### `get_master_label_defs() -> list[dict]`

Returns the full set of available label definitions from the app's built-in default registry. Used by the New Project dialog to show a checklist of available classes.

This reads from a bundled default registry file (or returns the hardcoded defaults from `AnnotationTab.LABEL_COLORS` + the current `label_registry.json` in `data/`). The point is to have a master list that new projects pick from.

### Change 2: Add `project_dir` to `GuiState`

**Location:** `scripts/gui/gui.py`

**What changes:** `GuiState` gets two new attributes:

```python
self.project_dir: Path | None = None
self.project_meta: dict | None = None  # parsed project.json contents
```

And a new method:

#### `GuiState.set_project(project_dir)`

1. Loads `project.json` from the directory
2. Calls `build_project_config(project_dir, overrides)` to build a new `GroupingConfig`
3. Sets `self.config` to the new config
4. Sets `self.project_dir` and `self.project_meta`
5. Fires a new event: `"project_changed"`

This is the single entry point for switching projects. Everything downstream reacts to the event.

### Change 3: Modify `_label_registry_path()` in `label_registry.py`

**Current behavior:** Hardcoded path relative to file location → `data/label_registry.json`

**New behavior:** If `self.state.project_dir` is set, return `{project_dir}/label_registry.json`. Otherwise, fall back to the current hardcoded path for backward compatibility.

```python
def _label_registry_path(self) -> Path:
    project_dir = getattr(self.state, "project_dir", None)
    if project_dir:
        return project_dir / "label_registry.json"
    # Legacy fallback
    return (
        Path(__file__).resolve().parent.parent.parent.parent
        / "data"
        / "label_registry.json"
    )
```

This is a two-line change. Everything downstream (`_load_label_registry_json`, `_save_label_registry_json`, `_persist_element_type_to_registry`) automatically follows because they all call `_label_registry_path()`.

### Change 4: Modify `CorrectionStore()` calls to use project path

**The pattern today:** Several places create `CorrectionStore()` with no arguments, which defaults to `DEFAULT_CORRECTIONS_DB` (which points at `data/corrections.db`).

**Key locations:**
- `pdf_loader.py` → `_run_session_retrain_and_advance()` creates `CorrectionStore()`
- `model_training.py` → `_on_train_model()` creates `CorrectionStore()`
- `retrain_trigger.py` → `startup_check()` uses `DEFAULT_CORRECTIONS_DB`
- `annotation_store.py` → store initialization

**The fix pattern:** Each of these needs to resolve the DB path from the project directory when one is active. The cleanest approach: add a helper to `GuiState`:

```python
def db_path(self) -> Path:
    if self.project_dir:
        return self.project_dir / "corrections.db"
    return DEFAULT_CORRECTIONS_DB
```

Then replace `CorrectionStore()` calls with `CorrectionStore(self.state.db_path())` where `self.state` is available. In background threads where only the path is passed through, pass `self.state.db_path()` as a parameter.

**How many call sites:** Roughly 8-10 places in the GUI code create a `CorrectionStore`. Most already have access to `self.state`. The changes are mechanical — find, replace, test.

### Change 5: New Project Dialog

**Location:** `scripts/gui/gui.py` (or a new `scripts/gui/project_dialog.py`)

**Triggered by:** File menu → "New Project..." (or a toolbar button)

**Dialog flow:**

**Step 1 — Name and location:**
- Text field: Project name (e.g., "ODOT Highway Plans")
- Text field: Description (optional)
- Folder chooser: Parent directory (default: `projects/` next to the app)
- The project folder name is derived from the project name (slugified)

**Step 2 — Choose classes:**
- Checklist of all available label types from the master registry
- Each entry shows: label name, color swatch, short description
- All standard types checked by default
- User can uncheck types they don't need (e.g., uncheck `county` if not relevant)
- "Add Custom..." button to define a new type inline (name, color, description)

**Step 3 — Config overrides (optional, collapsible):**
- Retrain threshold (default: 50)
- ML confidence threshold (default: 0.8)
- These are the settings most likely to vary by project type
- Advanced users can edit `project.json` by hand for anything else

**On confirm:**
1. Call `create_project()` to build the folder
2. Call `state.set_project()` to activate it
3. Window title updates to show project name
4. Status bar: "Project created: ODOT Highway Plans"

**Implementation note:** This can be a simple `tk.Toplevel` dialog with a few frames. It doesn't need to be a multi-page wizard — a single dialog with sections works fine for the minimal scope.

### Change 6: Open Project Dialog

**Location:** Same file as New Project

**Triggered by:** File menu → "Open Project..."

**Flow:**
1. Standard folder chooser dialog
2. Validate: does the chosen folder contain `project.json`?
3. If yes: call `state.set_project()` to activate it
4. If no: error message — "Not a valid project folder (no project.json found)"
5. Window title updates, status bar confirms

**That's it.** Opening a project is a folder picker + one function call.

### Change 7: Wire into GUI menu bar

**Location:** `scripts/gui/gui.py` → `_build_ui()`

**Current state:** The GUI doesn't have a File menu. PDF selection is in the Pipeline tab.

**New additions:** Add a menu bar with:
- File → New Project...
- File → Open Project...
- File → separator
- File → (existing PDF open behavior can stay in Pipeline tab)

The menu bar is a small addition. If you prefer, the New/Open Project buttons can go in the Pipeline tab instead — wherever feels most natural for your workflow.

### Change 8: React to `project_changed` event

**Location:** Various tabs that need to refresh when the project changes.

**Tabs that need to subscribe to `project_changed`:**

| Tab | What it does on project change |
|-----|-------------------------------|
| Annotation (ML Trainer) | Reload label registry, reload classifier, refresh stats |
| Pipeline | Update config display to show project settings |
| Database | Repoint at new DB path |
| Diagnostics | Refresh ML runtime summary |

Each tab subscribes in its `__init__`:
```python
self.state.subscribe("project_changed", self._on_project_changed)
```

The `_on_project_changed` callback is a few lines per tab — mostly "reload from the new paths."

---

## Backward Compatibility — No Project Mode

If the user never creates or opens a project, the app works exactly as it does today. All paths default to `data/`. The `project_dir` stays `None`. Every code path that checks `self.state.project_dir` falls back to the existing behavior.

This is critical. The project system is additive, not a migration.

---

## The Label Class Workflow In Practice

Here's what the project setup looks like for classes:

**Scenario: New ODOT highway project**

You create a new project and the class checklist shows:

```
[✓] notes_column    — General Notes Column
[✓] header          — Section or View Header
[✓] abbreviations   — Abbreviations Table
[✓] legend          — Legend / Symbol Key
[✓] revision        — Revision Block
[✓] standard_detail — Standard Detail Reference
[✓] title_block     — Title Block
[✓] misc_title      — Miscellaneous Title / Label
[ ] county          — County (uncheck — not needed for this project)
```

You uncheck `county`, click Create. The project's `label_registry.json` only contains the 8 types you selected. When you open the ML Trainer for this project, only those 8 types appear in the label dropdown. The classifier only trains on those 8 classes. No noise from irrelevant types.

**Later, you realize you need a new type:** Open the label registry editor (already exists in your GUI), add `traffic_schedule` with a color and description. It saves to the project's `label_registry.json`. Next pipeline run picks it up.

**Starting a different project type:** Create a new project for commercial building plans. Pick a different set of classes. Maybe add `fire_rating_schedule` and `occupancy_table`. Each project has its own class vocabulary.

---

## What Stays The Same

- **Pipeline processing** — `run_document()` doesn't know or care about projects. It uses `GroupingConfig` which already has all the paths.
- **Training loop** — `train_classifier()` reads JSONL and writes a model. The paths come from the config. No project awareness needed.
- **CorrectionStore** — Already takes `db_path` as a constructor argument. Just pass the project's DB path.
- **The `data/` directory** — Still works as the default "no project" mode. Existing users are unaffected.
- **Config save/load** — Pipeline tab's config save/load still works. Project settings are the *defaults* for the config, not a replacement.

---

## Implementation Order

### Step 1: `src/plancheck/config/project.py`
Pure logic — `create_project()`, `load_project()`, `build_project_config()`. Test standalone: create a project folder, verify files exist, verify config paths resolve correctly. Zero GUI risk.

### Step 2: `GuiState.project_dir` + `set_project()` + `db_path()`
Add the state attributes and the project activation method. Test: call `set_project()` manually, verify `config.ml_model_path` points at the project folder.

### Step 3: `_label_registry_path()` update
Two-line change. Test: set a project, verify labels load from the project folder.

### Step 4: `CorrectionStore()` call sites
Mechanical find-and-replace. Pass `self.state.db_path()` instead of no argument. Test: create a project, run the pipeline, verify `corrections.db` appears in the project folder (not `data/`).

### Step 5: New Project dialog
The user-facing UI. Test end-to-end: create project → choose classes → verify folder structure → run pipeline → train model → everything lands in the project folder.

### Step 6: Open Project dialog
Simpler than New Project — just a folder picker. Test: close app, reopen, open existing project folder, verify everything loads.

### Step 7: Menu bar + `project_changed` event subscriptions
Wire the dialogs into the UI and make tabs react to project switches.

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Existing `data/` workflow breaks | Low | High | Backward compat: `project_dir = None` falls back to all existing defaults |
| User opens a folder that isn't a project | Medium | Low | Validate `project.json` exists before accepting |
| CorrectionStore call sites missed | Medium | Medium | Grep for `CorrectionStore()` with no args; each is a mechanical fix |
| Label registry out of sync with project | Low | Medium | Single path resolution in `_label_registry_path()` — one source of truth |
| Config overrides conflict with runtime changes | Low | Low | Overrides are applied at project load time only; runtime changes via Pipeline tab still work normally |
