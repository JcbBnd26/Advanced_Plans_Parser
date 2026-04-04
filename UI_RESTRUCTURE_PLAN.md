# UI Restructure — Implementation Plan

## Purpose

Transform the PlanCheck GUI from a developer workbench into a product-grade interface. The current UI intermingles business-facing features, backend developer tools, and future plan-grading features across a flat tab layout. This plan consolidates navigation into a standard menu bar, converts the Pipeline tab into a live monitoring station, and moves power-user/backend features into Settings dialogs.

## Guiding Principles

- **Each step is independently testable.** No step depends on a later step to verify correctness.
- **What is NOT changing is stated explicitly** alongside what is.
- **Backward compatibility first.** Every feature must remain accessible — we are moving things, not deleting them.
- **Organic feel.** The UI should feel alive during processing, not static. Progress is visual, not just text.

---

## Current State (What Exists Today)

### Tab Structure

| Tab | Category | Description |
|-----|----------|-------------|
| Pipeline | Business-facing | PDF selection, pipeline execution, config display |
| Annotation (ML Trainer) | Business-facing | Page-by-page review, corrections, bbox editing, parent-child linking |
| Database | Backend/developer | Direct view into corrections.db, detection rows |
| Diagnostics | Backend/developer | ML runtime summary, model stats, drift metrics |
| Adv ML Runtime | Backend/developer | Advanced ML configuration and runtime controls |
| Optional ML Features | Backend/developer | Toggle switches for experimental ML features |
| LLM Runtime | Backend/developer | LLM integration settings, model selection, token config |

### Menu Bar (Partial — Built During Project System Work)

The File menu currently contains:
- File → New Project...
- File → Open Project...
- File → Export Project...
- File → Import Project...
- File → Recent Projects → (submenu)

### Notable UI Elements

- **TOCR checkbox (#86):** Currently a simple on/off toggle. Needs to become a per-page progress bar.
- **Pipeline tab layout:** Static "configure and run" design. No live feedback once processing starts.

---

## Target State

### Menu Bar (Standard Layout)

```
File    Edit    View    Settings    Help
```

**File** (already partially built):
  - New Project...
  - Open Project...
  - Open Recent → (submenu)
  - ---
  - Export Project...
  - Import Project...
  - ---
  - Exit

**Edit** (future-ready, minimal for now):
  - Undo (wire to existing undo system in Annotation tab)
  - Redo (wire to existing redo system in Annotation tab)
  - ---
  - Preferences... (alias for Settings → General, if desired)

**View** (tab/panel visibility controls):
  - Pipeline Monitor (shows/focuses the Pipeline tab)
  - ML Trainer (shows/focuses the Annotation tab)
  - ---
  - Database Inspector (shows/hides Database tab)
  - Diagnostics Panel (shows/hides Diagnostics tab)
  - ---
  - Status Bar (toggle visibility)

**Settings** (pop-up dialogs, NOT tabs):
  - General... (app-level preferences — theme, default paths, startup behavior)
  - ML Runtime... (absorbs current "Adv ML Runtime" tab content)
  - ML Features... (absorbs current "Optional ML Features" tab content)
  - LLM Configuration... (absorbs current "LLM Runtime" tab content)
  - Pipeline Defaults... (default pipeline config for new projects)
  - Label Registry... (existing label editor, promoted to menu access)

**Help** (standard):
  - About PlanCheck
  - Documentation (placeholder, can link to local docs later)

### Tab Structure (Post-Restructure)

Only business-facing tabs remain as tabs:

| Tab | Status |
|-----|--------|
| Pipeline (now "Monitor") | **Kept** — redesigned as live monitoring station |
| Annotation (ML Trainer) | **Kept** — unchanged in this plan |
| Database | **Kept** — accessible via View menu, hidden by default for new users |
| Diagnostics | **Kept** — accessible via View menu, hidden by default for new users |
| Adv ML Runtime | **Removed as tab** — becomes Settings → ML Runtime... dialog |
| Optional ML Features | **Removed as tab** — becomes Settings → ML Features... dialog |
| LLM Runtime | **Removed as tab** — becomes Settings → LLM Configuration... dialog |

### What Is NOT Changing

- **Annotation tab internals** — No changes to bbox editing, parent-child linking, corrections workflow, mixin architecture, or event handling. This plan does not touch `event_handler.py`, `canvas_renderer.py`, or any annotation mixin.
- **Pipeline processing logic** — `run_document()`, stage sequence, `GroupingConfig`, and all backend processing are untouched. We are changing how the *UI displays* pipeline activity, not how the pipeline *runs*.
- **Project system** — `project.py`, `project_dialog.py`, `GuiState.set_project()`, `db_path()` — all stay as-is. The File menu items that already exist simply move into the expanded menu bar.
- **Online learning / training session system** — `micro_retrain.py`, `page_repredict.py`, session lifecycle — all untouched.
- **CorrectionStore, label registry, classifier, training loop** — No backend changes.

---

## Phase 1 — Feature Audit and Classification

### Objective
Produce a definitive map of every UI element, which category it belongs to, and where it's going.

### Steps

**1.1 — Inventory every tab's contents**

Walk through each tab in the running app. For every widget (button, checkbox, dropdown, text field, label, panel), record:
- Widget name / label text
- What it does (one sentence)
- Category: `business`, `backend`, or `future`
- Current location (tab name + approximate position)
- Proposed new home

Deliver this as a table in a markdown file: `docs/ui_audit.md`

**1.2 — Identify cross-tab dependencies**

Some widgets in backend tabs may trigger actions that update business tabs (e.g., ML Runtime settings affect Annotation tab behavior). Document every case where a Settings dialog will need to fire an event that a tab listens to. Use the existing `GuiState.subscribe()` pattern to map these.

**1.3 — Flag "future / plan grading" features**

Any feature that exists for plan grading purposes (not yet functional) should be tagged. These stay in the codebase but should be hidden from the default UI. They can be shown via a developer toggle or a "Show Advanced" option in Settings.

### Test Criteria
- `docs/ui_audit.md` exists and covers every widget
- Every widget has a category and proposed location
- Cross-tab dependencies are documented with event names

### What Is NOT Changing In This Phase
- No code changes. This is a documentation-only phase.

---

## Phase 2 — Menu Bar Skeleton

### Objective
Build the full standard menu bar (File, Edit, View, Settings, Help) with all items wired to placeholder callbacks. Existing File menu items migrate into the new structure.

### Steps

**2.1 — Extend the menu bar in `scripts/gui/gui.py`**

The File menu already exists from the project system work. Extend `_build_ui()` to add Edit, View, Settings, and Help menus. All new menu items initially call a placeholder function that prints to the status bar: `"[Menu Item Name] — not yet implemented"`.

**Key detail:** The existing File menu items (New Project, Open Project, Export, Import, Recent) must be preserved exactly. Do not rewrite them — add the new menus alongside the existing File menu.

**2.2 — Wire Edit → Undo/Redo to existing annotation system**

The Annotation tab already has undo/redo logic. The Edit menu items should call into the same functions. If the Annotation tab is not active/focused, Undo/Redo should be disabled (grayed out). Use `menu.entryconfig()` to toggle enabled state based on active tab.

**2.3 — Wire View menu to tab visibility**

Each View menu item should toggle the corresponding tab's visibility in the notebook widget. Use `notebook.hide(tab_index)` and `notebook.add(tab_widget)` to show/hide tabs. Store visibility state in `GuiState` so it persists across sessions (save to project config or app config).

**Implementation note:** `ttk.Notebook` doesn't have a built-in hide/show per tab. The pattern is to call `notebook.forget(tab_id)` to hide and `notebook.insert(position, tab_widget)` to re-show. Track the original positions so tabs re-appear in the correct order.

### Test Criteria
- Menu bar displays all five menus (File, Edit, View, Settings, Help)
- All existing File menu items still work exactly as before
- Clicking any new menu item produces a status bar message (placeholder behavior is fine)
- View menu can hide and re-show the Database and Diagnostics tabs
- Undo/Redo in Edit menu calls into annotation tab when it's active

### What Is NOT Changing In This Phase
- Tab contents — no tab internals are modified
- No tabs are removed yet — all tabs remain, the View menu just adds hide/show capability
- Pipeline tab is not redesigned yet

---

## Phase 3 — Settings Dialogs (Tab-to-Dialog Migration)

### Objective
Move Adv ML Runtime, Optional ML Features, and LLM Runtime from tabs into standalone `tk.Toplevel` dialog windows launched from the Settings menu.

### Steps

**3.1 — Create `scripts/gui/settings_dialogs.py`**

New module containing three dialog classes:
- `MLRuntimeDialog(tk.Toplevel)` — absorbs Adv ML Runtime tab content
- `MLFeaturesDialog(tk.Toplevel)` — absorbs Optional ML Features tab content
- `LLMConfigDialog(tk.Toplevel)` — absorbs LLM Runtime tab content

Each dialog:
- Opens as a modal window (grabs focus, blocks main window interaction)
- Has OK / Cancel / Apply buttons at the bottom
- Reads current values from `GuiState` or project config on open
- Writes changes back on OK/Apply
- Fires appropriate `GuiState` events so subscribing tabs react (e.g., `ml_config_changed`, `llm_config_changed`)

**3.2 — Extract widget code from tab classes into dialog classes**

This is a **move, not a rewrite.** The widgets (checkboxes, dropdowns, sliders, text fields) that currently live in the tab classes get moved into the corresponding dialog class. The layout may need minor adjustment (tabs are horizontal-filling; dialogs are fixed-size windows), but the widget creation code and variable bindings should transfer directly.

**Trap to watch for:** If the tab classes store widget references as `self.some_widget` and other code accesses them (e.g., `self.tabs['adv_ml'].some_widget`), those references will break after migration. Audit all cross-references before moving. The correct pattern post-migration is to access values through `GuiState`, not through widget references.

**3.3 — Wire Settings menu items to dialog launchers**

Replace the placeholder callbacks from Phase 2 with actual dialog launches:
```
Settings → ML Runtime...       → opens MLRuntimeDialog
Settings → ML Features...      → opens MLFeaturesDialog
Settings → LLM Configuration...→ opens LLMConfigDialog
```

**3.4 — Remove the three migrated tabs from the notebook**

After confirming all functionality works through dialogs, remove the tab registrations from `gui.py`. The tab classes can remain in the codebase (commented out or in a `legacy/` folder) until you're confident nothing was missed.

**3.5 — Add General Settings dialog**

Create a `GeneralSettingsDialog(tk.Toplevel)` for app-level preferences:
- Default data directory
- Startup behavior (open last project, start blank)
- UI preferences (tab visibility defaults)
- Developer mode toggle (shows/hides Database and Diagnostics tabs by default)

This dialog is new — it doesn't migrate from an existing tab.

### Test Criteria
- Each dialog opens from the Settings menu and displays all widgets from the former tab
- Changing a value in a dialog and clicking OK correctly updates the underlying config
- Tabs that depend on these settings (Annotation, Pipeline) react to changes via events
- The three former tabs no longer appear in the notebook
- All settings persist across app restarts (saved to project config or app config file)

### What Is NOT Changing In This Phase
- Pipeline tab — still in its current form (monitoring redesign is Phase 4)
- Annotation tab — no changes
- Database and Diagnostics tabs — still present, just now toggle-able via View menu
- Backend logic for ML Runtime, ML Features, LLM — only the UI container changes (tab → dialog), not the underlying functionality

---

## Phase 4 — Pipeline Monitoring Station

### Objective
Redesign the Pipeline tab from a static "configure and run" layout into a live process monitoring dashboard. The tab becomes the app's front page — the first thing the user sees and interacts with.

### Steps

**4.1 — Define the monitoring layout**

The new Pipeline tab (consider renaming to "Monitor" in the notebook tab label) has three zones:

**Zone A — Document Status (top)**
- Currently loaded PDF name and page count
- Project name (if active)
- Quick-access buttons: Load PDF, Run Pipeline, Start Training Session

**Zone B — Stage Progress (middle, primary visual)**
- Horizontal pipeline stage indicators: Ingest → TOCR → Grouping → Analysis → Checks
- Each stage shows: status icon (pending/running/complete/error), elapsed time, item count
- Below the stage indicators: the **TOCR per-page progress bar**

**Zone C — Activity Log (bottom)**
- Scrolling log of pipeline events (replaces or supplements the existing status bar)
- Each entry: timestamp, stage name, message
- Color-coded: info (default), warning (yellow), error (red), success (green)

**4.2 — Build the TOCR per-page progress bar**

This replaces the current TOCR checkbox (#86 in the notes).

**Design:**
- A horizontal row of small square boxes, one per sheet in the loaded PDF
- Each box starts empty/gray
- As TOCR completes for a page, that box fills with color (e.g., green for success, yellow for warning, red for error)
- Hovering over a box shows a tooltip: page number, TOCR duration, word count extracted
- Clicking a box navigates to that page in the Annotation tab

**Implementation approach:**
- Use a `tk.Canvas` widget with small rectangles drawn per page
- Bind `<Enter>` and `<Leave>` events on each rectangle for tooltips
- Bind `<Button-1>` for click-to-navigate
- Pipeline worker thread emits a `tocr_page_complete` event via `GuiState` with page number and status
- The progress bar widget subscribes to this event and updates the corresponding box

**Scaling consideration:** For very large documents (100+ pages), boxes may need to shrink or wrap to a second row. Set a minimum box size (e.g., 8x8 pixels) and calculate layout dynamically based on page count and available width.

**4.3 — Wire pipeline stages to the monitoring display**

The pipeline worker thread (in `worker.py` or wherever `run_document()` is called from) needs to emit events for each stage transition:

| Event | Payload | When Fired |
|-------|---------|------------|
| `pipeline_stage_started` | `{stage: str, page: int}` | A stage begins processing |
| `pipeline_stage_complete` | `{stage: str, page: int, duration: float}` | A stage finishes |
| `pipeline_stage_error` | `{stage: str, page: int, error: str}` | A stage fails |
| `tocr_page_complete` | `{page: int, status: str, word_count: int, duration: float}` | TOCR finishes one page |
| `pipeline_complete` | `{total_duration: float, pages: int}` | Entire pipeline done |

**These events flow through `GuiState.subscribe()` / `GuiState.notify()`**, the same pub/sub pattern already used for `project_changed`. The monitoring widgets subscribe to these events and update themselves.

**Trap:** Pipeline runs on a background thread. GUI updates must happen on the main thread. Use `root.after(0, callback)` or the existing thread-safe event queue pattern from the online learning system to marshal updates to the main thread.

**4.4 — Relocate configuration controls**

Pipeline configuration (stage toggles, threshold sliders, output path selectors) currently lives in the Pipeline tab. With the tab becoming a monitoring station, config controls move to:
- **Settings → Pipeline Defaults...** for project-level defaults
- **A collapsible "Config" panel** within the monitoring tab for per-run overrides (collapsed by default, expand via a "Configure" button or disclosure triangle)

This keeps the monitoring tab clean while still allowing quick config changes before a run.

**4.5 — Add the "organic feel" animation**

Once the monitoring widgets are functional, add visual polish:
- Stage indicators pulse gently while active (subtle color oscillation, not distracting)
- TOCR progress boxes fill with a brief fade rather than instant color change
- Activity log entries slide in rather than appearing abruptly
- A subtle progress bar at the very top of the tab shows overall completion percentage

**Implementation note:** Tkinter animations use `root.after()` for scheduling. Keep frame rates low (10-15 fps) to avoid CPU overhead. The goal is "alive," not "flashy."

### Test Criteria
- Pipeline tab displays document status, stage progress, and activity log
- TOCR per-page progress bar shows one box per page, fills as pages complete
- Clicking a progress box navigates to that page in the Annotation tab
- Stage indicators update in real time during pipeline processing
- Configuration is accessible via collapsible panel or Settings menu
- No regressions in pipeline processing — backend is untouched, only UI reporting changes

### What Is NOT Changing In This Phase
- Pipeline processing backend — `run_document()`, stage sequence, all processing logic
- Annotation tab — no changes
- Online learning system — works as before, monitoring just shows its activity
- Project system — unchanged

---

## Phase 5 — Polish and Visual Cohesion

### Objective
With structure and monitoring in place, refine the visual language across the entire app for consistency and that "organic" feel.

### Steps

**5.1 — Establish a visual style guide**

Before touching any widgets, document the target aesthetic:
- Color palette (primary, secondary, accent, status colors for success/warning/error)
- Font choices and sizes (header, body, monospace for data)
- Spacing and padding standards
- Border and shadow treatment
- Animation timing curves and durations

Save as `docs/style_guide.md`. All subsequent visual work references this document.

**5.2 — Consistent widget styling via ttk themes**

Create a custom `ttk.Style` configuration that applies uniformly across all tabs and dialogs. This is a single setup call in `gui.py` that themes buttons, labels, frames, notebooks, and treeviews consistently.

**5.3 — Status bar redesign**

The status bar at the bottom of the main window should show:
- Left: Current project name and active document
- Center: Last pipeline/training action and result
- Right: ML model status (trained/untrained, example count, last F1 score)

**5.4 — Tooltip and feedback consistency**

Audit all tooltips, error messages, and status messages for consistent tone and formatting. Construction industry users expect clear, direct language — no jargon from the ML world unless it's genuinely useful.

**5.5 — Keyboard shortcuts**

Wire standard shortcuts:
- Ctrl+Z / Ctrl+Y — Undo/Redo (already exists in annotation, now globally accessible)
- Ctrl+N — New Project
- Ctrl+O — Open Project
- Ctrl+S — Save (context-dependent: save corrections, save config, etc.)
- F5 — Run Pipeline
- Ctrl+, — Open Settings (macOS convention, useful universally)

### Test Criteria
- Visual style guide document exists and is referenced
- All widgets across tabs and dialogs share consistent styling
- Status bar shows relevant, real-time information
- Keyboard shortcuts work from any tab

### What Is NOT Changing In This Phase
- All functionality — this phase is purely cosmetic and UX polish
- No backend changes of any kind
- Tab structure finalized in prior phases

---

## Implementation Order Summary

| Phase | Focus | Risk | Dependencies |
|-------|-------|------|-------------|
| 1 | Feature audit (docs only) | None | None |
| 2 | Menu bar skeleton | Low | None (existing File menu preserved) |
| 3 | Settings dialogs (tab migration) | Medium | Phase 2 (menu items exist to wire to) |
| 4 | Pipeline monitoring station | Medium | Phase 3 (config controls relocated) |
| 5 | Visual polish | Low | Phases 2-4 (everything in final position) |

---

## Traps and Edge Cases

**Tab removal ordering:** Don't remove a tab from the notebook until the replacement dialog is confirmed working. Run both in parallel during development — the tab AND the dialog — then remove the tab once parity is verified.

**Widget reference breakage (Phase 3):** Any code that reaches across tabs to read a widget value (e.g., `self.parent.tabs['adv_ml'].threshold_var.get()`) will break when that tab becomes a dialog. Grep the entire `scripts/gui/` directory for cross-tab references before migrating. The fix is to route all shared state through `GuiState` attributes, not widget references.

**Thread safety in monitoring (Phase 4):** Pipeline events fire from background threads. Every GUI update triggered by these events must be marshaled to the main thread via `root.after(0, callback)`. Failure to do this causes intermittent crashes that are extremely hard to reproduce. The online learning system already handles this correctly — follow the same pattern.

**Notebook tab position tracking (Phase 2):** When using `notebook.forget()` to hide tabs, the tab's position index changes. If you later `notebook.insert()` to re-show it, you need the original position, not the current one. Store original positions at startup in a dict: `{tab_id: original_index}`.

**Settings persistence:** Dialogs that change settings need to save those settings somewhere that survives app restart. Options: (a) write to `project.json` for project-scoped settings, (b) write to an app-level `config.json` in the app directory for global settings. Don't mix the two — project settings go in the project, app settings go in the app.

**TOCR progress bar scaling:** Test with both small (5-page) and large (200-page) documents. The box layout must handle both gracefully. Consider a minimum box size of 6x6px and a maximum row width equal to the tab width, with automatic row wrapping.

---

## Open Questions for Owner Review

1. **Tab renaming:** Should "Pipeline" become "Monitor" in the tab label? The role is changing from "configure and run" to "watch what's happening."

2. **Database and Diagnostics default visibility:** Should these tabs be hidden by default for new users (accessible via View menu), or visible by default with the option to hide?

3. **Developer mode toggle:** Should there be a single "Developer Mode" switch in General Settings that shows/hides all backend features at once, or should each backend feature be individually toggleable?

4. **Edit menu scope:** Beyond Undo/Redo, are there other edit operations that should be in the Edit menu? (e.g., Delete Selected, Select All, Copy Detection Data)

5. **Help menu content:** Is there existing documentation to link to, or should Help → Documentation be a placeholder for now?
