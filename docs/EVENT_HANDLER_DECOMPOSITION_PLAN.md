# Event Handler Decomposition Plan

## The Tool Pattern — A Construction Analogy

Right now, `EventHandlerMixin` is one superintendent trying to do every trade. When someone
hands him a mouse click, he checks 15 Post-it notes to figure out what trade he's working:
"Am I dragging a handle? Moving a box? Drawing a lasso? Linking a parent to a child?"

The Tool pattern gives each trade its own crew chief. Only one crew works at a time, and that
crew gets all the events. When the framing crew finishes, the superintendent swaps in the
next crew. Clean handoffs, no flag soup.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     AnnotationTab                            │
│  (thin coordinator — owns canvas, inspector, key bindings)   │
├──────────┬──────────┬───────────────────────────────────────┤
│          │          │                                        │
│  Canvas  │Inspector │   Services (shared by all tools)       │
│  Events  │  Panel   │   ┌─────────────┐ ┌──────────────┐    │
│    │     │          │   │ UndoManager │ │  Clipboard   │    │
│    ▼     │          │   └─────────────┘ └──────────────┘    │
│ ToolManager ────────┤   ┌─────────────┐ ┌──────────────┐    │
│    │                │   │  Selection  │ │BoxOperations │    │
│    ▼                │   │   State     │ │(accept,del)  │    │
│ Active Tool         │   └─────────────┘ └──────────────┘    │
│ ┌────────┐          │                                        │
│ │SelectTl│ (default)│                                        │
│ │MoveTool│          │                                        │
│ │ResizeTl│          │                                        │
│ │LassoTl │          │                                        │
│ │DrawTool│          │                                        │
│ │LinkTool│ (NEW)    │                                        │
│ └────────┘          │                                        │
└─────────────────────────────────────────────────────────────┘
```

**Key insight:** This is a migration, not a rewrite. We build the ToolManager alongside the
existing EventHandlerMixin, extract one tool at a time, and at each step the system works —
some events go through the tool manager, others still go through the old mixin. Like
renovating a house one room at a time while still living in it.

---

## Part 1: The Tool Interface (BaseTool)

Every tool implements the same interface. The ToolManager doesn't need to know what a
SelectTool or LinkTool *does* — it just knows they all respond to the same set of events.

This is the clever part: by standardizing the interface, adding a new tool later (a ruler
tool, a measurement tool, a highlight tool) costs almost nothing. You write the class, plug
it in, done.

### BaseTool Methods

| Method | When It Fires | What It Means |
|--------|--------------|---------------|
| `on_click(pdf_x, pdf_y, shift, ctrl)` | Left mouse down | User clicked on the canvas |
| `on_drag(pdf_x, pdf_y)` | Left mouse held + moving | User is dragging |
| `on_release(pdf_x, pdf_y)` | Left mouse up | User released |
| `on_right_click(pdf_x, pdf_y, event)` | Right mouse down | Context action |
| `on_motion(pdf_x, pdf_y)` | Mouse movement (no button) | Hover / rubber band |
| `on_key(key, event)` | Keyboard press | Tool-specific shortcut |
| `enter()` | Tool becomes active | Set cursor, show banner |
| `exit()` | Tool deactivates | Clean up state, reset cursor |

Every method has a default no-op implementation, so a tool only overrides what it cares about.
DrawTool doesn't care about `on_motion`, so it doesn't implement it. LinkTool cares deeply
about `on_motion` (rubber band line), so it does.

---

## Part 2: The ToolManager

About ~80–100 lines. Its job is simple:

1. Hold a reference to the active tool
2. Receive ALL canvas events from AnnotationTab
3. Forward them to the active tool
4. Provide `switch_to(tool_name)` and `revert_to_default()`

### Important behavior: Automatic reversion

When MoveTool finishes a drag, it tells the ToolManager "I'm done." The ToolManager
switches back to SelectTool automatically. The user never manually selects a tool — tools
activate and deactivate based on what the user does. This is why it feels seamless.

### Coordinate conversion happens ONCE

Currently, every handler repeats `cx = self._canvas.canvasx(event.x)` and the
`pdf_x = cx / eff` conversion. The ToolManager does this once and passes `(pdf_x, pdf_y)`
to the tool. DRY — and every tool gets clean PDF coordinates without boilerplate.

---

## Part 3: Services (Shared Infrastructure)

These are NOT tools — they're services that any tool can call. They live as standalone
classes or small mixins.

### SelectionState (~60 lines)

Owns the shared selection state that multiple tools read/write:

- `selected_box` — the currently selected CanvasBox
- `multi_selected` — the shift-click set
- `selected_word_rids` — word overlay selections

Methods: `select_box()`, `deselect()`, `toggle_multi_select()`, `clear_multi_select()`,
`set_word_selected()`, `toggle_word_selected()`, `clear_word_selection()`,
`update_multi_label()`, `update_word_selection_label()`

**Why separate?** SelectTool sets the selection. MoveTool reads it to know what to move.
ResizeTool reads it to know what to resize. LinkTool reads it to know what to group.
The selection state is shared context, not owned by any one tool.

### UndoManager (~220 lines)

Extracted directly from the current undo/redo code. Owns:

- `_undo_stack` / `_redo_stack`
- `push(action, cbox, extra)` — formerly `_push_undo`
- `undo()` — the full undo logic (relabel, reshape, delete, dismiss, accept, merge)
- `redo()` — the full redo logic

Tools call `undo_manager.push(...)` when they make changes. The annotation tab binds
Ctrl+Z / Ctrl+Y to `undo_manager.undo()` / `undo_manager.redo()`.

### Clipboard (~60 lines)

- `copy_box(cbox)` — store template
- `paste_box(pdf_x, pdf_y)` — create from template
- `has_content` — property for menu state

### BoxOperations (~200 lines)

The inspector button actions that aren't tied to any specific tool:

- `accept(targets)` — accept detection(s)
- `relabel(targets, new_label)` — relabel detection(s)
- `delete(targets)` — reject detection(s)
- `dismiss(targets)` — dismiss detection(s)
- `auto_refresh_text(cbox)` — re-extract text after move/reshape
- `rescan_text(cbox)` — manual text re-extraction

These are called by buttons and keyboard shortcuts, not by mouse interaction.
They use `undo_manager.push(...)` internally.

### MLPredictor (~100 lines)

The model suggestion logic:

- `_get_configured_classifier()`
- `_predict_model_suggestion()` / `_predict_model_suggestion_details()`
- `_format_stage2_candidates()`

Currently called by `_select_box()` and `_merge_words_into_detection()`. Stays a service.

---

## Part 4: The Tools — Method-by-Method Mapping

### SelectTool (~120 lines) — THE DEFAULT

This is where the user spends most of their time. Click a box, shift-click for multi,
click empty space to deselect. The key subtlety: SelectTool also **detects** when the user
starts a drag or grabs a handle, and tells the ToolManager to switch tools.

**Migrated from EventHandlerMixin:**

| Current Method | Destination | Notes |
|----------------|-------------|-------|
| `_on_canvas_click` (lines 176–248) | `SelectTool.on_click()` | The "router" — detects click target |
| `_toggle_multi_select` (line 1640) | Uses `SelectionState` | |
| `_clear_multi_select` (line 1649) | Uses `SelectionState` | |
| `_key_prev_box` (line 1379) | `SelectTool.on_key('Left')` | |
| `_key_next_box` (line 1395) | `SelectTool.on_key('Right')` | |
| `_key_select_all` (line 1699) | `SelectTool.on_key('Ctrl+A')` | |
| `_key_deselect` (line 1375) | `SelectTool.on_key('Escape')` | |

**The important behavior inside `on_click`:**
When SelectTool detects a handle hit → `tool_manager.switch_to('resize')`
When SelectTool detects a box click → prepares for move, then `tool_manager.switch_to('move')`
When SelectTool detects empty space click → `tool_manager.switch_to('lasso')`

SelectTool is the traffic cop. It figures out what the user INTENDS, then hands off.

### MoveTool (~100 lines)

Activated by SelectTool when user clicks a box and starts dragging.

| Current Method | Destination | Notes |
|----------------|-------------|-------|
| `_do_move_drag` (line 726) | `MoveTool.on_drag()` | Live position update |
| `_finalize_move` (line 766) | `MoveTool.on_release()` | Save correction |
| `_schedule_throttled_drag` (line 703) | `MoveTool` internal | Drag throttle |
| `_flush_throttled_drag` (line 718) | `MoveTool` internal | Drag throttle |

**State it owns:** `move_start_pdf`, `move_orig_bbox`, `move_orig_polygon`

**On completion:** Calls `tool_manager.revert_to_default()` → back to SelectTool.

### ResizeTool (~100 lines)

Activated by SelectTool when user grabs a resize handle.

| Current Method | Destination | Notes |
|----------------|-------------|-------|
| `_do_handle_drag` (line 802) | `ResizeTool.on_drag()` | Live reshape |
| `_finalize_reshape` (line 876) | `ResizeTool.on_release()` | Save correction |
| Drag throttle | Shared with MoveTool or duplicated (small) | |

**State it owns:** `drag_handle`, `drag_orig_bbox`, `drag_orig_polygon`

**On completion:** `tool_manager.revert_to_default()` → back to SelectTool.

### LassoTool (~80 lines)

Activated by SelectTool when user clicks empty space and drags.

| Current Method | Destination | Notes |
|----------------|-------------|-------|
| `_finalize_lasso` (line 2208) | `LassoTool.on_release()` | Select intersecting boxes |
| Lasso rect drawing in `_on_canvas_drag` | `LassoTool.on_drag()` | Dashed rectangle |

**State it owns:** `lasso_start`, `lasso_rect_id`, `lasso_word`

**On completion:** `tool_manager.revert_to_default()` → back to SelectTool.

### DrawTool (~180 lines)

Activated by Ctrl+Click (draw new box).

| Current Method | Destination | Notes |
|----------------|-------------|-------|
| `_on_word_click` (line 147) | `DrawTool.on_click()` | Also handles word overlay clicks |
| Draw rect in `_on_canvas_drag` | `DrawTool.on_drag()` | Dashed preview rect |
| `_finalize_add` (line 915) | `DrawTool.on_release()` | Type dialog + save |

**State it owns:** `draw_start`, `draw_rect_id`

**On completion:** `tool_manager.revert_to_default()` → back to SelectTool.

**Note:** The word-click / Ctrl+click path is slightly overloaded — it starts a new box
draw if the word overlay is OFF, but selects words if the overlay is ON. DrawTool handles
both branches.

### LinkTool (~200 lines) — NEW

The parent-child wire-drawing interaction. This is the feature that motivated the whole
refactor. It has its own internal state machine:

```
  IDLE ──right-click "Create Parent"──► PARENT_READY
    ▲                                        │
    │                                   left-click parent
    │                                        │
    │                                        ▼
    └──── Escape ◄──────────────────── LINKING
                                         │     ▲
                                  right-click   │ (automatic)
                                   on child     │
                                         │      │
                                         ▼      │
                                     CONNECTED ─┘
```

**State it owns:**
- `_link_state` — enum: PARENT_READY, LINKING
- `_link_parent` — the CanvasBox designated as parent
- `_rubber_band_id` — canvas line item following cursor
- `_group_id` — the group being built (created after naming dialog)

**Behavior by state:**

| State | Left Click | Right Click | Motion | Escape |
|-------|-----------|-------------|--------|--------|
| PARENT_READY | On parent → LINKING | — | Highlight valid targets | → IDLE |
| LINKING | — | On valid child → CONNECTED | Rubber band line | → IDLE |
| CONNECTED | — | — | — | (auto → LINKING) |

**Key interactions with existing code:**
- Calls `_store.create_group()` (same as current `_create_group`)
- Calls `_store.add_to_group()` (same as current `_add_children_to_group`)
- Calls `_draw_group_links()` to render permanent dashed lines
- Uses `SelectionState` to read the clicked box

**Visual feedback:**
- PARENT_READY: Parent box gets a glowing thick border (4px, pulsing or bright color)
- LINKING: Cursor → crosshair; rubber band line (solid, semi-transparent) from parent center to cursor; valid targets get highlighted border on hover; invalid targets (already grouped) stay dim
- CONNECTED: Flash confirmation, permanent dashed line drawn, stay in LINKING

### MergeAction (~180 lines) — NOT a tool, stays as a callable action

| Current Method | Destination | Notes |
|----------------|-------------|-------|
| `_on_merge` (line 1716) | `MergeAction.execute()` | |
| `_merge_words_into_detection` (line 1875) | `MergeAction.merge_words()` | |
| `_key_merge` (line 2040) | Key binding → `MergeAction.execute()` | |

Merge is triggered by a keyboard shortcut (M) or button, not by mouse interaction.
It reads from SelectionState and WordSelectionState, operates, and returns.
It's an action, not a tool.

### GroupActions (~180 lines) — NOT a tool, stays as a callable action

| Current Method | Destination | Notes |
|----------------|-------------|-------|
| `_create_group` (line 379) | `GroupActions.create()` | |
| `_on_create_group` (line 441) | Inspector button → `GroupActions.create()` | |
| `_on_add_to_group` (line 446) | Inspector button → `GroupActions.add_children()` | |
| `_add_children_to_group` (line 464) | `GroupActions.add_children()` | |
| `_remove_from_group` (line 497) | `GroupActions.remove()` | |
| `_on_remove_from_group` (line 492) | Inspector button → `GroupActions.remove()` | |
| `_update_group_inspector` (line 535) | `GroupActions.update_inspector()` | |

### LinkColumnAction (~130 lines) — NOT a tool, stays as a callable action

| Current Method | Destination | Notes |
|----------------|-------------|-------|
| `_on_link_column` (line 2075) | `LinkColumnAction.execute()` | |
| `_key_link_column` (line 2067) | Key binding → `LinkColumnAction.execute()` | |

### NavigationMixin (~50 lines) — Already independent

| Current Method | Destination | Notes |
|----------------|-------------|-------|
| `_on_mousewheel` (line 1318) | Stays as mixin or standalone | |
| `_on_shift_mousewheel` (line 1322) | Same | |
| `_on_pan_start/motion/end` (lines 1326–1340) | Same | |
| `_key_zoom_in/out/fit` (lines 1411–1423) | Same | |

Navigation (pan, zoom, scroll) always works regardless of active tool. These bypass the
ToolManager entirely and stay bound directly to the canvas.

### HoverTooltip (~90 lines) — Standalone

| Current Method | Destination | Notes |
|----------------|-------------|-------|
| `_on_canvas_motion` (line 2268) | `HoverTooltip.on_motion()` | |
| `_show_hover_tooltip` (line 2310) | `HoverTooltip.show()` | |
| `_hide_hover_tooltip` (line 2346) | `HoverTooltip.hide()` | |

Hover tooltips are always active, regardless of tool. BUT — when LinkTool is active, hover
needs to show link-target highlighting instead of (or in addition to) tooltips. The
ToolManager can give the active tool first crack at motion events, and if the tool doesn't
consume it, HoverTooltip gets it.

---

## Part 5: File Layout After Decomposition

```
scripts/gui/
├── tools/
│   ├── __init__.py              (~20 lines)  — exports
│   ├── base.py                  (~60 lines)  — BaseTool ABC + ToolState enum
│   ├── tool_manager.py          (~100 lines) — ToolManager class
│   ├── select_tool.py           (~120 lines) — SelectTool
│   ├── move_tool.py             (~100 lines) — MoveTool
│   ├── resize_tool.py           (~100 lines) — ResizeTool
│   ├── lasso_tool.py            (~80 lines)  — LassoTool
│   ├── draw_tool.py             (~180 lines) — DrawTool
│   └── link_tool.py             (~200 lines) — LinkTool (NEW)
├── services/
│   ├── __init__.py              (~10 lines)
│   ├── selection_state.py       (~60 lines)
│   ├── undo_manager.py          (~220 lines)
│   ├── clipboard.py             (~60 lines)
│   ├── box_operations.py        (~200 lines)
│   └── ml_predictor.py          (~100 lines)
├── actions/
│   ├── __init__.py              (~10 lines)
│   ├── merge_action.py          (~180 lines)
│   ├── group_actions.py         (~180 lines)
│   └── link_column_action.py    (~130 lines)
├── hover_tooltip.py             (~90 lines)  — extracted from event_handler
├── navigation.py                (~50 lines)  — pan/zoom/scroll
├── event_handler.py             (DELETED when migration complete)
├── context_menu.py              (UPDATED — uses ToolManager for "Create Parent")
├── canvas_renderer.py           (UNCHANGED)
├── annotation_state.py          (UNCHANGED)
├── tab_annotation.py            (SLIMMED — wires ToolManager + services)
└── ... (other files unchanged)
```

**Line count check:**
- Largest new file: UndoManager at ~220 lines
- All files well under the 800-line checkpoint
- Total lines redistributed: ~2,354 → ~2,540 (slight growth from interface overhead)

---

## Part 6: Implementation Sequence

Each step is independently testable. The system works after every step. No big-bang cutover.

### Step 1: Scaffold — BaseTool + ToolManager + SelectionState

**What:** Create `tools/base.py`, `tools/tool_manager.py`, `services/selection_state.py`
with the interfaces defined but NO methods migrated yet.

**Test:** Instantiate ToolManager, verify it routes a synthetic click event to the default
tool. Unit test only — no GUI changes yet.

**Why first:** Everything else depends on these interfaces. Nailing them before moving code
prevents rework.

### Step 2: Extract NavigationMixin → navigation.py

**What:** Move the 6 pan/zoom/scroll methods out of EventHandlerMixin into their own file.
Update `tab_annotation.py` to import from the new location.

**Test:** Open the app, verify scroll/pan/zoom still works. Quick smoke test.

**Why second:** Smallest, most independent extraction. Zero risk, builds confidence.

### Step 3: Extract HoverTooltip → hover_tooltip.py

**What:** Move `_on_canvas_motion`, `_show_hover_tooltip`, `_hide_hover_tooltip` into a
standalone class. Wire its `on_motion` into the canvas binding.

**Test:** Open app, hover over boxes, verify tooltips appear/disappear.

**Why third:** Also very independent. No shared state with other interactions.

### Step 4: Extract UndoManager → services/undo_manager.py

**What:** Move `_push_undo`, `_undo`, `_redo` and the two stacks into an UndoManager class.
Give it a reference to the canvas_boxes list and the drawing methods it needs.

**Test:** Make a correction, undo it, redo it. Verify all 6 action types (relabel, reshape,
delete, dismiss, accept, merge) undo/redo correctly.

**Why fourth:** UndoManager is used by almost every tool and action. Extracting it early
means every subsequent extraction can use it cleanly.

### Step 5: Extract Clipboard → services/clipboard.py

**What:** Move `_copy_box`, `_paste_box`, `_copied_box_template` into Clipboard class.

**Test:** Copy a box (Ctrl+C), paste it (Ctrl+V). Verify type and dimensions match.

### Step 6: Extract BoxOperations → services/box_operations.py

**What:** Move `_on_accept`, `_on_relabel`, `_on_delete`, `_on_dismiss`,
`_auto_refresh_text`, `_on_rescan_text` into BoxOperations. These are the inspector
button actions.

**Test:** Select a box, click Accept/Relabel/Delete/Dismiss. Verify each persists correctly.

### Step 7: Extract SelectTool → tools/select_tool.py

**What:** This is the big one. Extract the box-selection logic from `_on_canvas_click` into
SelectTool. Wire the ToolManager into `tab_annotation.py` so canvas events route through it.

**The critical change:** `tab_annotation.py` canvas bindings now point to the ToolManager
instead of directly to EventHandlerMixin methods:

```
BEFORE: self._canvas.bind("<Button-1>", self._on_canvas_click)
AFTER:  self._canvas.bind("<Button-1>", self._tool_manager.on_click)
```

**Test:** Click boxes to select them. Shift-click for multi-select. Click empty space to
deselect. Arrow keys to navigate. Ctrl+A to select all.

**Why this step is the watershed:** After this, the ToolManager is live. Every subsequent
tool extraction is the same pattern: create the class, register it, move methods.

### Step 8: Extract MoveTool → tools/move_tool.py

**What:** Move `_do_move_drag`, `_finalize_move`, and the drag throttle into MoveTool.
SelectTool's `on_click` detects a box hit and calls `tool_manager.switch_to('move')`.

**Test:** Click and drag a box. Verify it moves smoothly (throttled). Release and verify
the correction is saved. Verify it auto-reverts to SelectTool.

### Step 9: Extract ResizeTool → tools/resize_tool.py

**What:** Move `_do_handle_drag`, `_finalize_reshape` into ResizeTool. SelectTool's
`on_click` detects a handle hit and calls `tool_manager.switch_to('resize')`.

**Test:** Grab a handle and drag. Verify reshape works. Release and verify correction saved.

### Step 10: Extract LassoTool → tools/lasso_tool.py

**What:** Move `_finalize_lasso` and the lasso drawing logic into LassoTool. SelectTool's
`on_click` detects empty space and calls `tool_manager.switch_to('lasso')`.

**Test:** Click empty space and drag a rectangle. Verify boxes inside get multi-selected.
Test word-lasso mode (Ctrl+drag with word overlay on).

### Step 11: Extract DrawTool → tools/draw_tool.py

**What:** Move `_on_word_click`, `_finalize_add`, and the draw rectangle logic into DrawTool.
Ctrl+Click activates it.

**Test:** Ctrl+drag to draw a new box. Verify the type dialog appears. Verify the detection
is saved. Test Ctrl+click with word overlay on (word selection mode).

### Step 12: Extract Actions → actions/

**What:** Move `_on_merge`, `_merge_words_into_detection`, `_on_link_column`,
`_create_group`, `_add_children_to_group`, `_remove_from_group` into their respective
action files.

**Test:** Press M to merge. Press L to link column. Press G to create group. Use inspector
buttons for group add/remove. All should work as before.

### Step 13: Delete EventHandlerMixin

**What:** At this point, EventHandlerMixin should be empty (or nearly so). Remove it from
the AnnotationTab's inheritance chain. Update imports.

**Test:** Full regression — every interaction, every shortcut, every button.

### Step 14: Build LinkTool → tools/link_tool.py (THE NEW FEATURE)

**What:** Now that the architecture is in place, building LinkTool is straightforward. It's
just another Tool subclass with its own state machine.

- Update context_menu.py: "Create Parent" → activates LinkTool via ToolManager
- Implement the PARENT_READY → LINKING → CONNECTED state machine
- Add rubber band line rendering
- Add valid-target highlighting on hover
- Add cursor management (crosshair in LINKING mode)

**Test:**
1. Right-click a box → "Create Parent" → name dialog → box highlights
2. Left-click parent → rubber band line appears
3. Move to another box → it highlights as valid target
4. Right-click child → connection made, dashed line drawn
5. Move to another box → right-click → another connection
6. Press Escape → back to SelectTool
7. Verify groups persist in database

---

## Part 7: What Is NOT Changing

This section exists because Jake correctly values knowing what stays put:

- **CanvasBox dataclass** — `annotation_state.py` unchanged
- **Canvas rendering** — `canvas_renderer.py` unchanged (draw_box, draw_group_links, etc.)
- **PDF loading** — `pdf_loader.py` unchanged
- **Annotation store** — `annotation_store.py` unchanged
- **CorrectionStore** — the database layer is completely untouched
- **Model training mixin** — `model_training.py` unchanged
- **Filter controls mixin** — `filter_controls.py` unchanged
- **Label registry mixin** — `label_registry.py` unchanged
- **Overlay renderer** — `overlay_renderer.py` unchanged
- **All other tabs** — tab_pipeline, tab_database, tab_query, etc. untouched

The Tool pattern only reorganizes HOW user events reach the existing data operations.
Same pipes, different valve handles.

---

## Part 8: Traps and Edge Cases

### 1. The "Active Tab" Guard

Currently, every keyboard handler checks `_is_active_tab()` to avoid firing when another
tab is focused. The ToolManager should do this check ONCE in its event routing, not in
every tool. Centralize it.

### 2. The "Text Widget" Guard

Many key handlers check `isinstance(event.widget, (tk.Entry, tk.Text, ...))` to avoid
stealing keypresses from text fields. Same solution: ToolManager checks this once.

### 3. Drag Throttle Sharing

MoveTool and ResizeTool both need the `_schedule_throttled_drag` / `_flush_throttled_drag`
pattern. Two options:
- **Option A:** Put it in BaseTool as a helper method (shared inheritance)
- **Option B:** Make it a tiny utility function both import

Option B is cleaner — it's a utility, not a tool behavior.

### 4. The Canvas Release "Router"

`_on_canvas_release` currently checks multiple flags to figure out what's being released
(move, resize, draw, lasso). With the Tool pattern, only the active tool receives the
release event, so this router disappears entirely. Each tool's `on_release` knows exactly
what it's finalizing.

### 5. SelectTool's Dual Role

SelectTool both handles selection AND detects when to switch tools. This is intentional —
it's the "default" tool that routes to specialized tools. But it means SelectTool needs
access to the ToolManager to call `switch_to()`. Pass the ToolManager reference in the
constructor.

### 6. Context Menu Integration

The right-click context menu (`context_menu.py`) needs to know about LinkTool to offer
"Create Parent." The cleanest approach: the context menu calls
`tool_manager.switch_to('link')` and passes the clicked box as context. LinkTool's
`enter()` method receives it and enters PARENT_READY state.

### 7. Escape Key Behavior

Escape currently deselects. With LinkTool active, Escape should exit LinkTool first,
THEN deselect on a second press. The ToolManager handles this: if the active tool is
not SelectTool, Escape tells the active tool to `exit()` and reverts to SelectTool.
If already in SelectTool, Escape deselects as normal.

---

## Summary

| Component | Lines | Status |
|-----------|-------|--------|
| BaseTool + enums | ~60 | New |
| ToolManager | ~100 | New |
| SelectTool | ~120 | Extracted from event_handler |
| MoveTool | ~100 | Extracted from event_handler |
| ResizeTool | ~100 | Extracted from event_handler |
| LassoTool | ~80 | Extracted from event_handler |
| DrawTool | ~180 | Extracted from event_handler |
| **LinkTool** | **~200** | **NEW FEATURE** |
| SelectionState | ~60 | Extracted from event_handler |
| UndoManager | ~220 | Extracted from event_handler |
| Clipboard | ~60 | Extracted from event_handler |
| BoxOperations | ~200 | Extracted from event_handler |
| MLPredictor | ~100 | Extracted from event_handler |
| MergeAction | ~180 | Extracted from event_handler |
| GroupActions | ~180 | Extracted from event_handler |
| LinkColumnAction | ~130 | Extracted from event_handler |
| HoverTooltip | ~90 | Extracted from event_handler |
| Navigation | ~50 | Extracted from event_handler |
| **Total** | **~2,410** | All files under 220 lines |

The 2,354-line monolith becomes 18 focused files, none over 220 lines, each independently
testable, and the new LinkTool feature drops in as step 14 with no architectural negotiation.
