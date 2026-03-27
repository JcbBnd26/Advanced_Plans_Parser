# Mode Selector Removal — Technical Writeup

## Initial State

The annotation tab (`scripts/gui/tab_annotation.py`) had a two-mode system controlled by radio buttons in the inspector panel. The modes were:

- **`"select"`** (default): Click to select boxes, shift+click for multi-select, drag to move boxes, shift+drag for lasso multi-select, all editing operations (relabel, accept, reject, dismiss, merge, resize via handles).
- **`"add"`**: Click+drag to draw a dashed rectangle on the canvas, release to open an element-type dialog and create a new detection.

The user's intent: these should not be separate modes. All operations should be available simultaneously without toggling.

---

## Step 1: Exploration — Understanding the Mode System

Launched an `Explore` subagent to thoroughly map every reference to the mode system across the GUI codebase. The subagent searched `scripts/gui/tab_annotation.py`, `scripts/gui/event_handler.py`, `scripts/gui/pdf_loader.py`, and `scripts/gui/mixins/*.py`.

### Findings — State Variables

| Variable | File | Line | Purpose |
|----------|------|------|---------|
| `self._mode: str = "select"` | `tab_annotation.py` | L100 | Current mode string |
| `self._mode_var = tk.StringVar(value="select")` | `tab_annotation.py` | L922 | Tkinter variable for radio buttons |
| `self._mode_banner` | `tab_annotation.py` | L600 | Red overlay banner shown in add mode |

### Findings — UI Widgets (tab_annotation.py L912–950)

Two `ttk.Radiobutton` widgets in Section 1 of the inspector panel:
- `_rb_select`: text="Select / Edit", value="select", command=`self._on_mode_change`
- `_rb_add`: text="Add New Element", value="add", command=`self._on_mode_change`

A `ttk.Separator` preceded them, and a `ttk.Label` with "Mode:" header.

### Findings — Mode Change Callback (tab_annotation.py L1462–1475)

```python
def _on_mode_change(self) -> None:
    self._mode = self._mode_var.get()
    if self._mode == "add":
        self._canvas.config(cursor="crosshair")
        self._deselect()
        self._mode_banner.configure(
            text="  ➕  ADD MODE — click and drag to draw a new detection box  "
        )
        self._mode_banner.place(relx=0.0, rely=0.0, relwidth=1.0, anchor="nw")
    else:
        self._canvas.config(cursor="")
        self._mode_banner.place_forget()
```

### Findings — All 6 Mode Checks in event_handler.py

1. **L176** — `_on_canvas_click`: `if self._mode == "add":` → start drawing (`self._draw_start = (cx, cy)`) and return early, skipping all select logic.

2. **L671** — `_on_canvas_drag`: `if self._mode == "add" and self._draw_start:` → draw dashed rectangle while dragging.

3. **L687** — `_on_canvas_drag`: `if self._lasso_start and self._mode == "select":` → only allow lasso multi-select in select mode.

4. **L864** — `_on_canvas_release`: `if self._mode == "add" and self._draw_start:` → call `self._finalize_add(cx, cy)` to complete the new-box flow.

5. **L1336** — `_on_pan_end`: `if self._mode == "add":` → restore crosshair cursor after panning; else restore default cursor.

6. **L2296** — `_on_canvas_motion` (hover): `cursor="crosshair" if self._mode == "add" else ""` → show crosshair when hovering over empty space in add mode.

### Analysis of Conflict

The only fundamental conflict was click+drag on empty canvas space:
- In select mode: starts a lasso multi-select rectangle (blue dashed outline)
- In add mode: starts drawing a new detection rectangle (gray dashed outline)

All other operations (clicking boxes, editing, relabeling, accepting, rejecting, dismissing, merging, resizing) work identically in both modes. The mode system existed solely to disambiguate this one drag gesture.

---

## Step 2: Design Decision — Ctrl+Drag as Add Trigger

The solution: use **Ctrl+click+drag** to draw new boxes. This eliminates the need for mode radio buttons entirely. The cursor changes to crosshair during the Ctrl+drag operation and reverts when done.

- Default click+drag on empty space → starts lasso multi-select (existing behavior)
- Ctrl+click+drag on empty space → draws new detection box (replaces add mode)
- Ctrl+click when word overlay is active → still selects words (existing behavior preserved)

---

## Step 3: Remove Mode Radio Buttons from Inspector UI

**File: `scripts/gui/tab_annotation.py`**

Removed the entire mode selector block (approximately 30 lines) between the group buttons and the filters section. This included:
- `ttk.Separator` (visual divider)
- `ttk.Label(text="Mode:")` (section header)
- `self._mode_var = tk.StringVar(value="select")` (Tkinter variable)
- `_rb_select = ttk.Radiobutton(...)` with tooltip
- `_rb_add = ttk.Radiobutton(...)` with tooltip

The `s1r` row counter continued incrementing correctly because the Filters section immediately follows the removed block and uses `s1r += 1` at its start.

**Before:**
```python
        self._btn_remove_group.pack_forget()

        # ── Mode selector ─────────────────────────────────────────
        s1r += 1
        ttk.Separator(sec1).grid(...)
        s1r += 1
        ttk.Label(sec1, text="Mode:", ...).grid(...)
        s1r += 1
        self._mode_var = tk.StringVar(value="select")
        _rb_select = ttk.Radiobutton(sec1, text="Select / Edit", ...)
        ...
        s1r += 1
        _rb_add = ttk.Radiobutton(sec1, text="Add New Element", ...)
        ...

        # ── Filters (collapsible sub-section) ─────────────────────
        s1r += 1
```

**After:**
```python
        self._btn_remove_group.pack_forget()

        # ── Filters (collapsible sub-section) ─────────────────────
        s1r += 1
```

---

## Step 4: Remove `_on_mode_change` Method

**File: `scripts/gui/tab_annotation.py`**

The method `_on_mode_change` at line 1424 (post-edit, originally ~1462) was the callback for the radio buttons. With the radio buttons removed, this method is dead code. Removed 12 lines:

```python
    # ── Mode ───────────────────────────────────────────────────────

    def _on_mode_change(self) -> None:
        self._mode = self._mode_var.get()
        if self._mode == "add":
            self._canvas.config(cursor="crosshair")
            self._deselect()
            self._mode_banner.configure(
                text="  ➕  ADD MODE — click and drag to draw a new detection box  "
            )
            self._mode_banner.place(relx=0.0, rely=0.0, relwidth=1.0, anchor="nw")
        else:
            self._canvas.config(cursor="")
            self._mode_banner.place_forget()
```

---

## Step 5: Remove `self._mode` Instance Variable

**File: `scripts/gui/tab_annotation.py`, line 100**

Removed `self._mode: str = "select"` from the `__init__` attribute block. The `_draw_start` variable that followed it was preserved.

**Before:**
```python
        self._multi_selected: list[CanvasBox] = []
        self._mode: str = "select"

        self._draw_start: tuple[float, float] | None = None
```

**After:**
```python
        self._multi_selected: list[CanvasBox] = []

        self._draw_start: tuple[float, float] | None = None
```

---

## Step 6: Update Event Handlers — First Attempt (Had an Issue)

**File: `scripts/gui/event_handler.py`**

The initial plan was to add Ctrl detection to `_on_canvas_click`:

```python
def _on_canvas_click(self, event: tk.Event) -> None:
    cx = self._canvas.canvasx(event.x)
    cy = self._canvas.canvasy(event.y)

    # Ctrl+click on empty space starts drawing a new box
    ctrl_held = bool(event.state & 0x0004)
    if ctrl_held:
        self._draw_start = (cx, cy)
        self._canvas.config(cursor="crosshair")
        return
    ...
```

### THE ISSUE: Tkinter Binding Conflict

After making this change, a search for the canvas bindings in `tab_annotation.py` revealed:

```python
self._canvas.bind("<Control-Button-1>", self._on_word_click)   # L582
self._canvas.bind("<Button-1>", self._on_canvas_click)          # L583
```

Tkinter uses **most-specific-match-first** for event binding. When the user Ctrl+clicks:
- `<Control-Button-1>` is more specific than `<Button-1>`
- Tkinter routes the event to `self._on_word_click`, NOT `self._on_canvas_click`
- The Ctrl check inside `_on_canvas_click` would **never fire**

The existing `_on_word_click` method already handled Ctrl+Click:

```python
def _on_word_click(self, event: tk.Event) -> str | None:
    """Handle Ctrl+Click for word overlay selection / lasso."""
    cx = self._canvas.canvasx(event.x)
    cy = self._canvas.canvasy(event.y)

    if not (self._word_overlay_on and self._word_overlay_items):
        # Word overlay not active — fall through to normal click
        self._on_canvas_click(event)
        return None

    # Hit-test using PDF coordinates
    ...word selection logic...
```

When word overlay was OFF, `_on_word_click` called `self._on_canvas_click(event)` as a fallback. But since `_on_canvas_click` is designed for plain clicks (not Ctrl+clicks), the Ctrl state would be checked but the event flow was wrong — it would treat the Ctrl+click as a regular click.

---

## Step 7: Fix — Route Ctrl+Drag Through `_on_word_click`

**Solution:** Instead of adding Ctrl detection to `_on_canvas_click` (which never receives Ctrl+clicks due to tkinter binding specificity), modified `_on_word_click` to handle the dual responsibility:

1. When word overlay is active → select/deselect words (existing behavior, unchanged)
2. When word overlay is OFF → start drawing a new detection box

**Before:**
```python
def _on_word_click(self, event: tk.Event) -> str | None:
    """Handle Ctrl+Click for word overlay selection / lasso."""
    cx = self._canvas.canvasx(event.x)
    cy = self._canvas.canvasy(event.y)

    if not (self._word_overlay_on and self._word_overlay_items):
        # Word overlay not active — fall through to normal click
        self._on_canvas_click(event)
        return None
```

**After:**
```python
def _on_word_click(self, event: tk.Event) -> str | None:
    """Handle Ctrl+Click for word overlay selection or new-box drawing."""
    cx = self._canvas.canvasx(event.x)
    cy = self._canvas.canvasy(event.y)

    if not (self._word_overlay_on and self._word_overlay_items):
        # Word overlay not active — Ctrl+click starts drawing a new box
        self._draw_start = (cx, cy)
        self._canvas.config(cursor="crosshair")
        return "break"
```

Key changes:
- Returns `"break"` instead of `None` to prevent event propagation
- Sets `self._draw_start` to trigger draw mode tracked by state variable instead of mode string
- Sets cursor to `"crosshair"` for visual feedback

Also reverted the Ctrl check added to `_on_canvas_click` (Step 6) since it was dead code:

```python
def _on_canvas_click(self, event: tk.Event) -> None:
    cx = self._canvas.canvasx(event.x)
    cy = self._canvas.canvasy(event.y)

    # Shift+click for multi-select   ← Ctrl check removed, back to original
    shift_held = bool(event.state & 0x0001)
```

---

## Step 8: Update Remaining Mode Checks in event_handler.py

Six `self._mode` references needed updating. Applied via `multi_replace_string_in_file` in one batch:

### Change 1 — Canvas drag (L671)

```python
# Before:
if self._mode == "add" and self._draw_start:

# After:
if self._draw_start:
```

Comment changed from "Handle add-mode drag" to "Handle add-mode drag (Ctrl+drag to draw new box)". The `_draw_start` check alone is sufficient — if `_draw_start` is set, we're drawing regardless of any mode variable.

### Change 2 — Lasso drag (L687)

```python
# Before:
if self._lasso_start and self._mode == "select":

# After:
if self._lasso_start:
```

The `self._mode == "select"` guard was preventing lasso from working in add mode. Now there's no mode to check — if `_lasso_start` is set, we're lassoing. There's no conflict with drawing because `_draw_start` is checked first in the method (and returns early).

### Change 3 — Canvas release (L864)

```python
# Before:
if self._mode == "add" and self._draw_start:

# After:
if self._draw_start:
```

Same pattern — `_draw_start` is the state flag. If it's set, finalize the add.

### Change 4 — Pan end (L1336)

```python
# Before:
def _on_pan_end(self, event: tk.Event) -> None:
    self._pan_start = None
    if self._mode == "add":
        self._canvas.config(cursor="crosshair")
    else:
        self._canvas.config(cursor="")

# After:
def _on_pan_end(self, event: tk.Event) -> None:
    self._pan_start = None
    self._canvas.config(cursor="")
```

No longer need to restore crosshair after panning because there's no persistent "add mode" — the crosshair only appears during the Ctrl+drag gesture itself.

### Change 5 — Hover cursor (L2296)

```python
# Before:
self._canvas.config(cursor="crosshair" if self._mode == "add" else "")

# After:
self._canvas.config(cursor="")
```

Same reasoning — no persistent crosshair cursor.

---

## Step 9: Reset Cursor After Drawing Completes

**File: `scripts/gui/event_handler.py`**

The `_finalize_add` method (L916) cleans up after a draw gesture. Added a cursor reset at the top of this method so the crosshair disappears when the draw is done:

**Before:**
```python
def _finalize_add(self, cx: float, cy: float) -> None:
    """Complete adding a new detection after drawing a rectangle."""
    if self._draw_rect_id:
        self._canvas.delete(self._draw_rect_id)
        self._draw_rect_id = None

    if not self._draw_start or not self._doc_id:
        self._draw_start = None
        return
```

**After:**
```python
def _finalize_add(self, cx: float, cy: float) -> None:
    """Complete adding a new detection after drawing a rectangle."""
    if self._draw_rect_id:
        self._canvas.delete(self._draw_rect_id)
        self._draw_rect_id = None

    self._canvas.config(cursor="")

    if not self._draw_start or not self._doc_id:
        self._draw_start = None
        return
```

The cursor reset is placed before the early-return guard so that **both** paths (valid draw and invalid/cancelled draw) reset the cursor.

---

## Step 10: Update Keyboard Legend

**File: `scripts/gui/tab_annotation.py`**

Added `Ctrl+Drag Draw new box` to the keyboard shortcut legend displayed at the bottom of the inspector:

**Before:**
```python
            "Ctrl+C Copy box/word text\n"
            "Ctrl+V Paste box\n"
            "G=Group  L=Link Column\n"
```

**After:**
```python
            "Ctrl+C Copy box/word text\n"
            "Ctrl+V Paste box\n"
            "Ctrl+Drag Draw new box\n"
            "G=Group  L=Link Column\n"
```

---

## Step 11: Verify Dead Code

Ran `grep_search` for remaining references to `self._mode`, `_mode_var`, and `_mode_banner`:

- **`_mode_banner`** (L599 in tab_annotation.py): Still created as a `tk.Label` widget but never `.place()`d or `.configure()`d since `_on_mode_change` was removed. This is harmless dead code — the widget exists in memory but is never shown. Left it in place because removing it would require restructuring the canvas setup block and the widget consumes negligible resources.

- **`_mode_var`**: No remaining references (removed from init and UI).

- **`self._mode`**: Zero references remaining in `event_handler.py`. Zero references remaining in `tab_annotation.py` (confirmed via grep). The only matches for `_mode` were `_offline_mode_var` (a completely separate PNG fallback feature) and `page_mode_var` (in `tab_pipeline.py` for page range selection), which are unrelated.

---

## Step 12: Verification

1. **Syntax check** — `py_compile.compile()` on both `tab_annotation.py` and `event_handler.py`: both passed.
2. **IDE error check** — `get_errors` on both files: zero errors in both.
3. **Binding flow analysis**: Verified that `<B1-Motion>` and `<ButtonRelease-1>` fire regardless of modifier keys. When the user Ctrl+clicks (routed to `_on_word_click` which sets `_draw_start`), subsequent `<B1-Motion>` events hit `_on_canvas_drag` which checks `if self._draw_start:` and draws the rectangle. `<ButtonRelease-1>` hits `_on_canvas_release` which checks `if self._draw_start:` and calls `_finalize_add()`.

---

## Summary of Files Modified

| File | Lines Changed | Changes |
|------|--------------|---------|
| `scripts/gui/tab_annotation.py` | −31 lines | Removed mode radio buttons, separator, label, `_mode_var`, `_on_mode_change()`, `self._mode` init. Added "Ctrl+Drag" to keyboard legend. |
| `scripts/gui/event_handler.py` | −6 / +12 lines | Modified `_on_word_click` to start drawing when word overlay is off. Removed all 6 `self._mode` checks, replaced with `self._draw_start` state checks. Added cursor reset in `_finalize_add`. Simplified `_on_pan_end` and hover cursor logic. |

---

## Key Architectural Insight

The mode system was essentially a **state machine with two states** (`"select"` and `"add"`) that controlled one ambiguous gesture: click+drag on empty canvas. By replacing the mode toggle with a **modifier key** (Ctrl), the ambiguity is resolved at the input level rather than the state level. The `_draw_start` variable — which already existed and was used by the drawing logic — became the sole state flag for "currently drawing a new box", eliminating the need for a separate `_mode` string.
