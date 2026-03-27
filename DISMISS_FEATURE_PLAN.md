# Dismiss / Skip Feature — Agent Implementation Plan

## Problem

The annotation tab currently has two actions for unwanted detections: **Accept** (correct as-is) and **Reject** (mark as `__negative__` false positive for training). There is no way to say "I don't know what this is" or "skip for now" without poisoning the training data. Marking an ambiguous element as `__negative__` teaches the model to suppress things it should actually be detecting.

## Goal

Add a **Dismiss** action that removes a detection from the current review session without writing any training signal. Dismissed detections are tracked in a lightweight DB table so the user can revisit them later when they have a label for them.

---

## Architecture Overview

```
User clicks Dismiss (or presses X)
  → Box removed from canvas (visual only)
  → Row inserted into `dismissed_detections` table (tracking only)
  → NO row in `corrections` table
  → NO row in `training_examples` table
  → Undo supported (re-adds box to canvas, deletes dismissed row)
```

The key invariant: **`build_training_set()` never sees dismissed detections.** They are invisible to the ML pipeline.

---

## Phase 1 — Database Schema

### File: `src/plancheck/corrections/store.py`

Add the `dismissed_detections` table to the schema initialization. Find the `_ensure_tables()` or equivalent method that runs `CREATE TABLE IF NOT EXISTS` statements and add:

```sql
CREATE TABLE IF NOT EXISTS dismissed_detections (
    dismiss_id    TEXT PRIMARY KEY,
    detection_id  TEXT NOT NULL,
    doc_id        TEXT NOT NULL,
    page          INTEGER NOT NULL,
    dismissed_at  TEXT NOT NULL,
    session_id    TEXT DEFAULT '',
    reason        TEXT DEFAULT '',
    FOREIGN KEY (detection_id) REFERENCES detections(detection_id)
);
CREATE INDEX IF NOT EXISTS idx_dismissed_doc_page
    ON dismissed_detections(doc_id, page);
CREATE INDEX IF NOT EXISTS idx_dismissed_detection
    ON dismissed_detections(detection_id);
```

**Fields explained:**
- `dismiss_id` — Generated ID like `dis_<8hex>`, same pattern as `cor_` IDs.
- `detection_id` — Links back to the detection being skipped.
- `doc_id`, `page` — For filtering dismissed boxes when loading a page.
- `dismissed_at` — UTC ISO timestamp.
- `session_id` — Ties to the annotation session for audit trail.
- `reason` — Optional freetext. Not used in v1 but available for future "I don't know" vs "come back later" distinction.

### Store Methods to Add

Add these three methods to the `CorrectionStore` class (or whatever the store class is named):

```python
def dismiss_detection(
    self,
    detection_id: str,
    doc_id: str,
    page: int,
    session_id: str = "",
    reason: str = "",
) -> str:
    """Dismiss a detection without creating a training signal.

    Returns the dismiss_id.
    """
    dismiss_id = _gen_id("dis_")
    with self._write_lock():
        self._conn.execute(
            "INSERT OR IGNORE INTO dismissed_detections "
            "(dismiss_id, detection_id, doc_id, page, "
            " dismissed_at, session_id, reason) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (dismiss_id, detection_id, doc_id, page,
             _utcnow_iso(), session_id, reason),
        )
        self._conn.commit()
    return dismiss_id


def undismiss_detection(self, detection_id: str) -> None:
    """Remove a detection from the dismissed table (undo support)."""
    with self._write_lock():
        self._conn.execute(
            "DELETE FROM dismissed_detections WHERE detection_id = ?",
            (detection_id,),
        )
        self._conn.commit()


def get_dismissed_ids_for_page(
    self, doc_id: str, page: int
) -> set[str]:
    """Return detection_ids that were dismissed on this page."""
    rows = self._conn.execute(
        "SELECT detection_id FROM dismissed_detections "
        "WHERE doc_id = ? AND page = ?",
        (doc_id, page),
    ).fetchall()
    return {r["detection_id"] for r in rows}
```

---

## Phase 2 — GUI: Add the Dismiss Button

### File: `scripts/gui/tab_annotation.py`

Locate the button frame (around line 687–707). Add the Dismiss button after the Reject button:

```python
# ── After the existing Reject button (around line 707) ──

_btn_dismiss = ttk.Button(
    btn_frame, text="Skip ⊘", command=self._on_dismiss
)
_btn_dismiss.pack(side="left", padx=3)
self._tooltip(
    _btn_dismiss,
    "Dismiss this detection without affecting training data. "
    "Use when you're unsure of the label or want to skip for now. "
    "Shortcut: X.",
)
```

### Keybinding

In the keybinding block (around line 1190–1210), add:

```python
self.root.bind("<Key-x>", self._key_dismiss)
```

---

## Phase 3 — GUI: Dismiss Event Handler

### File: `scripts/gui/event_handler.py`

Add the `_on_dismiss` method. Model it after `_on_delete` but **do NOT call `save_correction`**. Call `dismiss_detection` instead.

```python
def _on_dismiss(self) -> None:
    """Remove detection(s) from the canvas without creating training data."""
    # Batch-aware: apply to all multi-selected + selected box
    targets = list(self._multi_selected)
    if self._selected_box and self._selected_box not in targets:
        targets.append(self._selected_box)
    if not targets or not self._doc_id:
        self._status.configure(text="No box selected")
        return

    # No confirmation dialog — dismiss is lightweight and undoable.
    # (Unlike Reject, there is no training consequence.)

    for cbox in targets:
        self._push_undo("dismiss", cbox)
        self._store.dismiss_detection(
            detection_id=cbox.detection_id,
            doc_id=self._doc_id,
            page=self._page,
            session_id=self._session_id,
        )

        # Remove from canvas (same cleanup as _on_delete)
        if cbox.rect_id:
            self._canvas.delete(cbox.rect_id)
        if cbox.label_id:
            self._canvas.delete(cbox.label_id)
        if cbox.conf_dot_id:
            self._canvas.delete(cbox.conf_dot_id)
        for hid in cbox.handle_ids:
            self._canvas.delete(hid)
        if cbox in self._canvas_boxes:
            self._canvas_boxes.remove(cbox)
        # NOTE: Do NOT increment _session_count.
        # Dismiss is not a correction — it shouldn't count toward
        # the "corrections this session" metric.

    self._selected_box = None
    self._multi_selected.clear()
    self._deselect()
    self._update_multi_label()
    self._update_page_summary()
    self._status.configure(text=f"Dismissed {len(targets)} detection(s)")
```

### Key handler wrapper

```python
def _key_dismiss(self, event: tk.Event) -> None:
    if self._is_active_tab() and not isinstance(
        event.widget, (tk.Entry, tk.Text, ttk.Entry, ttk.Spinbox)
    ):
        self._on_dismiss()
```

---

## Phase 4 — Undo Support for Dismiss

### File: `scripts/gui/event_handler.py`

In the `_undo` method, find the `elif action == "delete":` block (around line 1433). Add a parallel block for dismiss directly after it:

```python
elif action == "dismiss":
    # Undo dismiss: re-add box visually and remove from dismissed table
    self._store.undismiss_detection(rec["detection_id"])
    already_exists = any(
        cb.detection_id == rec["detection_id"]
        for cb in self._canvas_boxes
    )
    if already_exists:
        self._status.configure(
            text="Undo dismiss (box already restored)"
        )
    else:
        cbox = CanvasBox(
            detection_id=rec["detection_id"],
            element_type=rec["element_type"],
            confidence=rec.get("confidence"),
            text_content="",
            features={},
            pdf_bbox=rec["pdf_bbox"],
            corrected=rec.get("corrected", False),
        )
        self._canvas_boxes.append(cbox)
        self._draw_box(cbox)
        self._status.configure(text="Undo dismiss")
```

---

## Phase 5 — Filter Dismissed Detections on Page Load

When the annotation tab loads detections for a page, dismissed detections should be excluded so they don't reappear every time you navigate away and back.

### File: `scripts/gui/event_handler.py` (or wherever page loading happens)

Find the method that populates `self._canvas_boxes` from detections for the current page. After fetching the detection rows, add a filter:

```python
# After fetching detections for this page, filter out dismissed ones
dismissed_ids = self._store.get_dismissed_ids_for_page(
    self._doc_id, self._page
)
# Filter before creating CanvasBox objects
rows = [r for r in rows if r["detection_id"] not in dismissed_ids]
```

**Important:** Search for where `_canvas_boxes` is populated from DB queries. This is likely in a `_load_page` or `_render_detections` type method. The filter goes right after the SQL fetch and before the CanvasBox construction loop.

---

## Phase 6 — Diagnostics Visibility (Optional but Recommended)

### File: `scripts/gui/tab_database.py` (or tab_diagnostics.py)

Add a simple count of dismissed detections to whatever summary stats are shown. This lets the user see how many detections are in limbo:

```python
dismissed_count = self._store._conn.execute(
    "SELECT COUNT(*) FROM dismissed_detections"
).fetchone()[0]
```

Display it alongside existing stats like total corrections, total detections, etc.

---

## Behavioral Summary

| Action | Button | Key | Writes to `corrections`? | Writes to `training_examples`? | Writes to `dismissed_detections`? | Counts as session correction? |
|--------|--------|-----|--------------------------|-------------------------------|----------------------------------|------------------------------|
| Accept | Accept ✓ | A | Yes (`accept`) | Yes (positive) | No | Yes |
| Relabel | Relabel | R | Yes (`relabel`) | Yes (positive) | No | Yes |
| Reject | Reject ✗ | D | Yes (`delete`) | Yes (`__negative__`) | No | Yes |
| **Dismiss** | **Skip ⊘** | **X** | **No** | **No** | **Yes** | **No** |

---

## Testing Checklist

1. **Dismiss single box** — click box, press X. Box disappears. Status bar says "Dismissed 1 detection(s)". Session count does NOT increment.
2. **Dismiss multi-select** — select 3 boxes, press X. All 3 disappear.
3. **Undo dismiss** — Ctrl+Z after dismiss. Box reappears on canvas. Row removed from `dismissed_detections`.
4. **Persistence across page navigation** — dismiss a box, navigate to another page, come back. Dismissed box should NOT reappear.
5. **No training contamination** — dismiss 5 boxes, run `build_training_set()`. Verify zero of those detection_ids appear in `training_examples`.
6. **DB integrity** — check that `dismissed_detections` has the expected rows with correct `doc_id`, `page`, `detection_id`, and `session_id`.
7. **Coexistence with Reject** — reject a box (→ `__negative__` training example). Then on a different box, dismiss it. Verify the rejected one IS in training data and the dismissed one is NOT.

---

## Files Modified (Summary)

| File | Change |
|------|--------|
| `src/plancheck/corrections/store.py` | Add `dismissed_detections` table, 3 new methods |
| `scripts/gui/tab_annotation.py` | Add Skip button, add `<Key-x>` binding |
| `scripts/gui/event_handler.py` | Add `_on_dismiss()`, `_key_dismiss()`, undo handler for "dismiss", filter dismissed on page load |
| `scripts/gui/tab_database.py` (optional) | Show dismissed count in summary stats |
