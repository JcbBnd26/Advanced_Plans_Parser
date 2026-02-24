# Annotation Tab — User Guide

## Overview

The Annotation tab lets you visually inspect PDF pages, run the detection pipeline, and correct the results. Every correction you make is saved to a local database and feeds back into model training, so the system gets smarter over time.

---

## Getting Started

1. **Open a PDF** — Click **Browse…** and select a PDF file.
2. **Run the pipeline** — Click **Run Pipeline** to detect elements on the current page, or **Run All Pages** to process every page.
3. **Review detections** — Colored boxes appear on the page. Click any box to inspect and correct it.

---

## Top Bar Controls

| Control | What it does |
|---------|-------------|
| **Browse…** | Select a PDF file to work with |
| **◀ / ▶** | Go to the previous / next page |
| **Page spinbox** | Type a page number and press **Enter** to jump directly |
| **DPI spinbox** | Set render resolution (72–600). Higher = sharper but slower |
| **− / +** | Zoom out / in |
| **Fit** | Zoom so the full page fits in the window |
| **Run Pipeline** | Detect elements on the current page |
| **Run All Pages** | Detect elements on every page of the PDF |
| **Load Detections** | Reload previously saved detections from the database |
| **Words** | Toggle light-gray boxes around every word pdfplumber found (useful for checking OCR/text coverage) |

---

## Selecting Boxes

- **Click** a box to select it. It highlights and shows resize handles.
- **Shift+Click** additional boxes to multi-select.
- **Drag on empty space** to lasso-select multiple boxes at once.
- **Ctrl+A** selects all visible boxes.
- **Escape** deselects everything.
- **← / →** cycle through boxes one at a time.

---

## Moving and Resizing

- **Move**: Click a selected box and drag it to a new position.
- **Resize**: Drag any of the 8 small handles around a selected box (corners and edge midpoints).

Both actions save automatically and can be undone with **Ctrl+Z**.

---

## The Inspector Panel (Right Side)

When a box is selected, the inspector shows:

| Field | Description |
|-------|-------------|
| **ID** | Unique detection identifier (right-click to copy) |
| **Type** | Element type dropdown — select an existing type or type a new one |
| **Conf** | Detection confidence (green ≥90%, yellow 50–90%, red <50%) |
| **Text** | Extracted text content (read-only; right-click to copy) |

### Core Actions

| Button | Shortcut | What it does |
|--------|----------|-------------|
| **Accept ✓** | `A` | Confirm the detection is correct as-is |
| **Relabel** | `R` | Change the box's type to whatever is selected in the Type dropdown |
| **Delete ✗** | `D` | Remove a false-positive detection (asks for confirmation) |

### Batch Actions (when multiple boxes are selected)

| Button | What it does |
|--------|-------------|
| **Batch Accept** | Accept all selected boxes at once |
| **Batch Relabel** | Change the type of all selected boxes to the dropdown value |

---

## Adding New Boxes

1. Switch to **Add New Element** mode (radio button below the top bar).
2. Draw a rectangle on the page by clicking and dragging. The cursor becomes a crosshair.
3. A dialog asks you to pick the element type.
4. The new box is created and text is automatically extracted from the PDF under it.

Switch back to **Select / Edit** mode when done.

---

## Text Extraction

- When you create or paste a box, text is automatically extracted from the PDF region it covers.
- **Rescan ↻** (below the text field) re-extracts text if you've moved or resized the box.
- **Polygon-aware**: When a box has a polygon boundary (from merging), text extraction filters words by whether they fall inside the polygon — not just the rectangular bounding box. The status bar shows "(polygon)" or "(rect)" to indicate which mode was used.

---

## Word Overlay

Press **W** or check the **Words** toggle in the top bar to show light-gray rectangles around every word pdfplumber identified on the page. This is useful for verifying that all text has been recognized. The overlay refreshes automatically when you zoom or change pages.

---

## Merging Boxes

To combine multiple boxes into one irregular polygon shape:

1. Select 2+ boxes (Shift+Click or lasso).
2. Click **Merge ⊞** or press **M**.
3. If the boxes have different types, you'll be asked to pick the merged type.

The result is a single detection with a polygon outline matching the union of all the original boxes. Text is re-extracted using the merged polygon shape, so only words that fall inside the polygon are included — giving precise results for irregular shapes. The polygon is persisted to the database.

---

## Grouping Boxes (WBS Hierarchy)

Groups let you organize related boxes into parent-child hierarchies:

1. **Create a group**: Select a box, click **Create Group** (or press **G**), and give the group a name.
2. **Add children**: Select the group's root box, then Shift+Click other boxes, then click **Add to Group**.
3. **Remove from group**: Select a grouped box and click **Remove**. Removing the parent deletes the entire group.

When you select a grouped box, dashed lines connect it to related boxes in the group. Groups are saved to the database and persist across sessions.

You can also use the **right-click context menu** on the canvas for all group operations.

---

## Copy and Paste Boxes

- **Ctrl+C** (or right-click → **Copy Box**) copies the selected box's size and type.
- **Ctrl+V** (or right-click → **Paste Box**) creates a new box with the same dimensions, centered at the canvas view (keyboard) or at the cursor position (right-click).

Text is automatically extracted from the PDF under the pasted box's location.

---

## Undo / Redo

- **Ctrl+Z** undoes the last action (accept, relabel, delete, reshape, merge).
- **Ctrl+Y** redoes it.
- Undo fully restores merged boxes to their original state.

---

## Filtering Detections

The filter section in the inspector lets you narrow what's shown on the canvas:

- **Label checkboxes**: Toggle visibility per element type.
- **Min confidence slider**: Hide boxes below a confidence threshold.
- **Uncorrected only**: Show only boxes that haven't been reviewed yet.

The status bar shows "Showing N/M detections" when a filter is active.

---

## Model Training

The annotation tab includes a built-in machine learning loop:

1. **Correct detections**: Accept, relabel, or delete boxes to build training data.
2. **Train Model**: Click the **Train Model** button (needs ≥5 corrections). Training runs in the background.
3. **Model suggestions**: Once trained, selecting a box shows "Model suggests: X (Y%)" if the model disagrees with the current label. Click **Apply** to accept the suggestion.
4. **Suggest Next Page**: Uses active learning to navigate you to the page where the model is least confident — annotating those pages improves the model fastest.
5. **Metrics**: View accuracy and per-class performance.

---

## Sessions, Snapshots, and Stats

| Feature | What it does |
|---------|-------------|
| **Session counter** | Tracks how many corrections you've made in this session |
| **Save Session** | Saves a summary and resets the counter |
| **Snapshot** | Creates a timestamped backup of the entire corrections database |
| **Restore…** | Lists all snapshots and lets you revert to a previous state |
| **Refresh Stats** | Shows totals: documents, detections, corrections, and training examples |

---

## Panning and Scrolling

| Input | Action |
|-------|--------|
| **Mouse wheel** | Scroll vertically |
| **Shift + Mouse wheel** | Scroll horizontally |
| **Middle-click + drag** | Pan freely in any direction |

---

## Keyboard Shortcuts — Quick Reference

| Key | Action |
|-----|--------|
| `A` | Accept selected box |
| `D` | Delete selected box |
| `R` | Relabel selected box |
| `M` | Merge multi-selected boxes |
| `G` | Create group from selected box |
| `W` | Toggle word overlay |
| `F` | Fit page to window |
| `+` / `-` | Zoom in / out |
| `← / →` | Cycle through boxes |
| `Ctrl+← / Ctrl+→` | Previous / next page |
| `Ctrl+A` | Select all boxes |
| `Ctrl+Z / Ctrl+Y` | Undo / redo |
| `Ctrl+C / Ctrl+V` | Copy / paste box |
| `Escape` | Deselect |

---

## Tips

- **Right-click** any read-only field (ID, Confidence, Text) to copy its value.
- **Tooltips** appear when you hover over any button or control for 0.5 seconds.
- Corrections carry forward across pipeline re-runs — if you relabel a box and re-run the pipeline, the correction is automatically reapplied to any detection that overlaps the same area.
- The progress bar appears during pipeline runs and disappears when complete.
- You can register custom element types by typing a new name into the Type dropdown and pressing Enter, or by clicking the **+** button.
