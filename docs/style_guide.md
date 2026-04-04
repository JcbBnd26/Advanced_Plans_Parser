# PlanCheck Visual Style Guide

This document defines the visual language for the PlanCheck GUI.
All widgets, dialogs, and themed elements should reference these standards.

---

## Color Palette

| Role           | Hex       | Usage                                   |
|----------------|-----------|-----------------------------------------|
| Background     | `#1e1e1e` | Main window, canvas backgrounds         |
| Surface        | `#252526` | Frames, panels, card backgrounds        |
| Surface Alt    | `#2d2d30` | Alternate rows, hover states            |
| Border         | `#3c3c3c` | Frame borders, separators               |
| Text Primary   | `#d4d4d4` | Body text, labels                       |
| Text Secondary | `#808080` | Hints, disabled text, placeholders      |
| Accent         | `#0078d4` | Active tab indicator, focused controls  |
| Success        | `#2d5a2d` | Completed stages, pass indicators       |
| Warning        | `#8b7300` | Warning badges, caution states          |
| Error          | `#5a2d2d` | Error panels, failure indicators        |
| Running        | `#264f78` | In-progress stages, active processing   |

## Font Choices

| Context     | Font              | Size  | Weight |
|-------------|-------------------|-------|--------|
| Tab headers | Segoe UI          | 10    | Normal |
| Body labels | TkDefaultFont     | 9     | Normal |
| Headings    | Segoe UI          | 11    | Bold   |
| Monospace   | Consolas          | 9     | Normal |
| Buttons     | TkDefaultFont     | 9     | Normal |
| Run button  | TkDefaultFont     | 12    | Bold   |
| Tooltips    | Consolas          | 8     | Normal |

## Spacing Standards

| Element            | Padding      |
|--------------------|-------------|
| Frame outer        | 10px         |
| Control inner      | (6, 2)       |
| Section gap        | 8px vertical |
| Button padding     | (8, 4)       |
| Dialog margins     | 12px         |

## Status Colors (Pipeline)

These match `TOCRProgressBar._STATUS_COLORS` and `StageProgressBar._COLOURS`:

| Status    | Color     |
|-----------|-----------|
| Pending   | `#3c3c3c` |
| Running   | `#264f78` |
| Success   | `#2d5a2d` |
| Warning   | `#8b7300` |
| Error     | `#5a2d2d` |

## Status Bar Layout

```
┌──────────────────────┬──────────────────────┬──────────────────────┐
│ Project / Document   │ Last action          │ ML status            │
│ (left-aligned)       │ (center)             │ (right-aligned)      │
└──────────────────────┴──────────────────────┴──────────────────────┘
```

## Keyboard Shortcuts

| Shortcut    | Action              |
|-------------|---------------------|
| Ctrl+Z      | Undo                |
| Ctrl+Y      | Redo                |
| Ctrl+N      | New Project         |
| Ctrl+O      | Open Project        |
| Ctrl+S      | Context-dependent save |
| F5          | Run Pipeline        |
| Ctrl+R      | Run Pipeline        |
| Ctrl+,      | Open Settings       |
| Escape      | Close dialog        |
