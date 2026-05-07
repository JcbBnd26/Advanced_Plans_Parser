"""Grouper event handler mixin.

Provides all mouse event handlers for the Grouper tab:

Learn Session mode:
- Shift+click to accumulate GlyphBox selection
- Click to confirm group or inspect existing group
- Right-click to clear selection
- Group popup (single "Group ✓" button)

Edit mode:
- Multi-step gesture state machine for add/remove/split/merge

Implemented in Write 3 (Phase F + G).
"""

from __future__ import annotations


class GrouperEventsMixin:
    """Mixin providing all Grouper mouse-event handlers.

    Requires the host class to expose:
    - ``self._canvas`` — tkinter Canvas widget
    - ``self._gsession`` — ``GrouperSessionState`` instance
    - ``self._store`` — ``CorrectionStore`` instance
    - ``self._render_canvas()`` — full canvas redraw callable
    """
