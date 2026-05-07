"""Grouper canvas rendering mixin.

Provides all drawing methods for the Grouper tab's independent
tkinter Canvas.  Renders:

- PDF page background image
- Grey GlyphBox outlines (all tokens)
- Yellow Shift+click selection highlight
- Green bounding boxes for confirmed groups
- Dashed blue machine-group overlays (when Show Machine Groups is on)
- Cyan single-click inspect highlight

Implemented in Write 3 (Phase E).
"""

from __future__ import annotations


class GrouperCanvasMixin:
    """Mixin providing all Grouper canvas drawing methods.

    Requires the host class to expose:
    - ``self._canvas`` — tkinter Canvas widget
    - ``self._gsession`` — ``GrouperSessionState`` instance
    - ``self._scale`` — float: PDF-to-canvas pixel scale factor
    - ``self._photo`` — ImageTk.PhotoImage reference (kept alive)
    """
