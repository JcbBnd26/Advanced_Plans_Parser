"""Tool pattern for annotation tab event handling.

Each tool encapsulates a distinct user interaction mode (select, move,
resize, lasso, draw, link).  The :class:`ToolManager` routes canvas
events to whichever tool is currently active.
"""

from __future__ import annotations

from .base import BaseTool, ToolContext
from .draw_tool import DrawTool
from .lasso_tool import LassoTool
from .link_tool import LinkTool
from .move_tool import MoveTool
from .resize_tool import ResizeTool
from .select_tool import SelectTool
from .tool_manager import ToolManager

__all__ = [
    "BaseTool",
    "DrawTool",
    "LassoTool",
    "LinkTool",
    "MoveTool",
    "ResizeTool",
    "SelectTool",
    "ToolContext",
    "ToolManager",
]
