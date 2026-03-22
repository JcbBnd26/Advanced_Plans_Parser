"""Entry point for ``python -m scripts.gui``.

Launches the GUI under ``pythonw`` with an in-process splash screen,
avoiding the detached helper window that could be left behind on Windows.
"""

from __future__ import annotations

import os
import queue
import sys
import threading
import time
import tkinter as tk
import traceback
from pathlib import Path
from tkinter import ttk

# Under pythonw on Windows, sys.stdout / sys.stderr are None.
# Any code that writes to them (print, logging StreamHandler, etc.)
# would crash with AttributeError.  Redirect to devnull.
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w", encoding="utf-8")  # noqa: SIM115
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w", encoding="utf-8")  # noqa: SIM115


def _write_bootstrap_crash_log() -> None:
    """Write a startup crash log even if gui.py was not imported yet."""
    crash_path = Path("logs") / "gui_crash.txt"
    crash_path.parent.mkdir(exist_ok=True)
    crash_path.write_text(traceback.format_exc(), encoding="utf-8")


def _build_splash(root: tk.Tk) -> tuple[tk.Toplevel, tk.Label, ttk.Progressbar]:
    """Create a lightweight in-process splash window."""
    root.withdraw()

    splash = tk.Toplevel(root)
    splash.title("")
    splash.overrideredirect(True)
    splash.attributes("-topmost", True)
    splash.configure(bg="#f3f1eb")

    width, height = 360, 120
    x = (splash.winfo_screenwidth() - width) // 2
    y = (splash.winfo_screenheight() - height) // 2
    splash.geometry(f"{width}x{height}+{x}+{y}")

    tk.Label(
        splash,
        text="Advanced Plans Parser",
        font=("Segoe UI", 13, "bold"),
        bg="#f3f1eb",
        fg="#1f1f1f",
    ).pack(pady=(18, 6))

    status_label = tk.Label(
        splash,
        text="Loading…",
        font=("Segoe UI", 10),
        bg="#f3f1eb",
        fg="#555555",
    )
    status_label.pack(pady=(0, 6))

    progress = ttk.Progressbar(
        splash,
        mode="indeterminate",
        length=180,
        orient="horizontal",
    )
    progress.pack(pady=(2, 0))
    progress.start(12)

    root.update_idletasks()
    root.update()
    return splash, status_label, progress


def _destroy_splash(
    splash: tk.Toplevel | None,
    progress: ttk.Progressbar | None = None,
) -> None:
    """Close the splash if it still exists."""
    if progress is not None:
        try:
            progress.stop()
        except Exception:  # noqa: BLE001
            pass
    if splash is None:
        return
    try:
        if splash.winfo_exists():
            splash.destroy()
    except Exception:  # noqa: BLE001
        pass


def _bootstrap_gui(result_queue: queue.Queue) -> None:
    """Import GUI modules and configure logging off the Tk event loop thread."""
    try:
        from .gui import PlanParserGUI, _setup_logging, _write_crash_log

        _setup_logging()
        result_queue.put(
            {
                "ok": True,
                "PlanParserGUI": PlanParserGUI,
                "write_crash_log": _write_crash_log,
            }
        )
    except Exception as exc:  # noqa: BLE001
        result_queue.put(
            {
                "ok": False,
                "error": exc,
                "traceback": traceback.format_exc(),
            }
        )


def _handle_bootstrap_failure(
    root: tk.Tk,
    splash: tk.Toplevel | None,
    progress: ttk.Progressbar | None,
    payload: dict,
) -> None:
    """Tear down startup UI and persist bootstrap errors."""
    _destroy_splash(splash, progress)
    try:
        root.destroy()
    except Exception:  # noqa: BLE001
        pass

    trace_text = payload.get("traceback")
    if trace_text:
        crash_path = Path("logs") / "gui_crash.txt"
        crash_path.parent.mkdir(exist_ok=True)
        crash_path.write_text(trace_text, encoding="utf-8")
    else:
        _write_bootstrap_crash_log()

    raise payload["error"]


def _run() -> None:
    root = tk.Tk()
    splash, status_label, progress = _build_splash(root)
    result_queue: queue.Queue = queue.Queue()
    write_crash_log = _write_bootstrap_crash_log
    bootstrap_thread = threading.Thread(
        target=_bootstrap_gui,
        args=(result_queue,),
        daemon=True,
    )

    try:
        status_label.configure(text="Importing interface modules…")
        bootstrap_thread.start()

        payload: dict | None = None
        while payload is None:
            root.update_idletasks()
            root.update()
            try:
                payload = result_queue.get_nowait()
            except queue.Empty:
                time.sleep(0.02)

        if not payload.get("ok"):
            _handle_bootstrap_failure(root, splash, progress, payload)

        status_label.configure(text="Building workspace…")
        root.update_idletasks()
        root.update()

        plan_parser_gui = payload["PlanParserGUI"]
        write_crash_log = payload["write_crash_log"]

        _app = plan_parser_gui(root)  # noqa: F841

        _destroy_splash(splash, progress)
        root.deiconify()
        root.lift()
        root.focus_force()
        root.mainloop()
    except Exception:
        _destroy_splash(splash, progress)
        try:
            root.destroy()
        except Exception:  # noqa: BLE001
            pass
        try:
            write_crash_log()
        except Exception:  # noqa: BLE001
            _write_bootstrap_crash_log()
        raise


_run()
