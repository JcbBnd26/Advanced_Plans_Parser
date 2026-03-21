"""Entry point for ``python -m scripts.gui``.

Shows a lightweight splash screen in a **separate subprocess** so the
timer keeps ticking while the main process does heavy imports and
builds the UI (both of which block the tkinter event loop).

The splash is a standalone script (``_splash.py``) launched via
subprocess so it works reliably with both ``python`` and ``pythonw``.
"""

from __future__ import annotations

import os
import sys
import time

# Under pythonw on Windows, sys.stdout / sys.stderr are None.
# Any code that writes to them (print, logging StreamHandler, etc.)
# would crash with AttributeError.  Redirect to devnull.
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w", encoding="utf-8")  # noqa: SIM115
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w", encoding="utf-8")  # noqa: SIM115

import subprocess
import tkinter as tk
from pathlib import Path

_TRACE = Path("logs/startup_trace.log")


def _t(msg: str) -> None:
    try:
        with _TRACE.open("a", encoding="utf-8") as f:
            f.write(f"[{time.perf_counter():10.2f}] {msg}\n")
    except Exception:  # noqa: BLE001
        pass


def _run() -> None:
    _TRACE.parent.mkdir(exist_ok=True)
    # Truncate old trace
    _TRACE.write_text("", encoding="utf-8")

    _t(f"step1: exe={sys.executable}")
    _t(f"step1b: argv={sys.argv}")

    # Launch the splash as a completely independent subprocess.
    splash_script = Path(__file__).with_name("_splash.py")

    _exe = Path(sys.executable)
    _pythonw = _exe.with_name(_exe.name.replace("python", "pythonw"))
    splash_exe = str(_pythonw) if _pythonw.exists() else sys.executable

    _t(f"step2: splash_exe={splash_exe}")
    splash_proc = subprocess.Popen([splash_exe, str(splash_script)])
    _t(f"step3: splash pid={splash_proc.pid}")

    _t("step4: importing gui...")
    try:
        from .gui import PlanParserGUI, _setup_logging

        _t("step5: gui imported OK")
    except Exception as exc:
        _t(f"step5: IMPORT FAILED: {exc}")
        raise

    _setup_logging()
    _t("step6: logging configured")

    root = tk.Tk()
    _t("step7: Tk() created")

    _app = PlanParserGUI(root)  # noqa: F841
    _t("step8: PlanParserGUI created")

    try:
        splash_proc.terminate()
        splash_proc.wait(timeout=2)
    except Exception:  # noqa: BLE001
        pass
    _t("step9: splash killed, entering mainloop")

    root.mainloop()
    _t("step10: mainloop exited")


_run()
