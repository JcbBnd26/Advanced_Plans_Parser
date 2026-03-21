"""Diagnostic script to trace where pythonw hangs during GUI startup."""

import sys
import time

LOG = "logs/startup_debug.log"


def log(msg):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(f"[{time.perf_counter():8.2f}] {msg}\n")


log("step1: starting")

import tkinter as tk

log("step2: tk imported")

from scripts.gui.gui import PlanParserGUI, _setup_logging

log("step3: gui.py imported")

_setup_logging()
log("step4: logging configured")

root = tk.Tk()
log("step5: root Tk created")

app = PlanParserGUI(root)
log("step6: PlanParserGUI created — ready!")

# Auto-close after 3 seconds so we can read the log
root.after(3000, root.destroy)
root.mainloop()
log("step7: mainloop exited")
