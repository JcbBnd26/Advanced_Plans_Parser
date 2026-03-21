"""Standalone splash screen — launched as a subprocess by __main__.py.

Displays a centered splash with a counting timer until the process
is terminated by the parent.
"""

import time
import tkinter as tk


def main() -> None:
    root = tk.Tk()
    root.title("")
    root.overrideredirect(True)
    root.attributes("-topmost", True)
    sw, sh = 340, 110
    x = (root.winfo_screenwidth() - sw) // 2
    y = (root.winfo_screenheight() - sh) // 2
    root.geometry(f"{sw}x{sh}+{x}+{y}")

    tk.Label(
        root,
        text="Advanced Plans Parser\nLoading\u2026",
        font=("Segoe UI", 12),
    ).pack(expand=True, pady=(12, 0))

    timer_label = tk.Label(root, text="0.0s", font=("Segoe UI", 9), fg="#666666")
    timer_label.pack(pady=(0, 8))

    t0 = time.perf_counter()

    def _tick() -> None:
        timer_label.configure(text=f"{time.perf_counter() - t0:.1f}s")
        root.after(100, _tick)

    root.after(100, _tick)
    root.mainloop()


if __name__ == "__main__":
    main()
