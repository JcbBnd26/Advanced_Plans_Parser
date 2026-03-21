"""Trace which import hangs under pythonw."""

from __future__ import annotations

import os
import sys
import time

# Fix BOTH Python-level and C-level file descriptors for pythonw
_devnull_fd = os.open(os.devnull, os.O_RDWR)
if sys.stdout is None:
    os.dup2(_devnull_fd, 1)
    sys.stdout = open(os.devnull, "w", encoding="utf-8")  # noqa: SIM115
if sys.stderr is None:
    os.dup2(_devnull_fd, 2)
    sys.stderr = open(os.devnull, "w", encoding="utf-8")  # noqa: SIM115
if sys.stdin is None:
    os.dup2(_devnull_fd, 0)
    sys.stdin = open(os.devnull, "r", encoding="utf-8")  # noqa: SIM115
os.close(_devnull_fd)

LOG = "logs/import_trace.log"
os.makedirs("logs", exist_ok=True)

with open(LOG, "w", encoding="utf-8") as f:
    f.write("")


def _t(msg: str) -> None:
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(f"[{time.perf_counter():10.2f}] {msg}\n")
        f.flush()


_t(f"exe={sys.executable}")

_t("1. PIL...")
from PIL import Image, ImageDraw  # noqa: E402, F401

_t("   OK")

_t("2. pymupdf...")
import pymupdf  # noqa: E402, F401

_t("   OK")

_t("3. numpy...")
import numpy  # noqa: E402, F401

_t("   OK")

_t("4. plancheck (full)...")
import plancheck  # noqa: E402, F401

_t("   OK")

_t("ALL DONE")
