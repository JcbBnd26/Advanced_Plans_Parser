"""Minimal pipeline runner that captures errors to a file."""

import os
import subprocess
import sys

os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "1"

result = subprocess.run(
    [
        sys.executable,
        "-u",
        "scripts/run_pdf_batch.py",
        "input/IFC Operations Facilities McClain County - Drawings 25_0915.pdf",
        "--start",
        "2",
        "--end",
        "3",
        "--ocr-full-reconcile",
        "--ocr-debug",
    ],
    env={**os.environ, "PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK": "1"},
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    encoding="utf-8",
    errors="replace",
)

with open("run_page2_err.txt", "w", encoding="utf-8") as f:
    f.write(result.stdout)
    if result.returncode != 0:
        f.write(f"\nExit code: {result.returncode}\n")

print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
print(f"\nExit code: {result.returncode}")
