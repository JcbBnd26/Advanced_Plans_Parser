"""Quick import test — writes results to logs/import_test_result.txt

Tests each import in a fresh subprocess with a 15-second timeout.
"""

import os
import subprocess
import sys
import time

os.makedirs("logs", exist_ok=True)
OUT = "logs/import_test_result.txt"

modules = [
    ("PIL", "from PIL import Image"),
    ("numpy", "import numpy"),
    ("reportlab", "import reportlab"),
    (
        "plancheck.config.pipeline",
        "from plancheck.config.pipeline import PipelineConfig",
    ),
    ("plancheck.models", "from plancheck.models import GlyphBox"),
    (
        "plancheck.export.csv_export",
        "from plancheck.export.csv_export import export_notes_csv",
    ),
    ("plancheck.export.font_map", "from plancheck.export.font_map import resolve_font"),
    (
        "plancheck.export.page_data",
        "from plancheck.export.page_data import serialize_page",
    ),
    (
        "plancheck.export.overlays.colors",
        "from plancheck.export.overlays.colors import _get_color",
    ),
    (
        "plancheck.export.overlays.detection",
        "from plancheck.export.overlays.detection import draw_overlay",
    ),
    (
        "plancheck.export.overlays.structural",
        "from plancheck.export.overlays.structural import draw_columns_overlay",
    ),
    (
        "plancheck.reconcile.helpers",
        "from plancheck.reconcile.helpers import SymbolCandidate",
    ),
    ("plancheck.vocr.extract", "from plancheck.vocr.extract import extract_ocr_tokens"),
    (
        "plancheck.reconcile.reconcile",
        "from plancheck.reconcile.reconcile import ReconcileResult",
    ),
    (
        "plancheck.export.reconcile_overlay",
        "from plancheck.export.reconcile_overlay import draw_reconcile_debug",
    ),
    (
        "plancheck.export.sheet_recreation",
        "from plancheck.export.sheet_recreation import recreate_sheet",
    ),
    ("plancheck.ingest", "from plancheck.ingest import ingest_pdf"),
    ("plancheck (full)", "import plancheck"),
]

with open(OUT, "w", encoding="utf-8") as f:
    f.write(f"python: {sys.version}\n")
    f.write(f"exe: {sys.executable}\n\n")

    for name, stmt in modules:
        f.write(f"{name}: ")
        f.flush()
        t0 = time.perf_counter()
        try:
            r = subprocess.run(
                [sys.executable, "-c", stmt],
                timeout=15,
                capture_output=True,
                text=True,
            )
            dt = time.perf_counter() - t0
            if r.returncode == 0:
                f.write(f"OK ({dt:.1f}s)\n")
            else:
                err = r.stderr.strip().split("\n")[-1] if r.stderr else "unknown"
                f.write(f"FAIL ({dt:.1f}s): {err}\n")
        except subprocess.TimeoutExpired:
            dt = time.perf_counter() - t0
            f.write(f"TIMEOUT ({dt:.1f}s) — HUNG!\n")
        f.flush()

    f.write("\nDONE\n")
