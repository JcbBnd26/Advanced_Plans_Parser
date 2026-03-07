"""Test PaddleOCR with ProcessPoolExecutor from a thread."""

from __future__ import annotations

import os

os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

import concurrent.futures
import multiprocessing
import threading
from pathlib import Path


def ocr_page(page_idx: int) -> int:
    """Process a single page in a subprocess."""
    import numpy as np
    import pdfplumber

    from plancheck.config import GroupingConfig
    from plancheck.vocr.engine import _get_ocr

    pdf_path = Path(
        r"input/IFC Operations Facilities McClain County - Drawings 25_0915.pdf"
    )

    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[page_idx]
        img = page.to_image(resolution=150).original
        if img.mode != "RGB":
            img = img.convert("RGB")
        arr = np.array(img)

        cfg = GroupingConfig()
        ocr = _get_ocr(cfg)

        result = list(ocr.predict(arr))
        texts = result[0].get("rec_texts", []) if result else []
        return len(texts)


def worker_thread():
    """Simulates GUI worker thread using ProcessPoolExecutor for OCR."""
    print("[Thread] Starting worker...")

    # Use ProcessPoolExecutor to run OCR in subprocesses from within a thread
    with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
        for i in range(3):
            print(f"[Thread] Submitting page {i}...")
            future = executor.submit(ocr_page, i)
            result = future.result(timeout=300)
            print(f"[Thread] Page {i}: {result} text items")

    print("[Thread] Worker done!")


if __name__ == "__main__":
    multiprocessing.freeze_support()

    print("[Main] Starting worker thread...")
    t = threading.Thread(target=worker_thread, daemon=True)
    t.start()
    t.join(timeout=600)

    if t.is_alive():
        print("[Main] Thread still running!")
    else:
        print("[Main] SUCCESS!")
