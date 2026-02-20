from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from PIL import Image, ImageFilter, ImageOps


@dataclass
class OcrPreprocessConfig:
    """Standalone OCR image preprocessing configuration.

    This module is intentionally independent from the main parsing pipeline.
    It can be run as a pre-stage and its output image can be passed into the
    OCR entry point later.
    """

    enabled: bool = True
    grayscale: bool = True
    autocontrast: bool = False
    clahe: bool = True
    clahe_clip_limit: float = 2.0
    clahe_tile_size: int = 8
    median_denoise: bool = False
    median_kernel_size: int = 3
    adaptive_binarize: bool = False
    adaptive_block_size: int = 11
    adaptive_c: float = 2.0
    sharpen: bool = False
    sharpen_radius: int = 2
    sharpen_percent: int = 140
    save_intermediate: bool = False


@dataclass
class OcrPreprocessResult:
    """Result payload emitted by the standalone OCR preprocessing pipeline."""

    image: Image.Image
    applied_steps: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)


def _require_odd(value: int, floor: int = 3) -> int:
    """Clamp *value* to at least *floor* and ensure it is odd."""
    value = max(floor, int(value))
    return value if value % 2 == 1 else value + 1


def _as_uint8_gray(img: Image.Image):
    """Convert a PIL image to a uint8 grayscale numpy array."""
    import numpy as np

    if img.mode != "L":
        img = img.convert("L")
    arr = np.array(img)
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    return arr


def _measure_metrics(img: Image.Image) -> Dict[str, float]:
    """Compute basic image statistics (mean, std, sharpness, etc.)."""
    import numpy as np

    arr = _as_uint8_gray(img)
    metrics: Dict[str, float] = {
        "width": float(arr.shape[1]),
        "height": float(arr.shape[0]),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }

    try:
        import cv2

        metrics["sharpness_laplacian_var"] = float(cv2.Laplacian(arr, cv2.CV_64F).var())
    except ImportError:
        pass

    return metrics


def preprocess_image_for_ocr(
    image: Image.Image,
    cfg: OcrPreprocessConfig | None = None,
    intermediate_dir: str | None = None,
) -> OcrPreprocessResult:
    """Run standalone OCR preprocessing over a PIL image.

    The output is intended to be passed directly into OCR ingestion in a later
    integration stage ("mouth" of OCR pipeline).
    """
    if cfg is None:
        cfg = OcrPreprocessConfig()

    current = image.copy()
    steps: List[str] = []

    def _save_step(name: str) -> None:
        """Save the current intermediate image if configured."""
        if not (cfg.save_intermediate and intermediate_dir):
            return
        from pathlib import Path

        out_dir = Path(intermediate_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        step_idx = len(steps)
        out_path = out_dir / f"{step_idx:02d}_{name}.png"
        current.save(out_path)

    if not cfg.enabled:
        return OcrPreprocessResult(
            image=current,
            applied_steps=["disabled"],
            metrics=_measure_metrics(current),
        )

    _save_step("input")

    if cfg.grayscale and current.mode != "L":
        current = current.convert("L")
        steps.append("grayscale")
        _save_step("grayscale")

    if cfg.autocontrast:
        current = ImageOps.autocontrast(current)
        steps.append("autocontrast")
        _save_step("autocontrast")

    if cfg.clahe:
        try:
            import cv2

            arr = _as_uint8_gray(current)
            tile_size = max(2, int(cfg.clahe_tile_size))
            clahe = cv2.createCLAHE(
                clipLimit=float(cfg.clahe_clip_limit),
                tileGridSize=(tile_size, tile_size),
            )
            arr = clahe.apply(arr)
            current = Image.fromarray(arr)
            steps.append("clahe")
            _save_step("clahe")
        except ImportError:
            steps.append("clahe_skipped_no_cv2")

    if cfg.median_denoise:
        k = _require_odd(cfg.median_kernel_size, floor=3)
        current = current.filter(ImageFilter.MedianFilter(size=k))
        steps.append("median_denoise")
        _save_step("median_denoise")

    if cfg.adaptive_binarize:
        try:
            import cv2

            arr = _as_uint8_gray(current)
            block_size = _require_odd(cfg.adaptive_block_size, floor=3)
            binary = cv2.adaptiveThreshold(
                arr,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                block_size,
                float(cfg.adaptive_c),
            )
            current = Image.fromarray(binary)
            steps.append("adaptive_binarize")
            _save_step("adaptive_binarize")
        except ImportError:
            steps.append("adaptive_binarize_skipped_no_cv2")

    if cfg.sharpen:
        current = current.filter(
            ImageFilter.UnsharpMask(
                radius=cfg.sharpen_radius, percent=cfg.sharpen_percent
            )
        )
        steps.append("unsharp_mask")
        _save_step("unsharp_mask")

    return OcrPreprocessResult(
        image=current,
        applied_steps=steps,
        metrics=_measure_metrics(current),
    )
