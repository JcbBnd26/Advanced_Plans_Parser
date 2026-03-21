"""Surya OCR backend implementation.

Surya is a lightweight, thread-safe OCR library with excellent accuracy
for document processing. It uses transformer-based models for both
text detection and recognition.

See: https://github.com/VikParuchuri/surya
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from typing import TYPE_CHECKING, List, Tuple

import numpy as np

from ...config.exceptions import OCRBackendTimeoutError
from .base import OCRBackend, TextBox

if TYPE_CHECKING:
    from ...config import GroupingConfig

log = logging.getLogger(__name__)


def _preflight_environment() -> dict[str, str]:
    """Return an environment copy with conservative BLAS thread settings."""
    env = os.environ.copy()
    for var in (
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        env.setdefault(var, "1")
    return env


class SuryaOCRBackend(OCRBackend):
    """Surya-based OCR backend.

    Thread-safe implementation using Surya's detection and recognition
    predictors. Models are loaded lazily on first prediction.

    Parameters
    ----------
    device : str
        Device to run on: "cpu" or "gpu" (CUDA).
    cfg : GroupingConfig, optional
        Configuration for language selection and other options.
    """

    def __init__(
        self,
        device: str = "cpu",
        cfg: "GroupingConfig | None" = None,
    ) -> None:
        self._device = device
        self._cfg = cfg
        self._det_predictor = None
        self._rec_predictor = None
        self._initialized = False
        self._init_error = None
        self._preflight_complete = False

        # Get languages from config (default English)
        # Config stores as comma-separated string: "en,de,fr"
        self._languages = ["en"]
        if cfg is not None:
            langs_str = getattr(cfg, "surya_languages", "en")
            if langs_str:
                self._languages = [
                    lang.strip() for lang in langs_str.split(",") if lang.strip()
                ]

    def _load_predictor_classes(self) -> Tuple[type, type, type]:
        """Import and return the Surya predictor classes."""
        try:
            from surya.detection import DetectionPredictor
            from surya.recognition import FoundationPredictor, RecognitionPredictor
        except ImportError as exc:
            raise ImportError(
                "Surya OCR is not installed. Install with: pip install surya-ocr"
            ) from exc

        return DetectionPredictor, FoundationPredictor, RecognitionPredictor

    def _run_import_preflight(self) -> None:
        """Check the Surya import path in a child process before in-process init."""
        if self._preflight_complete:
            return

        timeout = 45
        if self._cfg is not None:
            timeout = getattr(self._cfg, "surya_init_timeout_sec", timeout)

        code = (
            "from surya.detection import DetectionPredictor\n"
            "from surya.recognition import FoundationPredictor, RecognitionPredictor\n"
            "print('ok')\n"
        )
        log.info("Running Surya import preflight (timeout=%ds)", timeout)
        try:
            completed = subprocess.run(
                [sys.executable, "-u", "-c", code],
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=timeout,
                check=False,
                env=_preflight_environment(),
            )
        except subprocess.TimeoutExpired as exc:
            detail = ((exc.stderr or "") or (exc.stdout or "")).strip()
            if detail:
                detail = detail.splitlines()[-1]
                message = f"Surya import preflight timed out after {timeout}s: {detail}"
            else:
                message = f"Surya import preflight timed out after {timeout}s"
            raise OCRBackendTimeoutError(message) from exc

        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "").strip()
            if detail:
                detail = detail.splitlines()[-1]
                raise ImportError(f"Surya import preflight failed: {detail}")
            raise ImportError("Surya import preflight failed")

        self._preflight_complete = True

    def _ensure_initialized(self) -> None:
        """Lazy-load Surya models on first use."""
        if self._initialized:
            return
        if self._init_error is not None:
            raise self._init_error

        # Prevent OpenBLAS memory-allocation crashes on Windows when
        # multiple BLAS-linked ML stacks share the same environment.
        import gc

        for var, value in _preflight_environment().items():
            if var in (
                "OPENBLAS_NUM_THREADS",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            ):
                os.environ.setdefault(var, value)

        # Reclaim any accumulated garbage before loading heavy models
        # (~1–1.5 GB for detection + recognition transformers).
        gc.collect()

        log.info(
            "Initializing Surya OCR (device=%s, langs=%s)...",
            self._device,
            self._languages,
        )
        try:
            self._run_import_preflight()
            DetectionPredictor, FoundationPredictor, RecognitionPredictor = (
                self._load_predictor_classes()
            )

            # Configure device
            device_str = "cuda" if self._device == "gpu" else "cpu"

            # Initialize predictors (RecognitionPredictor wraps a FoundationPredictor)
            self._det_predictor = DetectionPredictor(device=device_str)
            foundation = FoundationPredictor(device=device_str)
            self._rec_predictor = RecognitionPredictor(foundation)

            self._initialized = True
            log.info("Surya OCR initialized successfully")
        except Exception as exc:
            self._init_error = exc
            log.error("Surya OCR initialization failed", exc_info=True)
            raise

    @property
    def name(self) -> str:
        """Backend identifier."""
        return "surya"

    def predict(self, image: np.ndarray) -> List[TextBox]:
        """Run OCR on an image.

        Parameters
        ----------
        image : np.ndarray
            RGB image as HxWx3 numpy array (uint8).

        Returns
        -------
        List[TextBox]
            List of detected text boxes with pixel coordinates.
        """
        self._ensure_initialized()

        from PIL import Image

        # Convert numpy to PIL (Surya expects PIL images)
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)
        pil_image = Image.fromarray(image)

        log.debug(
            "Surya predict: image %dx%d px",
            pil_image.width,
            pil_image.height,
        )

        # Run detection + recognition
        predictions = self._rec_predictor(
            [pil_image],
            det_predictor=self._det_predictor,
        )

        return self._extract_textboxes(predictions[0]) if predictions else []

    def predict_batch(self, images: List[np.ndarray]) -> List[List[TextBox]]:
        """Run OCR on multiple images in a single Surya call.

        Surya natively supports batching, so this is more efficient than
        calling predict() sequentially for each image.

        Parameters
        ----------
        images : List[np.ndarray]
            List of RGB images as HxWx3 numpy arrays (uint8).

        Returns
        -------
        List[List[TextBox]]
            List of results, one per input image.
        """
        if not images:
            return []

        self._ensure_initialized()

        from PIL import Image

        # Convert all images to PIL
        pil_images = []
        for image in images:
            if image.dtype != np.uint8:
                image = image.astype(np.uint8)
            pil_images.append(Image.fromarray(image))

        # Run batch detection + recognition
        predictions = self._rec_predictor(
            pil_images,
            det_predictor=self._det_predictor,
        )

        return [self._extract_textboxes(p) for p in predictions]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_textboxes(page_result) -> List[TextBox]:
        """Convert a Surya OCRResult into a list of TextBox."""
        results: List[TextBox] = []
        for text_line in page_result.text_lines:
            text = text_line.text.strip() if text_line.text else ""
            if not text:
                continue

            confidence = getattr(text_line, "confidence", 1.0)

            # Surya 0.17+ provides polygon directly on TextLine
            polygon = text_line.polygon
            if polygon is None:
                # Fallback: derive from bbox [x0, y0, x1, y1]
                x0, y0, x1, y1 = text_line.bbox
                polygon = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]

            results.append(
                TextBox(
                    polygon=[[float(c) for c in pt] for pt in polygon],
                    text=text,
                    confidence=float(confidence),
                )
            )
        return results
