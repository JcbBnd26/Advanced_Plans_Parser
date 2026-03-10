"""Surya OCR backend implementation.

Surya is a lightweight, thread-safe OCR library with excellent accuracy
for document processing. It uses transformer-based models for both
text detection and recognition.

See: https://github.com/VikParuchuri/surya
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List

import numpy as np

from .base import OCRBackend, TextBox

if TYPE_CHECKING:
    from ...config import GroupingConfig

log = logging.getLogger(__name__)


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

        # Get languages from config (default English)
        # Config stores as comma-separated string: "en,de,fr"
        self._languages = ["en"]
        if cfg is not None:
            langs_str = getattr(cfg, "surya_languages", "en")
            if langs_str:
                self._languages = [lang.strip() for lang in langs_str.split(",") if lang.strip()]

    def _ensure_initialized(self) -> None:
        """Lazy-load Surya models on first use."""
        if self._initialized:
            return

        log.info("Initializing Surya OCR (device=%s, langs=%s)...",
                 self._device, self._languages)

        try:
            from surya.detection import DetectionPredictor
            from surya.recognition import RecognitionPredictor
        except ImportError as e:
            raise ImportError(
                "Surya OCR is not installed. Install with: "
                "pip install surya-ocr"
            ) from e

        # Configure device
        device_str = "cuda" if self._device == "gpu" else "cpu"

        # Initialize predictors
        self._det_predictor = DetectionPredictor(device=device_str)
        self._rec_predictor = RecognitionPredictor(device=device_str)

        self._initialized = True
        log.info("Surya OCR initialized successfully")

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

        # Run detection + recognition
        # Surya's predict method takes a list of images
        predictions = self._rec_predictor(
            [pil_image],
            [self._languages],
            self._det_predictor,
        )

        results: List[TextBox] = []

        if not predictions:
            return results

        # Process first (and only) page result
        page_result = predictions[0]

        for text_line in page_result.text_lines:
            # Skip low-confidence or empty results
            text = text_line.text.strip() if text_line.text else ""
            if not text:
                continue

            confidence = getattr(text_line, "confidence", 1.0)

            # Surya bbox is [x0, y0, x1, y1] - convert to polygon
            bbox = text_line.bbox
            polygon = self._bbox_to_polygon(bbox)

            results.append(TextBox(
                polygon=polygon,
                text=text,
                confidence=float(confidence),
            ))

        return results

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
            [self._languages] * len(pil_images),
            self._det_predictor,
        )

        batch_results: List[List[TextBox]] = []

        for page_result in predictions:
            results: List[TextBox] = []
            for text_line in page_result.text_lines:
                text = text_line.text.strip() if text_line.text else ""
                if not text:
                    continue

                confidence = getattr(text_line, "confidence", 1.0)
                bbox = text_line.bbox
                polygon = self._bbox_to_polygon(bbox)

                results.append(TextBox(
                    polygon=polygon,
                    text=text,
                    confidence=float(confidence),
                ))
            batch_results.append(results)

        return batch_results

    @staticmethod
    def _bbox_to_polygon(bbox: List[float]) -> List[List[float]]:
        """Convert [x0, y0, x1, y1] bbox to 4-point polygon.

        Returns corners in order: top-left, top-right, bottom-right, bottom-left.
        """
        x0, y0, x1, y1 = bbox
        return [
            [x0, y0],  # top-left
            [x1, y0],  # top-right
            [x1, y1],  # bottom-right
            [x0, y1],  # bottom-left
        ]
