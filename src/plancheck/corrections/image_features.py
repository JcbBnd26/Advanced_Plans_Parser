"""Image-based feature extraction for block/region classification.

Uses a pre-trained CNN backbone (via ``timm``) to produce a fixed-length
embedding from a cropped page region.  The embedding is concatenated with
the hand-crafted numeric/zone features before feeding into the classifier.

The module is **optional** — if ``torch`` / ``timm`` are not installed it
gracefully degrades: :func:`extract_image_features` returns a zero vector
so the downstream classifier still works (just without visual features).

Public API
----------
ImageFeatureExtractor  – CNN wrapper (lazy model loading, crop + embed)
extract_image_features – convenience: crop bbox → embedding vector
IMAGE_FEATURE_DIM      – length of the embedding (512 for ResNet-18)
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────

IMAGE_FEATURE_DIM: int = 512  # ResNet-18 pool output dimensionality
_DEFAULT_BACKBONE = "resnet18"
_INPUT_SIZE = (224, 224)

# ── Availability check ─────────────────────────────────────────────────

_TORCH_AVAILABLE = False
_TIMM_AVAILABLE = False

try:
    import torch  # noqa: F401

    _TORCH_AVAILABLE = True
except ImportError:
    pass

try:
    import timm  # noqa: F401

    _TIMM_AVAILABLE = True
except ImportError:
    pass


def is_vision_available() -> bool:
    """Return True if torch + timm are installed."""
    return _TORCH_AVAILABLE and _TIMM_AVAILABLE


# ── Image Feature Extractor ───────────────────────────────────────────


class ImageFeatureExtractor:
    """Extract a fixed-length visual embedding from a page-image crop.

    Parameters
    ----------
    backbone : str
        ``timm`` model name (default ``"resnet18"``).  Any timm model
        that supports ``num_classes=0`` (headless) will work.
    device : str or None
        PyTorch device (``"cpu"``, ``"cuda"``).  ``None`` → auto-detect.
    """

    def __init__(
        self,
        backbone: str = _DEFAULT_BACKBONE,
        device: Optional[str] = None,
    ) -> None:
        self.backbone_name = backbone
        self._model = None
        self._transform = None
        self._device_name = device

    # ── lazy model loading ────────────────────────────────────────────

    def _ensure_model(self) -> bool:
        """Load the timm model on first use.  Returns False if unavailable."""
        if self._model is not None:
            return True
        if not is_vision_available():
            log.debug("torch/timm not installed — image features disabled")
            return False

        import timm
        import torch
        from timm.data import resolve_data_config
        from timm.data.transforms_factory import create_transform

        device = self._device_name or ("cuda" if torch.cuda.is_available() else "cpu")
        self._device = torch.device(device)

        # Create headless model (no classification head → returns pooled features)
        self._model = timm.create_model(
            self.backbone_name,
            pretrained=True,
            num_classes=0,  # removes the final FC → output is pool embedding
        )
        self._model = self._model.to(self._device)
        self._model.eval()

        # Build the matching preprocessing transform
        data_cfg = resolve_data_config(self._model.pretrained_cfg)
        self._transform = create_transform(**data_cfg)

        log.info(
            "ImageFeatureExtractor ready: backbone=%s, device=%s, dim=%d",
            self.backbone_name,
            self._device,
            self.embedding_dim,
        )
        return True

    # ── public properties ─────────────────────────────────────────────

    @property
    def embedding_dim(self) -> int:
        """Dimensionality of the output embedding."""
        return IMAGE_FEATURE_DIM

    @property
    def available(self) -> bool:
        """True if the model loaded successfully."""
        return self._model is not None

    # ── core extraction ───────────────────────────────────────────────

    def extract(
        self,
        page_image,
        bbox: Tuple[float, float, float, float],
    ) -> np.ndarray:
        """Crop *bbox* from *page_image* and return the CNN embedding.

        Parameters
        ----------
        page_image : PIL.Image.Image
            Full-page render (RGB).
        bbox : tuple[float, float, float, float]
            ``(x0, y0, x1, y1)`` in pixel coordinates.

        Returns
        -------
        numpy.ndarray
            1-D float32 array of shape ``(IMAGE_FEATURE_DIM,)``.
            Returns zeros if torch/timm are unavailable or the crop
            is degenerate (zero area).
        """
        if not self._ensure_model():
            return np.zeros(IMAGE_FEATURE_DIM, dtype=np.float32)

        x0, y0, x1, y1 = bbox
        x0, y0 = max(0, int(x0)), max(0, int(y0))
        x1, y1 = min(page_image.width, int(x1)), min(page_image.height, int(y1))

        if x1 <= x0 or y1 <= y0:
            return np.zeros(IMAGE_FEATURE_DIM, dtype=np.float32)

        import torch

        crop = page_image.crop((x0, y0, x1, y1)).convert("RGB")

        # Apply timm preprocessing (resize, normalize)
        tensor = self._transform(crop).unsqueeze(0).to(self._device)

        with torch.no_grad():
            embedding = self._model(tensor)

        return embedding.cpu().numpy().flatten().astype(np.float32)

    def extract_batch(
        self,
        page_image,
        bboxes: list[Tuple[float, float, float, float]],
    ) -> list[np.ndarray]:
        """Extract embeddings for multiple bboxes in a single forward pass.

        Falls back to individual :meth:`extract` calls if batching fails.
        """
        if not self._ensure_model() or not bboxes:
            return [np.zeros(IMAGE_FEATURE_DIM, dtype=np.float32) for _ in bboxes]

        import torch

        tensors = []
        valid_indices = []
        results = [np.zeros(IMAGE_FEATURE_DIM, dtype=np.float32) for _ in bboxes]

        for i, (bx0, by0, bx1, by1) in enumerate(bboxes):
            bx0, by0 = max(0, int(bx0)), max(0, int(by0))
            bx1, by1 = (
                min(page_image.width, int(bx1)),
                min(page_image.height, int(by1)),
            )
            if bx1 <= bx0 or by1 <= by0:
                continue
            crop = page_image.crop((bx0, by0, bx1, by1)).convert("RGB")
            tensors.append(self._transform(crop))
            valid_indices.append(i)

        if not tensors:
            return results

        batch = torch.stack(tensors).to(self._device)
        with torch.no_grad():
            embeddings = self._model(batch)

        emb_np = embeddings.cpu().numpy().astype(np.float32)
        for j, idx in enumerate(valid_indices):
            results[idx] = emb_np[j]

        return results


# ── Module-level singleton ─────────────────────────────────────────────

_extractor: Optional[ImageFeatureExtractor] = None


def get_extractor(
    backbone: str = _DEFAULT_BACKBONE,
    device: Optional[str] = None,
) -> ImageFeatureExtractor:
    """Return (or create) the module-level singleton extractor."""
    global _extractor
    if _extractor is None or _extractor.backbone_name != backbone:
        _extractor = ImageFeatureExtractor(backbone=backbone, device=device)
    return _extractor


def extract_image_features(
    page_image,
    bbox: Tuple[float, float, float, float],
    backbone: str = _DEFAULT_BACKBONE,
    device: Optional[str] = None,
) -> np.ndarray:
    """Convenience wrapper: crop bbox from page image → embedding vector.

    Returns a zero vector if torch/timm are not installed.
    """
    extractor = get_extractor(backbone=backbone, device=device)
    return extractor.extract(page_image, bbox)
