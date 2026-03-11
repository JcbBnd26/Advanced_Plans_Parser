"""LayoutLMv3-based page layout detection and element classification.

Uses Microsoft's LayoutLMv3 (via HuggingFace ``transformers``) to perform
layout-aware page understanding.  The model jointly considers text tokens,
their 2-D bounding boxes, and the page image to predict element types.

The module is **optional** — if ``transformers`` / ``torch`` are not
installed it gracefully degrades: :func:`predict_layout` returns an
empty list so the downstream pipeline still works.

Public API
----------
LayoutModel         – LayoutLMv3 wrapper (lazy loading, tokenize + predict)
predict_layout      – convenience: page data → list of LayoutPrediction
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

log = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────

_DEFAULT_MODEL_NAME = "microsoft/layoutlmv3-base"
_MAX_SEQ_LENGTH = 512

# Map LayoutLMv3 label IDs to our ZoneTag values
LAYOUT_LABELS: list[str] = [
    "border",
    "drawing",
    "notes",
    "title_block",
    "legend",
    "abbreviations",
    "revisions",
    "details",
    "unknown",
]

# ── Availability check ─────────────────────────────────────────────────

_TRANSFORMERS_AVAILABLE = False
_TORCH_AVAILABLE = False

try:
    import torch  # noqa: F401

    _TORCH_AVAILABLE = True
except ImportError:
    pass

try:
    import transformers  # noqa: F401

    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    pass


def is_layout_available() -> bool:
    """Return True if transformers + torch are installed."""
    return _TRANSFORMERS_AVAILABLE and _TORCH_AVAILABLE


# ── Data structures ────────────────────────────────────────────────────


@dataclass
class LayoutPrediction:
    """A single layout element predicted by the model.

    Attributes
    ----------
    label : str
        Predicted element type (e.g. "notes", "title_block").
    confidence : float
        Model confidence (0–1).
    bbox : tuple[float, float, float, float]
        Bounding box ``(x0, y0, x1, y1)`` in page coordinates.
    token_indices : list[int]
        Indices of the input tokens that contributed to this prediction.
    """

    label: str
    confidence: float
    bbox: Tuple[float, float, float, float]
    token_indices: list[int] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "confidence": round(self.confidence, 4),
            "bbox": list(self.bbox),
        }


# ── Layout Model ───────────────────────────────────────────────────────


class LayoutModel:
    """LayoutLMv3-based page layout predictor.

    Parameters
    ----------
    model_name_or_path : str or Path
        HuggingFace model name or path to fine-tuned checkpoint.
        Defaults to ``"microsoft/layoutlmv3-base"``.
    device : str or None
        PyTorch device.  ``None`` → auto-detect.
    num_labels : int
        Number of output labels (default matches :data:`LAYOUT_LABELS`).
    """

    def __init__(
        self,
        model_name_or_path: str | Path = _DEFAULT_MODEL_NAME,
        device: Optional[str] = None,
        num_labels: int | None = None,
    ) -> None:
        self.model_name = str(model_name_or_path)
        self._model = None
        self._processor = None
        self._device_name = device
        self._num_labels = num_labels or len(LAYOUT_LABELS)

    # ── lazy model loading ────────────────────────────────────────────

    def _ensure_model(self) -> bool:
        """Load the LayoutLMv3 model on first use.  Returns False if unavailable."""
        if self._model is not None:
            return True
        if not is_layout_available():
            log.debug("transformers/torch not installed — layout model disabled")
            return False

        # Guard against using the unfine-tuned base checkpoint.  The base
        # model has a random classification head and produces meaningless
        # token-level layout labels.
        if self.model_name == _DEFAULT_MODEL_NAME:
            log.warning(
                "LayoutModel: '%s' is an unfine-tuned base checkpoint with "
                "a random classification head.  Layout predictions will be "
                "meaningless.  Supply a fine-tuned checkpoint path via "
                "ml_layout_model_path to enable layout detection.",
                self.model_name,
            )
            return False

        import torch
        from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Processor

        device = self._device_name or ("cuda" if torch.cuda.is_available() else "cpu")
        self._device = torch.device(device)

        try:
            self._processor = LayoutLMv3Processor.from_pretrained(
                self.model_name,
                apply_ocr=False,  # We supply our own tokens + bboxes
            )
            self._model = LayoutLMv3ForTokenClassification.from_pretrained(
                self.model_name,
                num_labels=self._num_labels,
            )
            self._model = self._model.to(self._device)
            self._model.eval()
        except Exception:  # noqa: BLE001 — model load may fail for many reasons
            log.warning(
                "Failed to load LayoutLMv3 model '%s'",
                self.model_name,
                exc_info=True,
            )
            self._model = None
            return False

        log.info(
            "LayoutModel ready: model=%s, device=%s, labels=%d",
            self.model_name,
            self._device,
            self._num_labels,
        )
        return True

    # ── public properties ─────────────────────────────────────────────

    @property
    def available(self) -> bool:
        """True if the model loaded successfully."""
        return self._model is not None

    @property
    def label_names(self) -> list[str]:
        """The label vocabulary in label-ID order."""
        return LAYOUT_LABELS[: self._num_labels]

    # ── core prediction ───────────────────────────────────────────────

    def predict(
        self,
        page_image,
        tokens: Sequence[Any],
        page_width: float,
        page_height: float,
    ) -> list[LayoutPrediction]:
        """Predict layout labels for page tokens.

        Parameters
        ----------
        page_image : PIL.Image.Image
            Full-page render (RGB).
        tokens : sequence[GlyphBox]
            Page tokens with ``text``, ``x0``, ``y0``, ``x1``, ``y1``.
        page_width, page_height : float
            Page dimensions in PDF points.

        Returns
        -------
        list[LayoutPrediction]
            Per-token layout predictions, grouped into contiguous regions.
        """
        if not self._ensure_model():
            return []

        if not tokens:
            return []

        import torch

        # Prepare inputs: text + normalized bboxes (0–1000 for LayoutLMv3)
        words = []
        boxes = []
        for t in tokens:
            words.append(t.text if hasattr(t, "text") else str(t))
            x0 = max(0, int(t.x0 / page_width * 1000))
            y0 = max(0, int(t.y0 / page_height * 1000))
            x1 = min(1000, int(t.x1 / page_width * 1000))
            y1 = min(1000, int(t.y1 / page_height * 1000))
            # Clamp: x1 >= x0, y1 >= y0
            x1 = max(x0, x1)
            y1 = max(y0, y1)
            boxes.append([x0, y0, x1, y1])

        # Truncate to max sequence length
        if len(words) > _MAX_SEQ_LENGTH:
            words = words[:_MAX_SEQ_LENGTH]
            boxes = boxes[:_MAX_SEQ_LENGTH]

        # Process through LayoutLMv3Processor
        page_rgb = page_image.convert("RGB")

        encoding = self._processor(
            page_rgb,
            words,
            boxes=boxes,
            return_tensors="pt",
            truncation=True,
            max_length=_MAX_SEQ_LENGTH,
            padding="max_length",
        )

        # Move to device
        encoding = {k: v.to(self._device) for k, v in encoding.items()}

        with torch.no_grad():
            outputs = self._model(**encoding)

        logits = outputs.logits[0]  # (seq_len, num_labels)
        probs = torch.softmax(logits, dim=-1)
        pred_ids = torch.argmax(probs, dim=-1).cpu().numpy()
        pred_confs = probs.max(dim=-1).values.cpu().numpy()

        # Map back to original tokens (skip special tokens)
        # LayoutLMv3 adds [CLS] at start, [SEP] at end, plus subword tokens.
        # Use word_ids() to correctly align subword predictions back to
        # original words rather than assuming a 1:1 offset.
        predictions = self._aggregate_predictions(
            pred_ids,
            pred_confs,
            tokens,
            boxes,
            page_width,
            page_height,
            encoding=encoding,
        )

        return predictions

    def _aggregate_predictions(
        self,
        pred_ids: np.ndarray,
        pred_confs: np.ndarray,
        tokens: Sequence[Any],
        boxes: list[list[int]],
        page_width: float,
        page_height: float,
        encoding: Any = None,
    ) -> list[LayoutPrediction]:
        """Aggregate per-subword predictions into per-region predictions.

        Uses ``word_ids()`` from the tokenizer encoding to correctly map
        subword tokens back to the original words, handling multi-piece
        tokenisation.  Falls back to a fixed CLS offset of 1 when
        ``word_ids()`` is unavailable (e.g. when *encoding* is ``None``).

        Groups contiguous tokens with the same label into a single
        LayoutPrediction with a merged bounding box.
        """
        n_tokens = min(len(tokens), _MAX_SEQ_LENGTH)

        # Build the subword→word mapping.  word_ids()[i] is the 0-based
        # word index for subword position *i*, or None for special tokens.
        word_id_map: list[int | None] | None = None
        if encoding is not None:
            try:
                word_id_map = encoding.word_ids(batch_index=0)
            except Exception:  # noqa: BLE001
                log.debug("word_ids() unavailable, falling back to CLS offset")

        # Per-word accumulator: word_idx → (label_id_sum, conf_sum, count)
        word_labels: dict[int, tuple[list[int], float, int]] = {}

        if word_id_map is not None:
            for subword_idx, wid in enumerate(word_id_map):
                if wid is None:
                    continue  # special token ([CLS], [SEP], padding)
                if wid >= n_tokens:
                    continue
                if subword_idx >= len(pred_ids):
                    break
                lid = int(pred_ids[subword_idx])
                conf = float(pred_confs[subword_idx])
                if wid not in word_labels:
                    word_labels[wid] = ([lid], conf, 1)
                else:
                    ids, cs, cnt = word_labels[wid]
                    ids.append(lid)
                    word_labels[wid] = (ids, cs + conf, cnt + 1)

            # Resolve each word — majority vote across its subwords
            word_pred: dict[int, tuple[str, float]] = {}
            for wid, (lids, cs, cnt) in word_labels.items():
                from collections import Counter

                most_common_lid = Counter(lids).most_common(1)[0][0]
                if most_common_lid >= len(LAYOUT_LABELS):
                    most_common_lid = len(LAYOUT_LABELS) - 1
                word_pred[wid] = (LAYOUT_LABELS[most_common_lid], cs / cnt)
        else:
            # Fallback: fixed CLS offset=1 (original behaviour)
            offset = 1
            word_pred = {}
            for i in range(n_tokens):
                idx = i + offset
                if idx >= len(pred_ids):
                    break
                lid = int(pred_ids[idx])
                if lid >= len(LAYOUT_LABELS):
                    lid = len(LAYOUT_LABELS) - 1
                word_pred[i] = (LAYOUT_LABELS[lid], float(pred_confs[idx]))

        # Group contiguous words with the same label into regions
        results: list[LayoutPrediction] = []
        current_label: str | None = None
        current_bbox: list[float] = [0, 0, 0, 0]
        current_indices: list[int] = []
        current_conf_sum: float = 0.0

        for i in range(n_tokens):
            if i not in word_pred:
                continue
            label, conf = word_pred[i]

            if label != current_label and current_label is not None:
                if current_indices:
                    avg_conf = current_conf_sum / len(current_indices)
                    results.append(
                        LayoutPrediction(
                            label=current_label,
                            confidence=avg_conf,
                            bbox=tuple(current_bbox),
                            token_indices=current_indices,
                        )
                    )
                current_label = label
                t = tokens[i]
                current_bbox = [t.x0, t.y0, t.x1, t.y1]
                current_indices = [i]
                current_conf_sum = conf
            elif current_label is None:
                current_label = label
                t = tokens[i]
                current_bbox = [t.x0, t.y0, t.x1, t.y1]
                current_indices = [i]
                current_conf_sum = conf
            else:
                t = tokens[i]
                current_bbox[0] = min(current_bbox[0], t.x0)
                current_bbox[1] = min(current_bbox[1], t.y0)
                current_bbox[2] = max(current_bbox[2], t.x1)
                current_bbox[3] = max(current_bbox[3], t.y1)
                current_indices.append(i)
                current_conf_sum += conf

        if current_label is not None and current_indices:
            avg_conf = current_conf_sum / len(current_indices)
            results.append(
                LayoutPrediction(
                    label=current_label,
                    confidence=avg_conf,
                    bbox=tuple(current_bbox),
                    token_indices=current_indices,
                )
            )

        return results

    # ── Fine-tuning support ───────────────────────────────────────────

    def prepare_training_data(
        self,
        page_image,
        tokens: Sequence[Any],
        labels: Sequence[str],
        page_width: float,
        page_height: float,
    ) -> dict | None:
        """Prepare a single training example for fine-tuning.

        Parameters
        ----------
        page_image : PIL.Image.Image
            Full-page render.
        tokens : sequence[GlyphBox]
            Page tokens.
        labels : sequence[str]
            Ground-truth label for each token.
        page_width, page_height : float
            Page dimensions.

        Returns
        -------
        dict or None
            Encoded inputs ready for the model, or None if unavailable.
        """
        if not is_layout_available():
            return None

        words = [t.text for t in tokens]
        boxes = []
        for t in tokens:
            x0 = max(0, int(t.x0 / page_width * 1000))
            y0 = max(0, int(t.y0 / page_height * 1000))
            x1 = min(1000, int(t.x1 / page_width * 1000))
            y1 = min(1000, int(t.y1 / page_height * 1000))
            x1, y1 = max(x0, x1), max(y0, y1)
            boxes.append([x0, y0, x1, y1])

        # Truncate
        if len(words) > _MAX_SEQ_LENGTH:
            words = words[:_MAX_SEQ_LENGTH]
            boxes = boxes[:_MAX_SEQ_LENGTH]
            labels = labels[:_MAX_SEQ_LENGTH]

        # Build label IDs
        label_to_id = {l: i for i, l in enumerate(LAYOUT_LABELS)}
        label_ids = [label_to_id.get(l, label_to_id["unknown"]) for l in labels]

        if self._processor is None:
            from transformers import LayoutLMv3Processor

            self._processor = LayoutLMv3Processor.from_pretrained(
                self.model_name,
                apply_ocr=False,
            )

        page_rgb = page_image.convert("RGB")
        encoding = self._processor(
            page_rgb,
            words,
            boxes=boxes,
            return_tensors="pt",
            truncation=True,
            max_length=_MAX_SEQ_LENGTH,
            padding="max_length",
        )

        import torch

        # Pad label IDs to max_length (use -100 for ignored positions)
        padded_labels = [-100] + label_ids  # CLS token gets -100
        while len(padded_labels) < _MAX_SEQ_LENGTH:
            padded_labels.append(-100)
        padded_labels = padded_labels[:_MAX_SEQ_LENGTH]
        encoding["labels"] = torch.tensor([padded_labels])

        return encoding

    def save(self, path: str | Path) -> None:
        """Save the fine-tuned model to disk."""
        if self._model is None:
            raise RuntimeError("No model to save — call predict() first to load")
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self._model.save_pretrained(path)
        if self._processor is not None:
            self._processor.save_pretrained(path)
        log.info("LayoutModel saved to %s", path)

    @classmethod
    def load(cls, path: str | Path, device: str | None = None) -> LayoutModel:
        """Load a fine-tuned model from disk."""
        instance = cls(model_name_or_path=str(path), device=device)
        instance._ensure_model()
        return instance


# ── Module-level convenience ───────────────────────────────────────────

_model: Optional[LayoutModel] = None


def get_layout_model(
    model_name_or_path: str | Path = _DEFAULT_MODEL_NAME,
    device: Optional[str] = None,
) -> LayoutModel:
    """Return (or create) the module-level singleton layout model."""
    global _model
    if _model is None or _model.model_name != str(model_name_or_path):
        _model = LayoutModel(model_name_or_path=model_name_or_path, device=device)
    return _model


def predict_layout(
    page_image,
    tokens: Sequence[Any],
    page_width: float,
    page_height: float,
    model_name_or_path: str | Path = _DEFAULT_MODEL_NAME,
    device: Optional[str] = None,
) -> list[LayoutPrediction]:
    """Convenience wrapper: page data → layout predictions.

    Returns an empty list if transformers/torch are not installed.
    """
    model = get_layout_model(model_name_or_path=model_name_or_path, device=device)
    return model.predict(page_image, tokens, page_width, page_height)
