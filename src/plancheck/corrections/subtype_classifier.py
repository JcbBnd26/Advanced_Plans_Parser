"""Title subtype classifier — Stage 2 of the hierarchical classification system.

This module mirrors :mod:`~plancheck.corrections.classifier` but is trained
on the ``title`` family only, producing one of seven fine-grained subtypes:

    page_title, plan_title, detail_title, section_title,
    graph_title, map_title, box_title

Three relational features beyond the base 42 are added that are only
meaningful for title disambiguation:

* ``near_north_arrow``   — element is near a north-arrow graphic
* ``near_detail_bubble`` — element is near a circled detail bubble
* ``near_section_arrow`` — element is near a section-cut arrow marker

The classifier is architecturally identical to :class:`ElementClassifier`:
same GBM, same training loop, same calibration — just a different label
set and a filtered training source.

Bootstrap / training readiness
-------------------------------
Stage 2 should **not** be trained until at least 20 labeled examples exist
for every subtype (140+ total).  :meth:`TitleSubtypeClassifier.model_exists`
returns ``False`` when the model file is absent, allowing the hierarchical
router to fall back gracefully.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Tuple

if TYPE_CHECKING:
    import numpy as np

from .classifier import _NUMERIC_KEYS, ZONE_VALUES, ElementClassifier, encode_features

log = logging.getLogger(__name__)

# ── Stage-2 label set ─────────────────────────────────────────────────

#: Canonical subtype labels produced by this classifier.
TITLE_SUBTYPES: list[str] = [
    "page_title",
    "plan_title",
    "detail_title",
    "section_title",
    "graph_title",
    "map_title",
    "box_title",
]

# ── Additional relational feature keys for Stage 2 ───────────────────

#: Feature keys specific to the title subtype classifier.
#: These are appended after the base :data:`~plancheck.corrections.classifier._NUMERIC_KEYS`.
_SUBTYPE_EXTRA_KEYS: list[str] = [
    "near_north_arrow",
    "near_detail_bubble",
    "near_section_arrow",
]

_DEFAULT_SUBTYPE_MODEL_PATH = Path("data/title_subtype_classifier.pkl")


def encode_subtype_features(
    feature_dict: dict,
    image_features: np.ndarray | None = None,
    text_embedding: np.ndarray | None = None,
) -> np.ndarray:
    """Encode a feature dict for the Stage-2 classifier.

    Identical to :func:`~plancheck.corrections.classifier.encode_features`
    but appends the three extra title-relational features
    (:data:`_SUBTYPE_EXTRA_KEYS`) after the base numeric+zone block.

    Parameters
    ----------
    feature_dict : dict
        Output of :func:`~plancheck.corrections.features.featurize` **plus**
        the three extra keys defined in :data:`_SUBTYPE_EXTRA_KEYS`
        (defaults to 0 when absent).
    image_features : numpy.ndarray, optional
        CNN image embedding (appended last when provided).
    text_embedding : numpy.ndarray, optional
        Text embedding vector (appended after image features when provided).

    Returns
    -------
    numpy.ndarray
        1-D float64 array.
    """
    import numpy as np

    # Base features (numeric + zone one-hot)
    base = encode_features(feature_dict, image_features=None, text_embedding=None)

    # Extra title-relational features (0/1 booleans)
    extra = np.array(
        [float(feature_dict.get(k, 0)) for k in _SUBTYPE_EXTRA_KEYS],
        dtype=np.float64,
    )

    combined = np.concatenate([base, extra])

    # Append optional dense features last
    parts = [combined]
    if image_features is not None and len(image_features) > 0:
        parts.append(image_features.astype(np.float64))
    if text_embedding is not None and len(text_embedding) > 0:
        parts.append(text_embedding.astype(np.float64))

    return np.concatenate(parts) if len(parts) > 1 else combined


def featurize_title_subtype(
    base_features: dict,
    *,
    near_north_arrow: bool = False,
    near_detail_bubble: bool = False,
    near_section_arrow: bool = False,
) -> dict:
    """Extend a base feature dict with Stage-2 title-relational features.

    Parameters
    ----------
    base_features : dict
        Output of :func:`~plancheck.corrections.features.featurize`.
    near_north_arrow : bool
        True when the element is spatially near a north-arrow graphic.
    near_detail_bubble : bool
        True when the element is near a circled detail-bubble annotation.
    near_section_arrow : bool
        True when the element is near a section-cut arrow marker.

    Returns
    -------
    dict
        A shallow copy of *base_features* with the three extra keys added.
    """
    extended = dict(base_features)
    extended["near_north_arrow"] = int(near_north_arrow)
    extended["near_detail_bubble"] = int(near_detail_bubble)
    extended["near_section_arrow"] = int(near_section_arrow)
    return extended


class TitleSubtypeClassifier:
    """Stage-2 GBM classifier that refines a ``title`` family prediction.

    Architecturally identical to :class:`~plancheck.corrections.classifier.ElementClassifier`
    but:

    * Trained only on data where Stage-1 family == ``"title"``.
    * Uses :data:`TITLE_SUBTYPES` as its label set.
    * Expects three extra relational features beyond the base 42:
      ``near_north_arrow``, ``near_detail_bubble``, ``near_section_arrow``.

    Parameters
    ----------
    model_path : Path
        Location of the persisted ``.pkl`` model file.  Defaults to
        ``data/title_subtype_classifier.pkl``.
    """

    def __init__(self, model_path: Path = _DEFAULT_SUBTYPE_MODEL_PATH) -> None:
        self.model_path = Path(model_path)
        self._model = None
        self._raw_model = None
        self._n_features_in: Optional[int] = None

    # ── Estimator builders (mirrors ElementClassifier) ────────────

    @staticmethod
    def _build_gbm():
        from sklearn.ensemble import GradientBoostingClassifier

        return GradientBoostingClassifier(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.1,
            min_samples_leaf=5,
            subsample=0.8,
            random_state=42,
        )

    @staticmethod
    def _build_hist_gbm():
        from sklearn.ensemble import HistGradientBoostingClassifier

        return HistGradientBoostingClassifier(
            max_iter=200,
            max_depth=5,
            learning_rate=0.1,
            min_samples_leaf=10,
            random_state=42,
        )

    def _build_ensemble(self):
        from sklearn.ensemble import VotingClassifier

        estimators: list[tuple] = [
            ("gbm", self._build_gbm()),
            ("hist_gbm", self._build_hist_gbm()),
        ]
        return (
            VotingClassifier(estimators=estimators, voting="soft"),
            ["gbm", "hist_gbm"],
        )

    # ── Training ─────────────────────────────────────────────────

    def train(
        self,
        jsonl_path: Path,
        *,
        calibrate: bool = True,
        ensemble: bool = False,
    ) -> dict:
        """Train from a JSONL file filtered to title-family examples.

        The JSONL file is the output of
        :meth:`~plancheck.corrections.store.CorrectionStore.export_training_jsonl`
        **already filtered** to ``family == "title"`` by the caller, or it
        may contain all families — rows where the label is not in
        :data:`TITLE_SUBTYPES` are rejected.

        Parameters
        ----------
        jsonl_path : Path
            Path to JSONL training data.
        calibrate : bool
            Apply isotonic calibration after training (default True).
        ensemble : bool
            Use a soft-voting ensemble instead of single GBM.

        Returns
        -------
        dict
            Training metrics dict (same schema as
            :meth:`~plancheck.corrections.classifier.ElementClassifier.train`).

        Raises
        ------
        ValueError
            If any example has a label not in :data:`TITLE_SUBTYPES`.
        """
        import json as _json

        # ── Validate labels before training ──────────────────────
        allowed = set(TITLE_SUBTYPES)
        bad_labels: set[str] = set()
        with open(jsonl_path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                lbl = _json.loads(line).get("label", "")
                if lbl not in allowed:
                    bad_labels.add(lbl)

        if bad_labels:
            raise ValueError(
                f"Stage-2 training data contains labels not in TITLE_SUBTYPES: "
                f"{sorted(bad_labels)}.  Filter the JSONL to title subtypes "
                f"before calling train()."
            )

        from .training_loop import train_classifier

        metrics, raw_model, fitted_model, n_features = train_classifier(
            jsonl_path,
            self.model_path,
            self._build_gbm,
            self._build_ensemble,
            calibrate=calibrate,
            ensemble=ensemble,
            encode_fn=encode_subtype_features,
        )
        self._raw_model = raw_model
        self._model = fitted_model
        self._n_features_in = n_features
        return metrics

    # ── Prediction ────────────────────────────────────────────────

    def _load_model(self) -> None:
        """Lazy-load the model from disk."""
        if self._model is not None:
            return
        import joblib

        data = joblib.load(self.model_path)
        self._raw_model = data.get("model")
        self._model = data.get("calibrated_model", data["model"])
        self._n_features_in = data.get("n_features_in")

    def model_exists(self) -> bool:
        """Return True if a trained model file exists on disk."""
        return self.model_path.is_file()

    def predict(
        self,
        feature_dict: dict,
        image_features: np.ndarray | None = None,
        text_embedding: np.ndarray | None = None,
    ) -> Tuple[str, float]:
        """Predict title subtype and confidence.

        Parameters
        ----------
        feature_dict : dict
            Extended feature dict from :func:`featurize_title_subtype`.
        image_features : numpy.ndarray, optional
        text_embedding : numpy.ndarray, optional

        Returns
        -------
        tuple[str, float]
            ``(subtype_label, confidence)``
        """
        import numpy as np

        self._load_model()
        x = encode_subtype_features(
            feature_dict, image_features, text_embedding
        ).reshape(1, -1)
        if self._n_features_in is not None and x.shape[1] > self._n_features_in:
            x = x[:, : self._n_features_in]
        proba = self._model.predict_proba(x)[0]
        idx = int(np.argmax(proba))
        return str(self._model.classes_[idx]), float(proba[idx])

    def predict_top_k(
        self,
        feature_dict: dict,
        k: int = 2,
        image_features: np.ndarray | None = None,
        text_embedding: np.ndarray | None = None,
    ) -> List[Tuple[str, float]]:
        """Return the top-*k* subtype predictions ordered by confidence.

        Parameters
        ----------
        feature_dict : dict
            Extended feature dict from :func:`featurize_title_subtype`.
        k : int
            Number of candidates to return (default 2).

        Returns
        -------
        list[tuple[str, float]]
            ``[(subtype_label, confidence), …]`` ordered highest-first.
        """
        import numpy as np

        self._load_model()
        x = encode_subtype_features(
            feature_dict, image_features, text_embedding
        ).reshape(1, -1)
        if self._n_features_in is not None and x.shape[1] > self._n_features_in:
            x = x[:, : self._n_features_in]
        proba = self._model.predict_proba(x)[0]
        classes = self._model.classes_
        top_indices = np.argsort(proba)[::-1][:k]
        return [(str(classes[i]), float(proba[i])) for i in top_indices]

    def get_feature_importance(self) -> dict[str, float]:
        """Return feature-name → importance mapping from the trained model."""
        self._load_model()
        model = self._raw_model if self._raw_model is not None else self._model
        if model is None:
            return {}

        feature_names = (
            list(_NUMERIC_KEYS)
            + [f"zone_{z}" for z in ZONE_VALUES]
            + list(_SUBTYPE_EXTRA_KEYS)
        )

        import numpy as np

        try:
            importances = np.asarray(model.feature_importances_)
        except Exception:  # noqa: BLE001 — not all models have feature_importances_
            log.warning(
                "feature_importances_ unavailable for model type", exc_info=True
            )
            return {}

        if importances.shape[0] != len(feature_names):
            return {}

        return {
            name: round(float(imp), 6)
            for name, imp in sorted(
                zip(feature_names, importances), key=lambda x: -x[1]
            )
        }

    def calibration_curve(self, jsonl_path: Path) -> dict:
        """Compute calibration curves for each Stage-2 subtype class."""
        self._load_model()
        if self._model is None:
            return {"curves": {}, "ece": 0.0}

        examples: list[dict] = []
        with open(jsonl_path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    examples.append(json.loads(line))

        val_ex = [e for e in examples if e.get("split") == "val"]
        if not val_ex:
            val_ex = examples
        if not val_ex:
            return {"curves": {}, "ece": 0.0}

        import numpy as np

        X = np.array([encode_subtype_features(e["features"]) for e in val_ex])
        y_true = [e["label"] for e in val_ex]
        probas = self._model.predict_proba(X)
        classes = list(self._model.classes_)

        from sklearn.calibration import calibration_curve as _sk_cal_curve

        curves: dict[str, dict] = {}
        total_ece = 0.0
        total_weight = 0

        for i, cls in enumerate(classes):
            y_binary = np.array([1 if yt == cls else 0 for yt in y_true])
            p_cls = probas[:, i]
            n_pos = int(y_binary.sum())
            if n_pos == 0 or n_pos == len(y_binary):
                continue
            try:
                frac_pos, mean_pred = _sk_cal_curve(
                    y_binary,
                    p_cls,
                    n_bins=min(10, max(3, n_pos)),
                    strategy="uniform",
                )
                curves[cls] = {
                    "mean_predicted": [round(float(v), 4) for v in mean_pred],
                    "fraction_positive": [round(float(v), 4) for v in frac_pos],
                }
                bin_counts = np.histogram(
                    p_cls,
                    bins=len(frac_pos),
                    range=(0, 1),
                )[0]
                ece = float(
                    np.sum(np.abs(frac_pos - mean_pred) * bin_counts)
                    / max(len(p_cls), 1)
                )
                total_ece += ece * n_pos
                total_weight += n_pos
            except Exception:  # noqa: BLE001
                log.warning("Calibration curve calc failed for a label", exc_info=True)
                continue

        weighted_ece = total_ece / total_weight if total_weight > 0 else 0.0
        return {"curves": curves, "ece": round(weighted_ece, 4)}
