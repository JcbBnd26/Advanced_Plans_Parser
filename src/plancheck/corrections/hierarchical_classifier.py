"""Hierarchical two-stage element classifier.

Implements the routing logic that chains:

    Stage 1 (ElementClassifier)
        → Stage 2 (TitleSubtypeClassifier, when Stage 1 returns ``"title"``)
            → LLM tiebreaker (when Stage 2 confidence < threshold)

Entry point
-----------
:func:`classify_element` — classify a single element given a pre-computed
feature dict and optional raw text, returning a :class:`ClassificationResult`.

Routing contract
----------------
The routing table, as specified in the architecture plan, is:

| Condition | Action |
|-----------|--------|
| Stage 1 confidence ≥ 0.7, family ≠ "title" | Return Stage 1 label as final. |
| Stage 1 confidence < 0.7, any family | Return Stage 1 label with ``low_confidence=True``. |
| Stage 1 family == "title", confidence ≥ 0.7 | Run Stage 2. |
| Stage 2 confidence ≥ 0.6 | Return Stage 2 subtype as final. |
| Stage 2 confidence < 0.6 | Invoke LLM tiebreaker; fall back to Stage 2 argmax. |

LLM is only invoked when ``enable_llm`` is True *and* the LLM tiebreaker
function is importable.  This keeps the router usable without any LLM
configured.

Stage 2 cold-start
------------------
When no trained Stage-2 model file exists,
:func:`classify_element` falls back to returning the Stage-1 ``"title"``
label directly (with a ``stage2_skipped=True`` flag in the result).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

# ── Public thresholds (tuneable via config in future) ─────────────────

#: Stage-1 minimum confidence to commit to a non-title family prediction.
STAGE1_CONFIDENCE_THRESHOLD: float = 0.7

#: Stage-2 minimum confidence to commit without invoking the LLM.
STAGE2_CONFIDENCE_THRESHOLD: float = 0.6

#: The Stage-1 family label that triggers Stage-2 refinement.
TITLE_FAMILY_LABEL: str = "title"


# ── Result dataclass ──────────────────────────────────────────────────


@dataclass
class ClassificationResult:
    """Fully-resolved classification result from the hierarchical pipeline.

    Attributes
    ----------
    label : str
        Final predicted label (Stage-1 family or Stage-2 subtype).
    confidence : float
        Confidence of the final label, in [0, 1].
    family : str
        Stage-1 family label (always populated).
    family_confidence : float
        Stage-1 confidence.
    subtype : Optional[str]
        Stage-2 subtype label when Stage 2 ran, else ``None``.
    subtype_confidence : Optional[float]
        Stage-2 confidence when Stage 2 ran, else ``None``.
    low_confidence : bool
        True when the final confidence is below the relevant threshold
        and the prediction should be flagged for human review.
    stage2_skipped : bool
        True when Stage 2 was not run (model file absent, family ≠ title).
    llm_used : bool
        True when the LLM tiebreaker was invoked.
    stage2_candidates : list[tuple[str, float]]
        Top-2 Stage-2 candidates (populated when Stage 2 ran).
    """

    label: str
    confidence: float
    family: str
    family_confidence: float
    subtype: Optional[str] = None
    subtype_confidence: Optional[float] = None
    low_confidence: bool = False
    stage2_skipped: bool = False
    llm_used: bool = False
    stage2_candidates: List[Tuple[str, float]] = field(default_factory=list)


# ── Classifier singletons (lazy-loaded per model path) ───────────────

_stage1_instances: Dict[Path, Any] = {}
_stage2_instances: Dict[Path, Any] = {}


def _get_stage1(model_path: Path) -> Any:
    """Return a cached :class:`~plancheck.corrections.classifier.ElementClassifier`."""
    if model_path not in _stage1_instances:
        from .classifier import ElementClassifier

        _stage1_instances[model_path] = ElementClassifier(model_path=model_path)
    return _stage1_instances[model_path]


def _get_stage2(model_path: Path) -> Any:
    """Return a cached :class:`~plancheck.corrections.subtype_classifier.TitleSubtypeClassifier`."""
    if model_path not in _stage2_instances:
        from .subtype_classifier import TitleSubtypeClassifier

        _stage2_instances[model_path] = TitleSubtypeClassifier(model_path=model_path)
    return _stage2_instances[model_path]


# ── Main entry point ──────────────────────────────────────────────────


def classify_element(
    feature_dict: Dict[str, Any],
    text: str = "",
    *,
    stage1_model_path: Optional[Path] = None,
    stage2_model_path: Optional[Path] = None,
    stage1_confidence_threshold: float = STAGE1_CONFIDENCE_THRESHOLD,
    stage2_confidence_threshold: float = STAGE2_CONFIDENCE_THRESHOLD,
    enable_llm: bool = False,
    llm_provider: str = "ollama",
    llm_model: str = "llama3.1:8b",
    llm_api_key: str = "",
    llm_api_base: str = "http://localhost:11434",
    llm_policy: str = "local_only",
    image_features: Optional[np.ndarray] = None,
    text_embedding: Optional[np.ndarray] = None,
) -> ClassificationResult:
    """Classify a single plan element using the two-stage hierarchy.

    Parameters
    ----------
    feature_dict : dict
        Feature dict from :func:`~plancheck.corrections.features.featurize`.
        For Stage-2 features to be populated, the dict should also contain
        the keys ``near_north_arrow``, ``near_detail_bubble``,
        ``near_section_arrow`` (produced by
        :func:`~plancheck.corrections.subtype_classifier.featurize_title_subtype`).
    text : str
        Raw text content of the element — passed to the LLM tiebreaker.
    stage1_model_path : Path, optional
        Path to the Stage-1 model file.  Defaults to
        ``data/element_classifier.pkl``.
    stage2_model_path : Path, optional
        Path to the Stage-2 model file.  Defaults to
        ``data/title_subtype_classifier.pkl``.
    stage1_confidence_threshold : float
        Minimum Stage-1 confidence to commit to a non-title label.
    stage2_confidence_threshold : float
        Minimum Stage-2 confidence to commit without the LLM.
    enable_llm : bool
        When *True*, invoke the LLM tiebreaker for low-confidence Stage-2
        predictions.  Requires a configured LLM provider.
    llm_provider : str
        LLM provider name (``"ollama"``, ``"openai"``, ``"anthropic"``).
    llm_model : str
        Model identifier.
    llm_api_key : str
        API key (not needed for Ollama).
    llm_api_base : str
        Base URL for the LLM API.
    llm_policy : str
        Data-privacy policy (``"local_only"`` or ``"cloud_allowed"``).
    image_features : numpy.ndarray, optional
        CNN image embedding for the element's visual crop.
    text_embedding : numpy.ndarray, optional
        Dense text embedding from sentence-transformer.

    Returns
    -------
    ClassificationResult
        Fully-resolved prediction with routing metadata.
    """
    # ── Stage-1: family classification ────────────────────────────
    from pathlib import Path as _Path

    s1_path = stage1_model_path or _Path("data/element_classifier.pkl")
    s1 = _get_stage1(s1_path)

    if not s1.model_exists():
        log.warning("Stage-1 model not found at %s — returning empty label", s1_path)
        return ClassificationResult(
            label="",
            confidence=0.0,
            family="",
            family_confidence=0.0,
            low_confidence=True,
            stage2_skipped=True,
        )

    family_label, family_conf = s1.predict(
        feature_dict,
        image_features=image_features,
        text_embedding=text_embedding,
    )

    # ── Low-confidence Stage-1: flag for review ───────────────────
    if family_conf < stage1_confidence_threshold:
        log.debug(
            "Stage-1 low confidence (%.3f) for family %r — flagging for review",
            family_conf,
            family_label,
        )
        return ClassificationResult(
            label=family_label,
            confidence=family_conf,
            family=family_label,
            family_confidence=family_conf,
            low_confidence=True,
            stage2_skipped=(family_label != TITLE_FAMILY_LABEL),
        )

    # ── Non-title families: Stage-1 result is final ───────────────
    if family_label != TITLE_FAMILY_LABEL:
        return ClassificationResult(
            label=family_label,
            confidence=family_conf,
            family=family_label,
            family_confidence=family_conf,
            stage2_skipped=True,
        )

    # ── Title family: run Stage-2 subtype classifier ──────────────
    s2_path = stage2_model_path or _Path("data/title_subtype_classifier.pkl")
    s2 = _get_stage2(s2_path)

    if not s2.model_exists():
        log.debug(
            "Stage-2 model not found at %s — returning title family label",
            s2_path,
        )
        return ClassificationResult(
            label=TITLE_FAMILY_LABEL,
            confidence=family_conf,
            family=TITLE_FAMILY_LABEL,
            family_confidence=family_conf,
            stage2_skipped=True,
        )

    # Get top-2 candidates from Stage 2 for the LLM tiebreaker
    try:
        candidates = s2.predict_top_k(feature_dict, k=2)
    except Exception:  # noqa: BLE001 — Stage 2 failure must not crash pipeline
        log.warning(
            "Stage-2 prediction failed — falling back to title family label",
            exc_info=True,
        )
        return ClassificationResult(
            label=TITLE_FAMILY_LABEL,
            confidence=family_conf,
            family=TITLE_FAMILY_LABEL,
            family_confidence=family_conf,
            stage2_skipped=True,
        )

    subtype_label, subtype_conf = candidates[0]

    # ── Stage-2 confident: return subtype directly ─────────────────
    if subtype_conf >= stage2_confidence_threshold:
        return ClassificationResult(
            label=subtype_label,
            confidence=subtype_conf,
            family=TITLE_FAMILY_LABEL,
            family_confidence=family_conf,
            subtype=subtype_label,
            subtype_confidence=subtype_conf,
            stage2_candidates=list(candidates),
        )

    # ── Stage-2 uncertain: LLM tiebreaker ─────────────────────────
    if enable_llm:
        llm_label, llm_conf = _run_llm_tiebreaker(
            text=text,
            feature_dict=feature_dict,
            candidates=list(candidates),
            provider=llm_provider,
            model=llm_model,
            api_key=llm_api_key,
            api_base=llm_api_base,
            policy=llm_policy,
        )
        if llm_label:
            log.info(
                "LLM tiebreaker: %r → %r (conf=%.3f)", text[:40], llm_label, llm_conf
            )
            return ClassificationResult(
                label=llm_label,
                confidence=llm_conf,
                family=TITLE_FAMILY_LABEL,
                family_confidence=family_conf,
                subtype=llm_label,
                subtype_confidence=llm_conf,
                llm_used=True,
                stage2_candidates=list(candidates),
            )

    # LLM unavailable or disabled — fall back to Stage-2 argmax
    return ClassificationResult(
        label=subtype_label,
        confidence=subtype_conf,
        family=TITLE_FAMILY_LABEL,
        family_confidence=family_conf,
        subtype=subtype_label,
        subtype_confidence=subtype_conf,
        low_confidence=True,
        stage2_candidates=list(candidates),
    )


# ── LLM tiebreaker helper ─────────────────────────────────────────────


def _run_llm_tiebreaker(
    text: str,
    feature_dict: Dict[str, Any],
    candidates: List[Tuple[str, float]],
    provider: str,
    model: str,
    api_key: str,
    api_base: str,
    policy: str,
) -> Tuple[str, float]:
    """Invoke the LLM tiebreaker for low-confidence Stage-2 predictions.

    Parameters
    ----------
    text : str
        Raw element text.
    feature_dict : dict
        Element features (used to populate zone / position context).
    candidates : list[tuple[str, float]]
        Top Stage-2 candidates ordered by confidence.
    provider, model, api_key, api_base, policy : str
        LLM configuration.

    Returns
    -------
    tuple[str, float]
        ``(subtype_label, confidence)`` from the LLM, or ``("", 0.0)``
        on failure.
    """
    try:
        from plancheck.checks.llm_checks import llm_classify_title_subtype
        from plancheck.llm.client import is_llm_available

        if not is_llm_available(provider):
            log.debug("LLM provider %r not available — skipping tiebreaker", provider)
            return "", 0.0

        return llm_classify_title_subtype(
            text=text,
            features=feature_dict,
            candidates=candidates,
            provider=provider,
            model=model,
            api_key=api_key,
            api_base=api_base,
            policy=policy,
        )
    except Exception:  # noqa: BLE001 — LLM failures must not break classification
        log.warning("LLM tiebreaker failed", exc_info=True)
        return "", 0.0
