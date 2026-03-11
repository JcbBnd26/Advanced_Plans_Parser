"""LLM-assisted semantic checks for construction-plan pages.

Adds an **optional** LLM pass after the rule-based
:mod:`~plancheck.checks.semantic_checks` stage.  The LLM analyses
notes-column text for vagueness, contradictions, missing references,
and code-compliance issues that deterministic rules cannot catch.

Three provider back-ends are supported:

* **ollama** — local Ollama server (default, no API key needed)
* **openai** — OpenAI-compatible API (GPT-4o-mini, etc.)
* **anthropic** — Anthropic API (Claude 3 Haiku, etc.)

All findings are returned as :class:`~plancheck.checks.semantic_checks.CheckResult`
with ``severity="llm_suggestion"`` so the UI can distinguish them from
deterministic findings.  Users can accept or dismiss each one.

The module is **optional** — if the required client library is not
installed, :func:`run_llm_checks` returns an empty list.

Public API
----------
run_llm_checks      – Analyse notes columns via LLM and return findings
is_llm_available    – Check whether a provider's library is installed
LLMClient           – Thin wrapper around provider APIs
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Sequence

log = logging.getLogger(__name__)

# ── Availability (delegated to plancheck.llm) ──────────────────────────

from plancheck.llm.client import is_llm_available  # noqa: E402, F401 — re-export

# ── Prompt templates ───────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are a construction-plan QA reviewer.  You will be given the text of \
construction notes extracted from an architectural/engineering plan sheet.

Analyse the notes for:
1. **Vagueness** — notes that are too vague to be actionable \
   (e.g. "per local codes" without specifying which code).
2. **Contradictions** — notes that contradict each other.
3. **Missing references** — notes that reference other details, sheets, \
   or specifications that may not exist.
4. **Code compliance** — notes that may not comply with common building \
   codes (IBC, ADA, NFPA, etc.).
5. **Incomplete information** — notes that appear truncated or missing \
   critical details.

Return your findings as a JSON array.  Each finding must have:
- "issue_type": one of "vague", "contradiction", "missing_ref", \
  "code_compliance", "incomplete"
- "severity": "info" or "warning" (never "error" — you are making suggestions)
- "message": a concise description
- "note_text": the verbatim note text that triggered the finding

If there are no issues, return an empty array: []

IMPORTANT: Return ONLY the JSON array, no other text.\
"""

_USER_PROMPT_TEMPLATE = """\
Here are the construction notes from page {page}:

---
{notes_text}
---

Analyse these notes and return findings as a JSON array.\
"""


# ── LLM Client (re-exported from plancheck.llm) ──────────────────────
# The canonical implementation now lives in plancheck.llm.client.
# We re-export here for backwards compatibility so existing code that does
#   from plancheck.checks.llm_checks import LLMClient
# continues to work.

from plancheck.llm.client import LLMClient  # noqa: F401 — re-export

# ── Response parsing ───────────────────────────────────────────────────


def _parse_llm_response(response_text: str, page: int = 0) -> list[dict]:
    """Parse the LLM's JSON response into a list of finding dicts.

    Tolerates markdown fences and minor formatting issues.
    """
    text = response_text.strip()

    # Strip markdown code fences if present
    if text.startswith("```"):
        # Remove opening fence (```json or ```)
        text = re.sub(r"^```(?:json)?\s*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)
        text = text.strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Try to extract a JSON array from the text
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
            except json.JSONDecodeError:
                log.warning("Could not parse LLM response as JSON")
                return []
        else:
            log.warning("No JSON array found in LLM response")
            return []

    if not isinstance(data, list):
        data = [data]

    # Validate each finding
    valid_types = {
        "vague",
        "contradiction",
        "missing_ref",
        "code_compliance",
        "incomplete",
    }
    findings = []
    for item in data:
        if not isinstance(item, dict):
            continue
        issue_type = item.get("issue_type", "")
        if issue_type not in valid_types:
            issue_type = "vague"  # fallback
        severity = item.get("severity", "info")
        if severity not in ("info", "warning"):
            severity = "info"  # LLM findings are suggestions, never errors
        message = item.get("message", "")
        if not message:
            continue
        findings.append(
            {
                "issue_type": issue_type,
                "severity": severity,
                "message": message,
                "note_text": item.get("note_text", ""),
            }
        )

    return findings


# ── Main entry point ───────────────────────────────────────────────────


def run_llm_checks(
    *,
    notes_columns: Sequence[Any] | None = None,
    page: int = 0,
    provider: str = "ollama",
    model: str = "llama3.1:8b",
    api_key: str = "",
    api_base: str = "http://localhost:11434",
    temperature: float = 0.1,
    policy: str = "local_only",
) -> list:
    """Run LLM-assisted semantic checks on notes columns.

    Parameters
    ----------
    notes_columns : sequence, optional
        Notes columns from the pipeline (each has a ``.text`` or
        ``.blocks`` attribute).
    page : int
        Page number for findings.
    provider : str
        LLM provider (``"ollama"``, ``"openai"``, ``"anthropic"``).
    model : str
        Model name.
    api_key : str
        API key (for openai/anthropic).
    api_base : str
        API base URL.
    temperature : float
        LLM temperature.

    Returns
    -------
    list[CheckResult]
        Findings with ``severity="llm_suggestion"`` or ``"info"``/``"warning"``.
    """
    from .semantic_checks import CheckResult

    if not notes_columns:
        return []

    if not is_llm_available(provider):
        log.debug("LLM provider %r not available — skipping LLM checks", provider)
        return []

    # Extract text from notes columns
    all_notes_text = []
    for col in notes_columns:
        # Try various ways to get text from the column
        col_text = ""
        if hasattr(col, "full_text") and callable(col.full_text):
            col_text = col.full_text()
        elif hasattr(col, "blocks"):
            lines = []
            for block in col.blocks:
                if hasattr(block, "get_all_boxes"):
                    boxes = block.get_all_boxes()
                    lines.append(" ".join(b.text for b in boxes))
                elif hasattr(block, "text"):
                    lines.append(block.text)
            col_text = "\n".join(lines)
        elif hasattr(col, "text"):
            col_text = col.text

        if col_text and col_text.strip():
            all_notes_text.append(col_text.strip())

    if not all_notes_text:
        return []

    notes_text = "\n\n".join(all_notes_text)

    # Truncate very long notes to avoid token limits
    max_chars = 8000
    if len(notes_text) > max_chars:
        notes_text = notes_text[:max_chars] + "\n\n[... truncated ...]"

    # Call the LLM
    client = LLMClient(
        provider=provider,
        model=model,
        api_key=api_key,
        api_base=api_base,
        temperature=temperature,
        policy=policy,
    )

    user_prompt = _USER_PROMPT_TEMPLATE.format(page=page, notes_text=notes_text)

    try:
        response = client.chat(_SYSTEM_PROMPT, user_prompt)
    except Exception as exc:
        log.warning("LLM call failed: %s", exc)
        return [
            CheckResult(
                check_id="LLM_ERROR",
                severity="info",
                message=f"LLM analysis failed: {exc}",
                page=page,
            )
        ]

    # Parse response into findings
    raw_findings = _parse_llm_response(response, page=page)

    # Convert to CheckResult
    results: list[CheckResult] = []
    for i, f in enumerate(raw_findings):
        check_id = f"LLM_{f['issue_type'].upper()}_{i + 1}"
        results.append(
            CheckResult(
                check_id=check_id,
                severity=f["severity"],
                message=f["message"],
                page=page,
                details={
                    "issue_type": f["issue_type"],
                    "note_text": f["note_text"],
                    "llm_provider": provider,
                    "llm_model": model,
                },
            )
        )

    log.info("LLM checks: %d findings from %s/%s", len(results), provider, model)
    return results


# ── Title-subtype tiebreaker ──────────────────────────────────────────

_SUBTYPE_SYSTEM_PROMPT = """\
You are analyzing elements extracted from an architectural or engineering plan sheet.
Your task is to classify a text element into exactly one subtype category based on its
text content, position on the sheet, and nearby context.

You must return ONLY a JSON object with two keys:
  "subtype": one of: page_title, plan_title, detail_title, section_title, graph_title, map_title, box_title
  "confidence": float 0.0-1.0

Do not explain your reasoning. Do not return any other text.\
"""

_SUBTYPE_USER_TEMPLATE = """\
Classify this title element:

Text: {text}
Zone: {zone}
Position: x={x_frac:.2f}, y={y_frac:.2f} (0=left/top, 1=right/bottom)
Width: {width_frac:.2f} (fraction of page width)

Top model candidates:
{candidates}

Return ONLY a JSON object with "subtype" and "confidence" keys.\
"""


def llm_classify_title_subtype(
    text: str,
    features: dict,
    candidates: list,
    *,
    provider: str = "ollama",
    model: str = "llama3.1:8b",
    api_key: str = "",
    api_base: str = "http://localhost:11434",
    policy: str = "local_only",
) -> tuple[str, float]:
    """Use the LLM to break ties in Stage-2 title subtype classification.

    This is invoked only when Stage-2 GBM confidence is below the threshold.
    It uses the element's text content and positional context — signals that
    the GBM cannot read — to choose between the top candidates.

    Parameters
    ----------
    text : str
        Raw text content of the title element.
    features : dict
        Feature dict from :func:`~plancheck.corrections.features.featurize`
        (used for zone and positional context).
    candidates : list[tuple[str, float]]
        Top Stage-2 candidates in ``[(label, confidence), …]`` order.
    provider : str
        LLM provider (``"ollama"``, ``"openai"``, ``"anthropic"``).
    model : str
        Model identifier.
    api_key : str
        API key (not needed for Ollama).
    api_base : str
        Base API URL.
    policy : str
        Data-privacy policy.

    Returns
    -------
    tuple[str, float]
        ``(subtype_label, confidence)`` from the LLM, or ``("", 0.0)``
        on failure or when the provider is unavailable.
    """
    if not is_llm_available(provider):
        log.debug(
            "LLM provider %r not available — cannot classify subtype", provider
        )
        return "", 0.0

    if not text or not text.strip():
        log.debug("Empty text — skipping LLM subtype classification")
        return "", 0.0

    candidates_text = "\n".join(
        f"  {i + 1}. {label} (confidence={conf:.3f})"
        for i, (label, conf) in enumerate(candidates)
    )

    user_prompt = _SUBTYPE_USER_TEMPLATE.format(
        text=text[:300],
        zone=features.get("zone", "unknown"),
        x_frac=features.get("x_frac", 0.0),
        y_frac=features.get("y_frac", 0.0),
        width_frac=features.get("width_frac", 0.0),
        candidates=candidates_text,
    )

    client = LLMClient(
        provider=provider,
        model=model,
        api_key=api_key,
        api_base=api_base,
        temperature=0.0,  # deterministic for classification
        policy=policy,
    )

    try:
        parsed, _meta = client.chat_structured(_SUBTYPE_SYSTEM_PROMPT, user_prompt)
    except Exception as exc:  # noqa: BLE001 — LLM failures must not break pipeline
        log.warning("LLM subtype tiebreaker failed: %s", exc)
        return "", 0.0

    subtype = parsed.get("subtype", "")
    confidence = float(parsed.get("confidence", 0.0))

    # Validate that the LLM returned a known subtype
    from plancheck.corrections.subtype_classifier import TITLE_SUBTYPES

    if subtype not in TITLE_SUBTYPES:
        log.warning(
            "LLM returned unknown subtype %r — ignoring", subtype
        )
        return "", 0.0

    log.debug(
        "LLM tiebreaker: %r → %r (conf=%.3f) via %s/%s",
        text[:40],
        subtype,
        confidence,
        provider,
        model,
    )
    return subtype, confidence
