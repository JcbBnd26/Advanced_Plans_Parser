"""Helper functions for semantic checks.

Contains severity adjustment logic and date parsing utilities.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Optional

# ── Severity attenuation based on OCR confidence ────────────────────

_SEVERITY_DOWNGRADE = {"error": "warning", "warning": "info"}


def _adjusted_severity(base: str, mean_ocr_conf: float, threshold: float = 0.6) -> str:
    """Downgrade severity when mean OCR confidence is below *threshold*.

    - ``error`` → ``warning``
    - ``warning`` → ``info``
    - ``info`` stays ``info``

    When *mean_ocr_conf* ≥ *threshold* (or is exactly 1.0, i.e. PDF-only),
    the base severity is returned unchanged.
    """
    if mean_ocr_conf >= threshold:
        return base
    return _SEVERITY_DOWNGRADE.get(base, base)


# ── Date parsing for revision checks ────────────────────────────────

_DATE_PATTERNS = [
    (re.compile(r"(\d{1,2})/(\d{1,2})/(\d{2,4})"), "MDY"),  # MM/DD/YYYY
    (re.compile(r"(\d{1,2})-(\d{1,2})-(\d{2,4})"), "MDY"),  # MM-DD-YYYY
    (re.compile(r"(\d{4})-(\d{1,2})-(\d{1,2})"), "YMD"),  # YYYY-MM-DD
    (re.compile(r"(\d{1,2})\s+\w+\s+(\d{4})"), "DMonY"),  # 12 Jan 2024
    (re.compile(r"\w+\s+(\d{1,2}),?\s+(\d{4})"), "MonDY"),  # Jan 12, 2024
]

_MONTH_NAMES = {
    "JAN": 1,
    "FEB": 2,
    "MAR": 3,
    "APR": 4,
    "MAY": 5,
    "JUN": 6,
    "JUL": 7,
    "AUG": 8,
    "SEP": 9,
    "OCT": 10,
    "NOV": 11,
    "DEC": 12,
    "JANUARY": 1,
    "FEBRUARY": 2,
    "MARCH": 3,
    "APRIL": 4,
    "JUNE": 6,
    "JULY": 7,
    "AUGUST": 8,
    "SEPTEMBER": 9,
    "OCTOBER": 10,
    "NOVEMBER": 11,
    "DECEMBER": 12,
}


def _parse_date(text: str) -> Optional[datetime]:
    """Try to parse a date string using common US engineering formats."""
    text = text.strip()
    if not text:
        return None

    # MM/DD/YYYY or MM-DD-YYYY
    for pattern, fmt in _DATE_PATTERNS[:2]:
        m = pattern.search(text)
        if m:
            month, day, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
            if year < 100:
                year += 2000
            try:
                return datetime(year, month, day)
            except ValueError:
                continue

    # YYYY-MM-DD
    m = _DATE_PATTERNS[2][0].search(text)
    if m:
        try:
            return datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            pass

    # Try free-form with dateutil-style month names
    parts = re.findall(r"[A-Za-z]+|\d+", text.upper())
    nums = [int(p) for p in parts if p.isdigit()]
    months = [_MONTH_NAMES[p] for p in parts if p in _MONTH_NAMES]

    if months and len(nums) >= 2:
        month = months[0]
        # Determine which num is day vs year
        if nums[-1] > 31:  # last number is year
            year = nums[-1] if nums[-1] > 100 else nums[-1] + 2000
            day = nums[0]
        else:
            day = nums[0]
            year = nums[1] if nums[1] > 100 else nums[1] + 2000
        try:
            return datetime(year, month, day)
        except ValueError:
            pass

    return None
