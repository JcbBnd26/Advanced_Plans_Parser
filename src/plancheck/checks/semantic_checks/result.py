"""CheckResult data class for semantic check findings."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple


@dataclass
class CheckResult:
    """A single finding from a semantic check."""

    check_id: str  # e.g. "ABBREV_DUP"
    severity: str  # "error" | "warning" | "info"
    message: str  # Human-readable description
    page: int = 0
    bbox: Optional[Tuple[float, float, float, float]] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize check result to a JSON-compatible dict."""
        d: Dict[str, Any] = {
            "check_id": self.check_id,
            "severity": self.severity,
            "message": self.message,
            "page": self.page,
        }
        if self.bbox:
            d["bbox"] = list(self.bbox)
        if self.details:
            d["details"] = self.details
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "CheckResult":
        """Deserialize from a dict produced by :meth:`to_dict`."""
        return cls(
            check_id=d.get("check_id", ""),
            severity=d.get("severity", "info"),
            message=d.get("message", ""),
            page=d.get("page", 0),
            bbox=tuple(d["bbox"]) if d.get("bbox") else None,
            details=d.get("details", {}),
        )
