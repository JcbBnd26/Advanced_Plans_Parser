"""Cross-page (document-level) semantic checks.

Extracted from :mod:`plancheck.pipeline` for maintainability.
"""

from __future__ import annotations

from typing import Any, Dict, List

from .page_result import PageResult


def _run_document_checks(pages: List[PageResult]) -> List[Any]:
    """Run document-level (cross-page) checks.

    These checks look at data across pages and flag inconsistencies that
    single-page checks cannot detect.
    """
    from .checks.semantic_checks import CheckResult

    findings: List[CheckResult] = []

    if not pages:
        return findings

    # 1. Duplicate sheet numbers across pages
    # Collect all sheet numbers → page lists first so we can detect
    # duplicates in a single pass.  Normalise to uppercase because
    # "A-101" and "a-101" are the same sheet in practice.
    sheet_map: Dict[str, List[int]] = {}
    for pr in pages:
        for tb in pr.title_blocks:
            sn = tb.sheet_number.strip().upper()
            if sn:
                sheet_map.setdefault(sn, []).append(pr.page)
    for sn, pg_list in sheet_map.items():
        if len(pg_list) > 1:
            findings.append(
                CheckResult(
                    check_id="DOC_DUP_SHEET",
                    severity="error",
                    message=(f"Sheet number '{sn}' appears on pages " f"{pg_list}"),
                    details={"sheet_number": sn, "pages": pg_list},
                )
            )

    # 2. Abbreviation consistency: same code should mean the same thing
    # Normalise to uppercase to catch case-variation false positives.
    # Structure: code → {meaning → [pages]}.  Multiple meanings for the
    # same code means the drawings are internally inconsistent.
    global_abbrevs: Dict[str, Dict[str, List[int]]] = {}
    for pr in pages:
        for region in pr.abbreviation_regions:
            for entry in region.entries:
                code = entry.code.strip().upper()
                meaning = entry.meaning.strip().upper()
                if code:
                    global_abbrevs.setdefault(code, {}).setdefault(meaning, []).append(
                        pr.page
                    )
    for code, meanings in global_abbrevs.items():
        if len(meanings) > 1:
            findings.append(
                CheckResult(
                    check_id="DOC_ABBREV_CONFLICT",
                    severity="error",
                    message=(
                        f"Abbreviation '{code}' has different meanings "
                        f"across pages: {list(meanings.keys())}"
                    ),
                    details={"code": code, "meanings": dict(meanings)},
                )
            )

    # 3. Pages missing title blocks
    pages_without_tb = [pr.page for pr in pages if not pr.title_blocks]
    if pages_without_tb and len(pages_without_tb) < len(pages):
        # Only flag if *some* pages have title blocks (mixed document)
        findings.append(
            CheckResult(
                check_id="DOC_TITLE_INCONSISTENT",
                severity="info",
                message=(
                    f"Pages {pages_without_tb} have no title block "
                    f"while other pages do"
                ),
                details={"pages_without": pages_without_tb},
            )
        )

    # 4. Legend consistency: same symbol description shouldn't change
    # Build a unique key from symbol type + bbox to identify the same
    # symbol across pages.  If the same key maps to different textual
    # descriptions then the legend disagrees with itself.
    global_legends: Dict[str, Dict[str, List[int]]] = {}
    for pr in pages:
        for region in pr.legend_regions:
            for entry in region.entries:
                desc = entry.description.strip().upper()
                sym_key = ""
                if entry.symbol and entry.symbol.element_type:
                    sb = entry.symbol_bbox or (
                        entry.symbol.x0,
                        entry.symbol.y0,
                        entry.symbol.x1,
                        entry.symbol.y1,
                    )
                    sym_key = f"{entry.symbol.element_type}_{sb}"
                if sym_key and desc:
                    global_legends.setdefault(sym_key, {}).setdefault(desc, []).append(
                        pr.page
                    )
    for sym_key, descs in global_legends.items():
        if len(descs) > 1:
            findings.append(
                CheckResult(
                    check_id="DOC_LEGEND_CONFLICT",
                    severity="warning",
                    message=(
                        f"Legend symbol '{sym_key}' has different "
                        f"descriptions across pages: {list(descs.keys())}"
                    ),
                    details={"symbol": sym_key, "descriptions": dict(descs)},
                )
            )

    # 5. Revision sequence gaps: look for missing revision numbers
    # Revisions should form a contiguous sequence 1..N.  Gaps suggest
    # missing revision entries or OCR misreads.
    all_rev_nums: List[int] = []
    for pr in pages:
        for region in pr.revision_regions:
            for entry in region.entries:
                num_str = entry.number.strip()
                if num_str.isdigit():
                    all_rev_nums.append(int(num_str))
    if all_rev_nums:
        rev_set = set(all_rev_nums)
        max_rev = max(rev_set)
        missing = [n for n in range(1, max_rev + 1) if n not in rev_set]
        if missing:
            findings.append(
                CheckResult(
                    check_id="DOC_REVISION_GAP",
                    severity="warning",
                    message=(
                        f"Revision sequence has gaps: missing "
                        f"revision(s) {missing} (found {sorted(rev_set)})"
                    ),
                    details={"missing": missing, "found": sorted(rev_set)},
                )
            )

    # 6. Page quality outliers: flag pages significantly below average
    # Need at least 3 pages to establish a meaningful average.  The
    # 60% threshold is empirical: below that, OCR accuracy degrades
    # noticeably and downstream analysis becomes unreliable.
    quality_scores = [pr.page_quality for pr in pages if pr.page_quality > 0]
    if len(quality_scores) >= 3:
        avg_q = sum(quality_scores) / len(quality_scores)
        threshold = avg_q * 0.6  # 60% of average is suspicious
        low_quality = [
            (pr.page, pr.page_quality)
            for pr in pages
            if 0 < pr.page_quality < threshold
        ]
        if low_quality:
            findings.append(
                CheckResult(
                    check_id="DOC_LOW_QUALITY",
                    severity="info",
                    message=(
                        f"Pages with unusually low quality: "
                        f"{[(p, f'{q:.0%}') for p, q in low_quality]} "
                        f"(avg {avg_q:.0%})"
                    ),
                    details={
                        "low_pages": low_quality,
                        "average_quality": avg_q,
                    },
                )
            )

    return findings
