"""
Font Metrics Anomaly Detection

Detects fonts where PDF-reported character widths don't match actual rendered glyph widths.
This is common in AutoCAD-generated PDFs using fonts like RomanT, where advance widths
are ~1.7x the actual visual glyph width.

Two detection methods:
1. Heuristic: Compare reported char widths to expected ratios (fast, less accurate)
2. Visual: Compare rendered pixel extent to reported bbox (accurate, slower)

Usage:
    from plancheck.grouping.font_metrics import FontMetricsAnalyzer, VisualMetricsAnalyzer

    # Fast heuristic check
    analyzer = FontMetricsAnalyzer()
    report = analyzer.analyze_page(pdf_path, page_num)

    # Accurate visual check (uses rendered image)
    visual = VisualMetricsAnalyzer()
    visual_report = visual.analyze_page(pdf_path, page_num)
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pdfplumber
from PIL import Image


@dataclass
class FontMetricsAnomaly:
    """Represents a detected font metrics anomaly."""

    fontname: str
    sample_count: int
    avg_width_ratio: float  # reported_width / expected_width (based on size)
    inflation_factor: float  # how much the bbox is inflated (>1 means inflated)
    confidence: float  # 0-1, based on sample consistency
    sample_chars: List[str] = field(default_factory=list)
    detection_method: str = "heuristic"  # "heuristic" or "visual"

    def is_anomalous(self, threshold: float = 1.3, confidence_min: float = 0.7) -> bool:
        """Return True if font metrics are significantly inflated."""
        return self.inflation_factor > threshold and self.confidence > confidence_min

    def to_dict(self) -> dict:
        return {
            "fontname": self.fontname,
            "sample_count": self.sample_count,
            "avg_width_ratio": round(self.avg_width_ratio, 4),
            "inflation_factor": round(self.inflation_factor, 4),
            "confidence": round(self.confidence, 4),
            "is_anomalous": self.is_anomalous(),
            "detection_method": self.detection_method,
            "sample_chars": self.sample_chars[:10],  # First 10 samples
        }


@dataclass
class PageMetricsReport:
    """Full metrics report for a page."""

    page_num: int
    font_anomalies: Dict[str, FontMetricsAnomaly] = field(default_factory=dict)
    total_chars_analyzed: int = 0
    anomalous_char_count: int = 0

    def has_anomalies(self) -> bool:
        return any(a.is_anomalous() for a in self.font_anomalies.values())

    def get_anomalous_fonts(self) -> List[str]:
        return [name for name, a in self.font_anomalies.items() if a.is_anomalous()]

    def get_correction_factor(self, fontname: str) -> float:
        """Get the correction factor to apply to a font's widths."""
        if fontname in self.font_anomalies:
            anomaly = self.font_anomalies[fontname]
            if anomaly.is_anomalous():
                return 1.0 / anomaly.inflation_factor
        return 1.0

    def to_dict(self) -> dict:
        return {
            "page_num": self.page_num,
            "total_chars_analyzed": self.total_chars_analyzed,
            "anomalous_char_count": self.anomalous_char_count,
            "has_anomalies": self.has_anomalies(),
            "anomalous_fonts": self.get_anomalous_fonts(),
            "font_details": {
                name: anomaly.to_dict() for name, anomaly in self.font_anomalies.items()
            },
        }


# Expected width ratios for common characters (width / font_size)
# Based on typical proportional fonts
EXPECTED_WIDTH_RATIOS = {
    # Uppercase
    "A": 0.65,
    "B": 0.65,
    "C": 0.70,
    "D": 0.70,
    "E": 0.60,
    "F": 0.55,
    "G": 0.75,
    "H": 0.70,
    "I": 0.30,
    "J": 0.50,
    "K": 0.65,
    "L": 0.55,
    "M": 0.80,
    "N": 0.70,
    "O": 0.75,
    "P": 0.60,
    "Q": 0.75,
    "R": 0.65,
    "S": 0.60,
    "T": 0.60,
    "U": 0.70,
    "V": 0.65,
    "W": 0.90,
    "X": 0.65,
    "Y": 0.65,
    "Z": 0.60,
    # Lowercase
    "a": 0.50,
    "b": 0.55,
    "c": 0.50,
    "d": 0.55,
    "e": 0.50,
    "f": 0.30,
    "g": 0.55,
    "h": 0.55,
    "i": 0.25,
    "j": 0.25,
    "k": 0.50,
    "l": 0.25,
    "m": 0.80,
    "n": 0.55,
    "o": 0.55,
    "p": 0.55,
    "q": 0.55,
    "r": 0.35,
    "s": 0.45,
    "t": 0.35,
    "u": 0.55,
    "v": 0.50,
    "w": 0.75,
    "x": 0.50,
    "y": 0.50,
    "z": 0.45,
    # Digits
    "0": 0.55,
    "1": 0.35,
    "2": 0.55,
    "3": 0.55,
    "4": 0.55,
    "5": 0.55,
    "6": 0.55,
    "7": 0.55,
    "8": 0.55,
    "9": 0.55,
}


class FontMetricsAnalyzer:
    """Analyzes font metrics to detect anomalies in PDF text extraction."""

    def __init__(
        self,
        inflation_threshold: float = 1.3,
        min_samples: int = 5,
        confidence_min: float = 0.7,
    ):
        """
        Args:
            inflation_threshold: Fonts with inflation_factor above this are flagged.
            min_samples: Minimum character samples required to assess a font.
            confidence_min: Minimum confidence to consider an anomaly valid.
        """
        self.inflation_threshold = inflation_threshold
        self.min_samples = min_samples
        self.confidence_min = confidence_min

    def analyze_page(self, pdf_path: Path | str, page_num: int) -> PageMetricsReport:
        """
        Analyze a page for font metrics anomalies.

        Args:
            pdf_path: Path to PDF file
            page_num: Page number (0-indexed)

        Returns:
            PageMetricsReport with detected anomalies
        """
        report = PageMetricsReport(page_num=page_num)

        # Collect character metrics by font
        font_chars: Dict[str, List[dict]] = {}

        with pdfplumber.open(str(pdf_path)) as pdf:
            if page_num >= len(pdf.pages):
                return report

            page = pdf.pages[page_num]
            chars = page.chars

            for char in chars:
                text = char.get("text", "")
                fontname = char.get("fontname", "Unknown")

                # Skip non-printable or space characters
                if not text or text.isspace():
                    continue

                if fontname not in font_chars:
                    font_chars[fontname] = []
                font_chars[fontname].append(char)

        # Analyze each font
        for fontname, chars in font_chars.items():
            anomaly = self._analyze_font(fontname, chars)
            if anomaly:
                report.font_anomalies[fontname] = anomaly
                report.total_chars_analyzed += anomaly.sample_count
                if anomaly.is_anomalous():
                    report.anomalous_char_count += anomaly.sample_count

        return report

    def _analyze_font(
        self, fontname: str, chars: List[dict]
    ) -> Optional[FontMetricsAnomaly]:
        """Analyze a font's character metrics for anomalies."""

        width_ratios = []
        sample_chars = []

        for char in chars:
            text = char.get("text", "")

            # Only analyze characters we have expected ratios for
            if text not in EXPECTED_WIDTH_RATIOS:
                continue

            char_width = char.get("width", 0)
            font_size = char.get("size", 0)

            if font_size <= 0 or char_width <= 0:
                continue

            # Calculate actual ratio
            actual_ratio = char_width / font_size
            expected_ratio = EXPECTED_WIDTH_RATIOS[text]

            # Calculate inflation (how much bigger than expected)
            if expected_ratio > 0:
                inflation = actual_ratio / expected_ratio
                width_ratios.append(inflation)
                sample_chars.append(text)

        if len(width_ratios) < self.min_samples:
            # Not enough samples to make a determination
            return None

        # Calculate statistics
        avg_inflation = sum(width_ratios) / len(width_ratios)

        # Calculate consistency (standard deviation relative to mean)
        variance = sum((r - avg_inflation) ** 2 for r in width_ratios) / len(
            width_ratios
        )
        std_dev = variance**0.5

        # Confidence based on consistency (lower variance = higher confidence)
        # If std_dev is small relative to mean, we're confident
        coefficient_of_variation = std_dev / avg_inflation if avg_inflation > 0 else 1.0
        confidence = max(0.0, min(1.0, 1.0 - coefficient_of_variation))

        return FontMetricsAnomaly(
            fontname=fontname,
            sample_count=len(width_ratios),
            avg_width_ratio=sum(
                r * EXPECTED_WIDTH_RATIOS.get(c, 0.5)
                for r, c in zip(width_ratios, sample_chars)
            )
            / len(width_ratios),
            inflation_factor=avg_inflation,
            confidence=confidence,
            sample_chars=sample_chars,
        )

    def correct_box_width(
        self, x0: float, x1: float, fontname: str, report: PageMetricsReport
    ) -> Tuple[float, float]:
        """
        Apply correction to a box's x-coordinates based on font metrics.

        Args:
            x0: Original left x coordinate
            x1: Original right x coordinate
            fontname: Font name for the text
            report: PageMetricsReport with anomaly data

        Returns:
            Tuple of (corrected_x0, corrected_x1)
        """
        factor = report.get_correction_factor(fontname)
        if factor == 1.0:
            return x0, x1

        # Shrink from the right side (text is left-aligned)
        width = x1 - x0
        corrected_width = width * factor
        corrected_x1 = x0 + corrected_width

        return x0, corrected_x1


def analyze_pdf(
    pdf_path: Path | str, pages: Optional[List[int]] = None
) -> Dict[int, PageMetricsReport]:
    """
    Analyze an entire PDF for font metrics anomalies.

    Args:
        pdf_path: Path to PDF file
        pages: Optional list of page numbers to analyze (0-indexed).
               If None, analyzes all pages.

    Returns:
        Dict mapping page number to PageMetricsReport
    """
    analyzer = FontMetricsAnalyzer()
    results = {}

    with pdfplumber.open(str(pdf_path)) as pdf:
        if pages is None:
            pages = range(len(pdf.pages))

        for page_num in pages:
            results[page_num] = analyzer.analyze_page(pdf_path, page_num)

    return results


def save_metrics_report(report: PageMetricsReport, output_path: Path | str) -> None:
    """Save a metrics report to JSON."""
    Path(output_path).write_text(json.dumps(report.to_dict(), indent=2))


@dataclass
class WordVisualAnomaly:
    """Represents a detected visual anomaly for a specific word."""

    text: str
    fontname: str
    reported_bbox: Tuple[float, float, float, float]  # x0, y0, x1, y1
    visual_bbox: Tuple[float, float, float, float]  # x0, y0, x1, y1
    reported_width: float
    visual_width: float
    inflation_factor: float
    overhang_percent: float  # % of reported width that's empty

    def is_anomalous(self, threshold: float = 1.3) -> bool:
        return self.inflation_factor > threshold

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "fontname": self.fontname,
            "reported_bbox": [float(x) for x in self.reported_bbox],
            "visual_bbox": [float(x) for x in self.visual_bbox],
            "reported_width": float(round(self.reported_width, 2)),
            "visual_width": float(round(self.visual_width, 2)),
            "inflation_factor": float(round(self.inflation_factor, 3)),
            "overhang_percent": float(round(self.overhang_percent, 1)),
            "is_anomalous": bool(self.is_anomalous()),
        }


@dataclass
class VisualMetricsReport:
    """Report from visual (pixel-based) font metrics analysis."""

    page_num: int
    resolution: int
    word_anomalies: List[WordVisualAnomaly] = field(default_factory=list)
    font_inflation_factors: Dict[str, float] = field(default_factory=dict)

    def has_anomalies(self, threshold: float = 1.3) -> bool:
        return any(a.is_anomalous(threshold) for a in self.word_anomalies)

    def get_anomalous_fonts(self, threshold: float = 1.3) -> List[str]:
        return list(
            set(a.fontname for a in self.word_anomalies if a.is_anomalous(threshold))
        )

    def get_correction_factor(self, fontname: str) -> float:
        """Get correction factor for a font based on visual analysis."""
        if fontname in self.font_inflation_factors:
            factor = self.font_inflation_factors[fontname]
            if factor > 1.3:
                return 1.0 / factor
        return 1.0

    def to_dict(self) -> dict:
        return {
            "page_num": self.page_num,
            "resolution": self.resolution,
            "has_anomalies": self.has_anomalies(),
            "anomalous_fonts": self.get_anomalous_fonts(),
            "font_inflation_factors": {
                k: round(v, 3) for k, v in self.font_inflation_factors.items()
            },
            "word_anomalies": [a.to_dict() for a in self.word_anomalies],
        }


class VisualMetricsAnalyzer:
    """
    Analyzes font metrics by comparing rendered pixels to reported bboxes.

    This is more accurate than heuristic analysis but slower since it requires
    rendering the page.
    """

    def __init__(self, resolution: int = 300, dark_threshold: int = 200):
        """
        Args:
            resolution: DPI for rendering (higher = more accurate but slower)
            dark_threshold: Grayscale value below which pixels are considered "dark" (text)
        """
        self.resolution = resolution
        self.dark_threshold = dark_threshold

    def analyze_page(
        self,
        pdf_path: Path | str,
        page_num: int,
        sample_words: Optional[List[str]] = None,
    ) -> VisualMetricsReport:
        """
        Analyze a page for visual font metrics anomalies.

        Args:
            pdf_path: Path to PDF file
            page_num: Page number (0-indexed)
            sample_words: Optional list of specific words to analyze.
                         If None, samples words from each font.

        Returns:
            VisualMetricsReport with detected anomalies
        """
        report = VisualMetricsReport(page_num=page_num, resolution=self.resolution)
        scale = self.resolution / 72.0

        with pdfplumber.open(str(pdf_path)) as pdf:
            if page_num >= len(pdf.pages):
                return report

            page = pdf.pages[page_num]

            # Render page
            img = page.to_image(resolution=self.resolution)
            pil_img = img.original
            gray_arr = np.array(pil_img.convert("L"))

            # Get words and their fonts (need to map words to chars for font info)
            words = page.extract_words()
            chars = page.chars

            # Group chars by font
            font_words: Dict[str, List[dict]] = {}

            for word in words:
                # Find a char that overlaps with this word to get font
                word_chars = [
                    c
                    for c in chars
                    if c["x0"] >= word["x0"] - 1
                    and c["x1"] <= word["x1"] + 1
                    and c["top"] >= word["top"] - 1
                    and c["bottom"] <= word["bottom"] + 1
                ]

                if word_chars:
                    fontname = word_chars[0].get("fontname", "Unknown")
                    if fontname not in font_words:
                        font_words[fontname] = []
                    font_words[fontname].append({**word, "fontname": fontname})

            # Sample words from each font (or use provided list)
            words_to_analyze = []

            if sample_words:
                # Find specific words
                for word in words:
                    if word.get("text", "") in sample_words:
                        word_chars = [
                            c
                            for c in chars
                            if c["x0"] >= word["x0"] - 1 and c["x1"] <= word["x1"] + 1
                        ]
                        fontname = (
                            word_chars[0].get("fontname", "Unknown")
                            if word_chars
                            else "Unknown"
                        )
                        words_to_analyze.append({**word, "fontname": fontname})
            else:
                # Sample up to 5 words per font (prefer longer words)
                for fontname, font_word_list in font_words.items():
                    # Sort by length, take longest words
                    sorted_words = sorted(
                        font_word_list,
                        key=lambda w: len(w.get("text", "")),
                        reverse=True,
                    )
                    words_to_analyze.extend(sorted_words[:5])

            # Analyze each word
            font_inflations: Dict[str, List[float]] = {}

            for word in words_to_analyze:
                anomaly = self._analyze_word(
                    word, gray_arr, scale, page.width, page.height
                )
                if anomaly:
                    report.word_anomalies.append(anomaly)

                    fontname = anomaly.fontname
                    if fontname not in font_inflations:
                        font_inflations[fontname] = []
                    font_inflations[fontname].append(anomaly.inflation_factor)

            # Calculate average inflation per font
            for fontname, inflations in font_inflations.items():
                if inflations:
                    report.font_inflation_factors[fontname] = sum(inflations) / len(
                        inflations
                    )

        return report

    def _analyze_word(
        self,
        word: dict,
        gray_arr: np.ndarray,
        scale: float,
        page_width: float,
        page_height: float,
    ) -> Optional[WordVisualAnomaly]:
        """Analyze a single word's visual extent vs reported bbox."""

        text = word.get("text", "")
        if not text or len(text) < 3:
            return None

        fontname = word.get("fontname", "Unknown")

        # Reported bbox
        rep_x0 = word["x0"]
        rep_x1 = word["x1"]
        rep_y0 = word["top"]
        rep_y1 = word["bottom"]

        # Convert to pixel coordinates with some padding
        pad = 5  # pixels
        px_x0 = max(0, int(rep_x0 * scale) - pad)
        px_x1 = min(gray_arr.shape[1], int(rep_x1 * scale) + pad)
        px_y0 = max(0, int(rep_y0 * scale) - pad)
        px_y1 = min(gray_arr.shape[0], int(rep_y1 * scale) + pad)

        if px_x1 <= px_x0 or px_y1 <= px_y0:
            return None

        # Extract region
        region = gray_arr[px_y0:px_y1, px_x0:px_x1]

        # Find dark pixels
        has_dark = np.any(region < self.dark_threshold, axis=0)
        dark_cols = np.where(has_dark)[0]

        if len(dark_cols) < 3:
            return None

        # Visual extent in pixel coordinates
        vis_px_x0 = px_x0 + dark_cols[0]
        vis_px_x1 = px_x0 + dark_cols[-1]

        # Convert back to PDF coordinates
        vis_x0 = vis_px_x0 / scale
        vis_x1 = vis_px_x1 / scale

        reported_width = rep_x1 - rep_x0
        visual_width = vis_x1 - vis_x0

        if visual_width <= 0:
            return None

        inflation_factor = reported_width / visual_width
        overhang = reported_width - visual_width
        overhang_percent = (
            (overhang / reported_width) * 100 if reported_width > 0 else 0
        )

        return WordVisualAnomaly(
            text=text,
            fontname=fontname,
            reported_bbox=(rep_x0, rep_y0, rep_x1, rep_y1),
            visual_bbox=(vis_x0, rep_y0, vis_x1, rep_y1),
            reported_width=reported_width,
            visual_width=visual_width,
            inflation_factor=inflation_factor,
            overhang_percent=overhang_percent,
        )


if __name__ == "__main__":
    # Test on the problematic PDF
    import sys

    pdf_path = Path(
        "input/IFC Operations Facilities McClain County - Drawings 25_0915.pdf"
    )
    if not pdf_path.exists():
        print(f"PDF not found: {pdf_path}")
        sys.exit(1)

    print(f"Analyzing: {pdf_path.name}")
    print("=" * 70)

    # Heuristic analysis (fast)
    print("\n1. HEURISTIC ANALYSIS (based on expected char width ratios)")
    print("-" * 70)

    analyzer = FontMetricsAnalyzer()
    report = analyzer.analyze_page(pdf_path, 2)

    print(f"  Total chars analyzed: {report.total_chars_analyzed}")
    print(f"  Has anomalies: {report.has_anomalies()}")

    for fontname, anomaly in report.font_anomalies.items():
        status = "⚠️  ANOMALOUS" if anomaly.is_anomalous() else "✓ Normal"
        print(f"\n  {fontname}: {status}")
        print(
            f"    Samples: {anomaly.sample_count}, Inflation: {anomaly.inflation_factor:.2f}x"
        )

    # Visual analysis (accurate but slower)
    print("\n2. VISUAL ANALYSIS (pixel-based comparison)")
    print("-" * 70)

    visual = VisualMetricsAnalyzer(resolution=200)
    # Analyze without specific words to get samples from each font region
    visual_report = visual.analyze_page(pdf_path, 2)

    # Filter to just anomalous words
    anomalous_words = [a for a in visual_report.word_anomalies if a.is_anomalous()]
    normal_words = [a for a in visual_report.word_anomalies if not a.is_anomalous()]

    print(f"  Words analyzed: {len(visual_report.word_anomalies)}")
    print(f"  Anomalous words: {len(anomalous_words)}")
    print(f"  Normal words: {len(normal_words)}")

    if anomalous_words:
        print("\n  ⚠️  ANOMALOUS WORDS DETECTED:")
        for anomaly in anomalous_words:
            print(
                f"\n    '{anomaly.text}' at ({anomaly.reported_bbox[0]:.0f}, {anomaly.reported_bbox[1]:.0f})"
            )
            print(f"      Font: {anomaly.fontname}")
            print(f"      Reported width: {anomaly.reported_width:.1f}")
            print(f"      Visual width:   {anomaly.visual_width:.1f}")
            print(
                f"      Inflation:      {anomaly.inflation_factor:.2f}x ({anomaly.overhang_percent:.0f}% empty)"
            )
            print(
                f"      → Correction: shrink x1 by {anomaly.reported_width - anomaly.visual_width:.1f}"
            )

    # Save report
    output_path = Path("font_metrics_report.json")
    output_path.write_text(json.dumps(visual_report.to_dict(), indent=2))
    print(f"\n  Report saved to: {output_path}")
