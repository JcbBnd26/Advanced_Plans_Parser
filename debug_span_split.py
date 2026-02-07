import pdfplumber

from plancheck.config import GroupingConfig
from plancheck.grouping import build_lines, compute_median_space_gap, split_line_spans
from plancheck.models import GlyphBox

pdf_path = "input/IFC Operations Facilities McClain County - Drawings 25_0915.pdf"
with pdfplumber.open(pdf_path) as pdf:
    page = pdf.pages[1]  # page 2
    words = page.extract_words(keep_blank_chars=False, use_text_flow=False)
    tokens = [
        GlyphBox(
            page=1,
            x0=float(w["x0"]),
            y0=float(w["top"]),
            x1=float(w["x1"]),
            y1=float(w["bottom"]),
            text=w["text"],
        )
        for w in words
    ]

cfg = GroupingConfig()
lines = build_lines(tokens, cfg)
msg = compute_median_space_gap(lines, tokens)
print(f"median_space_gap = {msg:.4f}")
print(f"span_gap_mult = {cfg.span_gap_mult}")
print(f"threshold = {msg * cfg.span_gap_mult:.4f}")
print()

for line in lines:
    if line.line_id in (106, 107):
        print(f"=== Line {line.line_id} ===")
        print(f"baseline_y = {line.baseline_y:.2f}")
        sorted_indices = sorted(line.token_indices, key=lambda i: tokens[i].x0)
        print(f"Tokens ({len(sorted_indices)}):")
        for pos, idx in enumerate(sorted_indices):
            t = tokens[idx]
            if pos < len(sorted_indices) - 1:
                next_idx = sorted_indices[pos + 1]
                gap = tokens[next_idx].x0 - t.x1
            else:
                gap = None
            gap_str = f"  gap_to_next={gap:.2f}" if gap is not None else ""
            thresh_mark = ""
            if gap is not None and gap > msg * cfg.span_gap_mult:
                thresh_mark = " ** SPLIT **"
            print(
                f"  [{idx}] x0={t.x0:.2f} x1={t.x1:.2f} text='{t.text}'{gap_str}{thresh_mark}"
            )

        # Now show spans
        split_line_spans(line, tokens, msg, cfg.span_gap_mult)
        print(f"Spans ({len(line.spans)}):")
        for si, span in enumerate(line.spans):
            print(f"  S{si}: col_id={span.col_id} text='{span.text(tokens)}'")
        print()
