"""Report viewer widget – scrollable detail panel for a selected run."""

from __future__ import annotations

import json
import os
import sys
import tkinter as tk
import webbrowser
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Any

from .widgets import CollapsibleFrame


class ReportViewerWidget:
    """Right-panel widget: renders run manifest details and handles artifact export.

    Parameters
    ----------
    parent :
        Parent tkinter widget.
    root_window :
        The top-level ``tk.Tk`` window (used for Toplevel dialogs and
        global mousewheel binding).
    """

    def __init__(self, parent, root_window: tk.Misc) -> None:
        self._root = root_window
        self._current_run_dir: Path | None = None

        self.frame = ttk.LabelFrame(parent, text="Run Details", padding=4)
        self.frame.columnconfigure(0, weight=1)
        self.frame.rowconfigure(0, weight=1)

        # Mousewheel scroll gate
        self._detail_wheel_active: bool = False

        # Scrollable canvas
        self._detail_canvas = tk.Canvas(self.frame, highlightthickness=0)
        detail_sb = ttk.Scrollbar(
            self.frame, orient="vertical", command=self._detail_canvas.yview
        )
        self._detail_inner = ttk.Frame(self._detail_canvas)
        self._detail_inner.columnconfigure(0, weight=1)
        self._detail_inner.bind(
            "<Configure>",
            lambda e: self._detail_canvas.configure(
                scrollregion=self._detail_canvas.bbox("all")
            ),
        )
        self._detail_canvas_window = self._detail_canvas.create_window(
            (0, 0), window=self._detail_inner, anchor="nw"
        )
        self._detail_canvas.configure(yscrollcommand=detail_sb.set)
        self._detail_canvas.grid(row=0, column=0, sticky="nsew")
        detail_sb.grid(row=0, column=1, sticky="ns")

        def _resize_detail(event):
            self._detail_canvas.itemconfig(
                self._detail_canvas_window, width=event.width
            )

        self._detail_canvas.bind("<Configure>", _resize_detail)
        self._detail_canvas.bind(
            "<Enter>",
            lambda e: setattr(self, "_detail_wheel_active", True),
        )
        self._detail_canvas.bind(
            "<Leave>",
            lambda e: setattr(self, "_detail_wheel_active", False),
        )

        self._root.bind_all("<MouseWheel>", self._detail_scroll, add="+")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def show_run(self, run_dir: Path, manifest: dict | None) -> None:
        """Display details for *run_dir* using the provided *manifest* (may be None)."""
        self._current_run_dir = run_dir
        self.clear()

        if manifest is None:
            ttk.Label(self._detail_inner, text="No manifest.json found.").grid(
                row=0, column=0, sticky="w", padx=6, pady=4
            )
            return

        self._populate_detail(manifest)

    def clear(self) -> None:
        """Remove all widgets from the detail panel."""
        for w in self._detail_inner.winfo_children():
            w.destroy()

    # ------------------------------------------------------------------
    # Scroll
    # ------------------------------------------------------------------

    def _detail_scroll(self, event) -> None:
        if not self._detail_wheel_active or not self._detail_canvas.winfo_ismapped():
            return
        self._detail_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    # ------------------------------------------------------------------
    # Detail panel population
    # ------------------------------------------------------------------

    def _populate_detail(self, manifest: dict) -> None:
        """Build the detail panel from a manifest dict."""
        row = 0
        pages = manifest.get("pages", [])

        # ── Summary ─────────────────────────────────────────────────
        summary_frame = CollapsibleFrame(
            self._detail_inner, "Summary", initially_open=True
        )
        summary_frame.grid(row=row, column=0, sticky="ew", pady=2, padx=4)
        summary_frame.content.columnconfigure(1, weight=1)
        row += 1

        total_ms = 0
        page_count = len(pages)
        error_count = 0
        for pd_ in pages:
            if not isinstance(pd_, dict):
                continue
            if "error" in pd_ and "stages" not in pd_:
                error_count += 1
                continue
            for si in pd_.get("stages", {}).values():
                if isinstance(si, dict):
                    total_ms += si.get("duration_ms", 0)

        srow = 0
        srow = self._add_label_pair(
            summary_frame.content, srow, "Run ID", manifest.get("run_id", "?")
        )
        srow = self._add_label_pair(
            summary_frame.content, srow, "Created", manifest.get("created_at", "?")
        )
        srow = self._add_label_pair(
            summary_frame.content, srow, "Source PDF", manifest.get("pdf_name", "?")
        )
        fp = manifest.get("input_fingerprint", "?")
        srow = self._add_label_pair(
            summary_frame.content,
            srow,
            "Fingerprint",
            fp[:16] + "..." if len(fp) > 16 else fp,
        )
        srow = self._add_label_pair(
            summary_frame.content,
            srow,
            "Resolution",
            f"{manifest.get('render_resolution_dpi', '?')} DPI",
        )
        pages_list = manifest.get("pages_processed", [])
        srow = self._add_label_pair(
            summary_frame.content, srow, "Pages", str(pages_list)
        )

        if total_ms > 0:
            timing_text = f"{self._fmt_duration(total_ms)} across {page_count} page(s)"
            if error_count:
                timing_text += f" ({error_count} errored)"
            srow = self._add_label_pair(
                summary_frame.content, srow, "Pipeline Time", timing_text
            )

        status = self._compute_status(manifest)
        status_fg = (
            "#e06c75"
            if "Failed" in status
            else "#e5c07b" if "Partial" in status else "#98c379"
        )
        srow = self._add_label_pair(
            summary_frame.content, srow, "Status", status, fg=status_fg
        )

        # ── Cross-page findings ──────────────────────────────────────
        cross_findings = manifest.get("cross_page_findings", [])
        if cross_findings:
            findings_frame = CollapsibleFrame(
                self._detail_inner, f"Cross-Page Findings ({len(cross_findings)})"
            )
            findings_frame.grid(row=row, column=0, sticky="ew", pady=2, padx=4)
            row += 1
            for i, finding in enumerate(cross_findings):
                sev = finding.get("severity", "info")
                msg = finding.get("message", str(finding))
                ttk.Label(
                    findings_frame.content,
                    text=f"[{sev.upper()}] {msg}",
                    foreground=self._severity_color(sev),
                    wraplength=400,
                ).grid(row=i, column=0, sticky="w", padx=4, pady=1)

        # ── Per-page details ─────────────────────────────────────────
        for page_data in pages:
            if not isinstance(page_data, dict):
                continue
            page_num = page_data.get("page", "?")

            if "error" in page_data and "stages" not in page_data:
                err_title = f"Page {page_num}  \u2718 ERROR"
                err_frame = CollapsibleFrame(self._detail_inner, err_title)
                err_frame.grid(row=row, column=0, sticky="ew", pady=2, padx=4)
                row += 1
                ttk.Label(
                    err_frame.content,
                    text=str(page_data["error"]),
                    foreground="#e06c75",
                    wraplength=420,
                ).grid(row=0, column=0, sticky="w", padx=4, pady=4)
                continue

            quality = page_data.get("page_quality", None)
            findings_count = page_data.get("semantic_findings_count", 0)
            pw = page_data.get("page_width", 0)
            ph = page_data.get("page_height", 0)
            skew = page_data.get("skew_degrees", 0.0)

            page_title = f"Page {page_num}"
            if quality is not None:
                page_title += (
                    f"  quality: {quality:.2f}"
                    if isinstance(quality, float)
                    else f"  quality: {quality}"
                )
            if pw and ph:
                page_title += f"  |  {pw:.0f}\u00d7{ph:.0f} pt"
            if skew:
                page_title += f"  skew: {skew:.2f}\u00b0"
            if findings_count:
                page_title += f"  [{findings_count} findings]"

            page_frame = CollapsibleFrame(self._detail_inner, page_title)
            page_frame.grid(row=row, column=0, sticky="ew", pady=2, padx=4)
            page_frame.content.columnconfigure(1, weight=1)
            row += 1

            prow = 0

            stages = page_data.get("stages", {})
            if isinstance(stages, dict) and stages:
                prow = self._add_section_heading(
                    page_frame.content, prow, "Pipeline Stages"
                )

                for stage_info in stages.values():
                    if not isinstance(stage_info, dict):
                        continue
                    s_name = stage_info.get("stage", "?")
                    s_status = stage_info.get("status", "?")
                    s_dur = stage_info.get("duration_ms", 0)
                    s_skip = stage_info.get("skip_reason")
                    s_error = stage_info.get("error")

                    if s_status == "success":
                        icon, fg = "\u2714", "#98c379"
                    elif s_status == "skipped":
                        icon, fg = "\u2500", "#7f848e"
                    else:
                        icon, fg = "\u2718", "#e06c75"

                    header = f"  {icon} {s_name}"
                    if s_dur:
                        header += f"  ({self._fmt_duration(s_dur)})"
                    if s_skip:
                        header += f"  [{s_skip}]"

                    ttk.Label(page_frame.content, text=header, foreground=fg).grid(
                        row=prow, column=0, columnspan=2, sticky="w", padx=4, pady=1
                    )
                    prow += 1

                    s_counts = stage_info.get("counts", {})
                    if s_counts:
                        interesting = [
                            f"{k}={v}"
                            for k, v in s_counts.items()
                            if isinstance(v, (int, float)) and v != 0
                        ]
                        if interesting:
                            ttk.Label(
                                page_frame.content,
                                text="      " + "  ".join(interesting),
                                foreground="#abb2bf",
                                wraplength=400,
                            ).grid(row=prow, column=0, columnspan=2, sticky="w", padx=4)
                            prow += 1

                    s_outputs = stage_info.get("outputs", {})
                    if s_outputs:
                        for out_key in ("font_names", "font_sizes"):
                            out_val = s_outputs.get(out_key)
                            if out_val and isinstance(out_val, dict):
                                top_items = sorted(
                                    out_val.items(),
                                    key=lambda x: x[1],
                                    reverse=True,
                                )[:5]
                                summary = ", ".join(f"{k} ({v})" for k, v in top_items)
                                if len(out_val) > 5:
                                    summary += f" +{len(out_val) - 5} more"
                                label_text = (
                                    "Fonts" if out_key == "font_names" else "Sizes"
                                )
                                ttk.Label(
                                    page_frame.content,
                                    text=f"      {label_text}: {summary}",
                                    foreground="#abb2bf",
                                    wraplength=400,
                                ).grid(
                                    row=prow,
                                    column=0,
                                    columnspan=2,
                                    sticky="w",
                                    padx=4,
                                )
                                prow += 1

                    if s_error:
                        err_text = (
                            str(s_error)
                            if not isinstance(s_error, dict)
                            else s_error.get("message", str(s_error))
                        )
                        ttk.Label(
                            page_frame.content,
                            text=f"      Error: {err_text}",
                            foreground="#e06c75",
                            wraplength=400,
                        ).grid(row=prow, column=0, columnspan=2, sticky="w", padx=4)
                        prow += 1

            stage_health = page_data.get("stage_health", {})
            if stage_health:
                health_items = [k for k, v in stage_health.items() if v]
                if health_items:
                    ttk.Label(
                        page_frame.content,
                        text="\u26a0 " + ", ".join(health_items),
                        foreground="#e5c07b",
                    ).grid(row=prow, column=0, columnspan=2, sticky="w", padx=4, pady=1)
                    prow += 1

            region_confs = page_data.get("region_confidences", {})
            if region_confs:
                ttk.Separator(page_frame.content, orient="horizontal").grid(
                    row=prow, column=0, columnspan=2, sticky="ew", pady=4
                )
                prow += 1
                prow = self._add_section_heading(
                    page_frame.content, prow, "Region Confidences"
                )
                for rtype, confs in region_confs.items():
                    if not confs:
                        continue
                    conf_strs = []
                    for c in confs:
                        if isinstance(c, (int, float)):
                            conf_strs.append(f"{c:.2f}")
                        else:
                            conf_strs.append(str(c))
                    min_c = min(
                        (c for c in confs if isinstance(c, (int, float))), default=1.0
                    )
                    fg = (
                        "#e06c75"
                        if min_c < 0.4
                        else "#e5c07b" if min_c < 0.6 else "#98c379"
                    )
                    ttk.Label(
                        page_frame.content,
                        text=f"  {rtype}: {', '.join(conf_strs)}",
                        foreground=fg,
                    ).grid(row=prow, column=0, columnspan=2, sticky="w", padx=4, pady=1)
                    prow += 1

            counts = page_data.get("counts", {})
            other_counts = {
                k: v
                for k, v in counts.items()
                if isinstance(v, (int, float))
            }
            if other_counts:
                ttk.Separator(page_frame.content, orient="horizontal").grid(
                    row=prow, column=0, columnspan=2, sticky="ew", pady=4
                )
                prow += 1
                prow = self._add_section_heading(page_frame.content, prow, "Counts")
                count_text = "  ".join(f"{k}={v}" for k, v in other_counts.items())
                ttk.Label(
                    page_frame.content,
                    text=count_text,
                    wraplength=420,
                    foreground="gray",
                ).grid(row=prow, column=0, columnspan=2, sticky="w", padx=4)
                prow += 1

            findings = page_data.get("semantic_findings", [])
            if findings:
                ttk.Separator(page_frame.content, orient="horizontal").grid(
                    row=prow, column=0, columnspan=2, sticky="ew", pady=4
                )
                prow += 1
                prow = self._add_section_heading(
                    page_frame.content, prow, "Semantic Findings"
                )
                for finding in findings:
                    if isinstance(finding, dict):
                        sev = finding.get("severity", "info")
                        msg = finding.get("message", str(finding))
                    else:
                        sev, msg = "info", str(finding)
                    ttk.Label(
                        page_frame.content,
                        text=f"  [{sev}] {msg}",
                        foreground=self._severity_color(sev),
                        wraplength=400,
                    ).grid(row=prow, column=0, columnspan=2, sticky="w", padx=4, pady=1)
                    prow += 1

            artifacts = page_data.get("artifacts", {})
            if artifacts:
                ttk.Separator(page_frame.content, orient="horizontal").grid(
                    row=prow, column=0, columnspan=2, sticky="ew", pady=4
                )
                prow += 1
                prow = self._add_section_heading(page_frame.content, prow, "Artifacts")
                for art_name, art_path in artifacts.items():
                    link = ttk.Label(
                        page_frame.content,
                        text=f"  {art_name}: {Path(art_path).name}",
                        foreground="#61afef",
                        cursor="hand2",
                    )
                    link.grid(
                        row=prow, column=0, columnspan=2, sticky="w", padx=4, pady=1
                    )
                    full_path = (
                        self._current_run_dir / art_path
                        if self._current_run_dir
                        else Path(art_path)
                    )
                    link.bind(
                        "<Button-1>", lambda e, p=full_path: self._open_artifact(p)
                    )
                    prow += 1

            exports = page_data.get("exports", {})
            if exports:
                ttk.Separator(page_frame.content, orient="horizontal").grid(
                    row=prow, column=0, columnspan=2, sticky="ew", pady=4
                )
                prow += 1
                prow = self._add_section_heading(page_frame.content, prow, "Exports")
                for exp_name, exp_path in exports.items():
                    link = ttk.Label(
                        page_frame.content,
                        text=f"  {exp_name}: {Path(exp_path).name}",
                        foreground="#61afef",
                        cursor="hand2",
                    )
                    link.grid(
                        row=prow, column=0, columnspan=2, sticky="w", padx=4, pady=1
                    )
                    full_path = (
                        self._current_run_dir / exp_path
                        if self._current_run_dir
                        else Path(exp_path)
                    )
                    link.bind(
                        "<Button-1>", lambda e, p=full_path: self._open_artifact(p)
                    )
                    prow += 1

        # ── Config snapshot ──────────────────────────────────────────
        config = manifest.get("config_snapshot", {})
        if config:
            cfg_frame = CollapsibleFrame(self._detail_inner, "Config Snapshot")
            cfg_frame.grid(row=row, column=0, sticky="ew", pady=2, padx=4)

            cfg_text = tk.Text(
                cfg_frame.content,
                height=8,
                wrap="word",
                bg="#1e1e1e",
                fg="#d4d4d4",
                font=("Consolas", 8),
                state="normal",
                relief="sunken",
            )
            cfg_text.pack(fill="both", expand=True, padx=2, pady=2)
            cfg_text.insert("1.0", json.dumps(config, indent=2))
            cfg_text.config(state="disabled")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_status(manifest: dict) -> str:
        """Derive a run-level status string from page entries."""
        pages = manifest.get("pages", [])
        if not pages:
            return "?"
        errors = sum(
            1
            for p in pages
            if isinstance(p, dict) and "error" in p and "stages" not in p
        )
        if errors == len(pages):
            return "\u2718 Failed"
        if errors > 0:
            return "\u26a0 Partial"
        return "\u2714 OK"

    @staticmethod
    def _fmt_duration(ms: int | float) -> str:
        """Format milliseconds into a human-readable string."""
        if ms < 1000:
            return f"{ms:.0f} ms"
        secs = ms / 1000
        if secs < 60:
            return f"{secs:.1f} s"
        mins = int(secs // 60)
        return f"{mins}m {secs - mins * 60:.1f}s"

    def _add_label_pair(
        self, parent, row: int, label: str, value: str, fg: str | None = None
    ) -> int:
        """Add a bold-label + value pair to a grid; return next row."""
        ttk.Label(
            parent,
            text=f"{label}:",
            font=("TkDefaultFont", 9, "bold"),
        ).grid(row=row, column=0, sticky="w", padx=(4, 8), pady=1)
        kw: dict[str, Any] = {"wraplength": 350}
        if fg:
            kw["foreground"] = fg
        ttk.Label(parent, text=str(value), **kw).grid(
            row=row, column=1, sticky="w", pady=1
        )
        return row + 1

    def _add_section_heading(self, parent, row: int, text: str) -> int:
        """Add a small bold heading inside a page frame; return next row."""
        ttk.Label(
            parent,
            text=text,
            font=("TkDefaultFont", 9, "bold"),
        ).grid(row=row, column=0, columnspan=2, sticky="w", padx=4, pady=(4, 1))
        return row + 1

    def _severity_color(self, sev: str) -> str:
        s = sev.lower()
        if "error" in s:
            return "#e06c75"
        if "warn" in s:
            return "#e5c07b"
        return "#d4d4d4"

    def _open_artifact(self, path: Path) -> None:
        """Open an artifact file based on its extension."""
        if not path.exists():
            messagebox.showinfo("Not Found", f"File not found: {path}")
            return

        suffix = path.suffix.lower()
        if suffix in (".png", ".jpg", ".jpeg", ".bmp"):
            if sys.platform == "win32":
                os.startfile(path)
        elif suffix == ".html":
            webbrowser.open(str(path))
        elif suffix in (".json", ".csv", ".txt"):
            self._show_text_viewer(path)
        else:
            if sys.platform == "win32":
                os.startfile(path)

    def _show_text_viewer(self, path: Path) -> None:
        """Show a file in a popup text window."""
        win = tk.Toplevel(self._root)
        win.title(f"Artifact: {path.name}")
        win.geometry("700x500")
        win.columnconfigure(0, weight=1)
        win.rowconfigure(0, weight=1)

        text = tk.Text(
            win,
            wrap="none",
            bg="#1e1e1e",
            fg="#d4d4d4",
            font=("Consolas", 9),
            state="normal",
        )
        v_sb = ttk.Scrollbar(win, orient="vertical", command=text.yview)
        h_sb = ttk.Scrollbar(win, orient="horizontal", command=text.xview)
        text.configure(yscrollcommand=v_sb.set, xscrollcommand=h_sb.set)
        text.grid(row=0, column=0, sticky="nsew")
        v_sb.grid(row=0, column=1, sticky="ns")
        h_sb.grid(row=1, column=0, sticky="ew")

        try:
            content = path.read_text(encoding="utf-8")
            if path.suffix == ".json":
                try:
                    parsed = json.loads(content)
                    content = json.dumps(parsed, indent=2)
                except Exception:  # noqa: BLE001 — show raw content if parse fails
                    pass
            text.insert("1.0", content)
        except Exception as e:
            text.insert("1.0", f"Error reading file: {e}")
        text.config(state="disabled")

        ttk.Button(win, text="Close", command=win.destroy).grid(row=2, column=0, pady=4)
