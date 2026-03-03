"""Tab – Database: browse all persisted detections, corrections, training
history, groups, and snapshots directly from the SQLite correction store.

This tab is fully independent of the pipeline — it reads whatever the DB
contains, including results from past runs that are no longer on disk.
"""

from __future__ import annotations

import tkinter as tk
from datetime import datetime
from tkinter import messagebox, ttk
from typing import Any

from plancheck.corrections.store import CorrectionStore

from .widgets import CollapsibleFrame


def _fmt_bytes(n: int) -> str:
    """Human-readable byte size."""
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024:
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} {unit}"
        n /= 1024  # type: ignore[assignment]
    return f"{n:.1f} TB"


def _fmt_ts(ts: str | None) -> str:
    """Format an ISO timestamp for display, or em-dash."""
    if not ts:
        return "—"
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d  %H:%M:%S")
    except Exception:
        return ts[:19] if len(ts) >= 19 else ts


class DatabaseTab:
    """Tab: Database inspector — read-only view of the CorrectionStore."""

    def __init__(self, notebook: ttk.Notebook, gui_state: Any) -> None:
        self.notebook = notebook
        self.state = gui_state
        self.root = notebook.winfo_toplevel()

        self.frame = ttk.Frame(notebook)
        self.frame.columnconfigure(0, weight=0)
        self.frame.columnconfigure(1, weight=1)
        self.frame.rowconfigure(1, weight=1)
        notebook.add(self.frame, text="Database")

        self._store = CorrectionStore()
        self._selected_doc_id: str | None = None
        self._selected_run_id: str | None = None

        # Mousewheel scroll state for the right-hand detail panel
        self._detail_wheel_active: bool = False

        self._build_ui()
        # Defer initial refresh until the tab is first selected so the
        # user doesn't see stale historical data on startup.
        self._needs_refresh = True
        self.notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed, add="+")

        # Auto-refresh when a pipeline run finishes or PDF changes
        self.state.subscribe("run_completed", self._refresh)
        self.state.subscribe("pdf_changed", self._refresh)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        pad = {"padx": 6, "pady": 4}

        # ── Toolbar ──────────────────────────────────────────────────
        toolbar = ttk.Frame(self.frame)
        toolbar.grid(row=0, column=0, columnspan=2, sticky="ew", **pad)
        ttk.Button(toolbar, text="Refresh", command=self._refresh).pack(
            side="left", padx=2
        )
        ttk.Button(toolbar, text="Snapshot DB", command=self._on_snapshot).pack(
            side="left", padx=2
        )
        ttk.Button(toolbar, text="Restore…", command=self._on_restore).pack(
            side="left", padx=2
        )
        self._status_label = ttk.Label(toolbar, text="", foreground="gray")
        self._status_label.pack(side="right", padx=8)

        # ── Left panel: document selector ────────────────────────────
        left = ttk.LabelFrame(self.frame, text="Documents", padding=4)
        left.grid(row=1, column=0, sticky="nsew", **pad)
        left.rowconfigure(0, weight=1)
        left.columnconfigure(0, weight=1)

        self._doc_tree = ttk.Treeview(
            left,
            columns=("filename", "pages"),
            show="headings",
            height=12,
            selectmode="browse",
        )
        self._doc_tree.heading("filename", text="Filename")
        self._doc_tree.heading("pages", text="Pages")
        self._doc_tree.column("filename", width=180, minwidth=120)
        self._doc_tree.column("pages", width=50, minwidth=40, anchor="center")
        self._doc_tree.grid(row=0, column=0, sticky="nsew")

        doc_sb = ttk.Scrollbar(left, orient="vertical", command=self._doc_tree.yview)
        doc_sb.grid(row=0, column=1, sticky="ns")
        self._doc_tree.configure(yscrollcommand=doc_sb.set)
        self._doc_tree.bind("<<TreeviewSelect>>", self._on_doc_selected)

        # Run filter
        run_frame = ttk.Frame(left)
        run_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(4, 0))
        ttk.Label(run_frame, text="Run:").pack(side="left")
        self._run_var = tk.StringVar(value="(latest)")
        self._run_combo = ttk.Combobox(
            run_frame,
            textvariable=self._run_var,
            state="readonly",
            width=28,
        )
        self._run_combo.pack(side="left", fill="x", expand=True, padx=4)
        self._run_combo.bind("<<ComboboxSelected>>", self._on_run_selected)

        # ── Right panel: scrollable detail area ──────────────────────
        right = ttk.Frame(self.frame)
        right.grid(row=1, column=1, sticky="nsew", **pad)
        right.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=1)

        self._detail_canvas = tk.Canvas(right, highlightthickness=0, borderwidth=0)
        detail_sb = ttk.Scrollbar(
            right, orient="vertical", command=self._detail_canvas.yview
        )
        self._detail_inner = ttk.Frame(self._detail_canvas)
        self._detail_inner.columnconfigure(0, weight=1)

        self._detail_window = self._detail_canvas.create_window(
            (0, 0), window=self._detail_inner, anchor="nw"
        )
        self._detail_canvas.configure(yscrollcommand=detail_sb.set)
        self._detail_canvas.grid(row=0, column=0, sticky="nsew")
        detail_sb.grid(row=0, column=1, sticky="ns")

        self._detail_inner.bind(
            "<Configure>",
            lambda e: self._detail_canvas.configure(
                scrollregion=self._detail_canvas.bbox("all")
            ),
        )
        self._detail_canvas.bind(
            "<Configure>",
            lambda e: self._detail_canvas.itemconfig(
                self._detail_window, width=e.width
            ),
        )

        # Mousewheel scroll
        self._detail_canvas.bind(
            "<Enter>",
            lambda e: setattr(self, "_detail_wheel_active", True),
        )
        self._detail_canvas.bind(
            "<Leave>",
            lambda e: setattr(self, "_detail_wheel_active", False),
        )

        # Bind once globally; handler is gated by _detail_wheel_active
        self.root.bind_all("<MouseWheel>", self._on_detail_mousewheel, add="+")

    def _on_detail_mousewheel(self, event) -> None:
        if not self._detail_wheel_active or not self._detail_canvas.winfo_ismapped():
            return
        self._detail_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    # ------------------------------------------------------------------
    # Refresh / populate
    # ------------------------------------------------------------------

    def _on_tab_changed(self, event: tk.Event) -> None:
        """Lazy-load on first visit to the Database tab."""
        if not self._needs_refresh:
            return
        try:
            selected = self.notebook.index(self.notebook.select())
            my_index = self.notebook.index(self.frame)
        except Exception:
            return
        if selected == my_index:
            self._refresh()
            self._needs_refresh = False

    def _refresh(self) -> None:
        """Reload everything from the database."""
        self._populate_doc_tree()
        self._populate_overview()
        now = datetime.now().strftime("%H:%M:%S")
        self._status_label.configure(text=f"Refreshed {now}")
        self._needs_refresh = False

    def _populate_doc_tree(self) -> None:
        self._doc_tree.delete(*self._doc_tree.get_children())
        docs = self._store.get_all_documents()
        for doc in docs:
            self._doc_tree.insert(
                "",
                "end",
                iid=doc["doc_id"],
                values=(doc["filename"], doc["page_count"]),
            )

    def _on_doc_selected(self, _event: tk.Event | None = None) -> None:
        sel = self._doc_tree.selection()
        if not sel:
            return
        self._selected_doc_id = sel[0]
        # Populate run combo
        runs = self._store.get_run_ids_for_doc(self._selected_doc_id)
        choices = ["(all runs)"] + runs
        self._run_combo["values"] = choices
        if runs:
            self._run_var.set(runs[0])  # latest
            self._selected_run_id = runs[0]
        else:
            self._run_var.set("(all runs)")
            self._selected_run_id = None
        self._populate_detail()

    def _on_run_selected(self, _event: tk.Event | None = None) -> None:
        val = self._run_var.get()
        self._selected_run_id = None if val == "(all runs)" else val
        self._populate_detail()

    # ------------------------------------------------------------------
    # Detail panel
    # ------------------------------------------------------------------

    def _clear_detail(self) -> None:
        for w in self._detail_inner.winfo_children():
            w.destroy()

    def _populate_overview(self) -> None:
        """Show global DB overview when no doc is selected."""
        self._clear_detail()
        row = 0

        # ── DB Overview ──────────────────────────────────────────────
        ov = self._store.get_db_overview()
        sec = CollapsibleFrame(self._detail_inner, "DB Overview", initially_open=True)
        sec.grid(row=row, column=0, sticky="ew", pady=2, padx=4)
        row += 1

        pairs = [
            ("DB Path", ov["db_path"]),
            ("DB Size", _fmt_bytes(ov["db_size_bytes"])),
            ("Documents", str(ov["total_documents"])),
            ("Detections", str(ov["total_detections"])),
            ("Corrections", str(ov["total_corrections"])),
            ("Groups", str(ov["total_groups"])),
            ("Training Runs", str(ov["total_training_runs"])),
            ("Last Detection", _fmt_ts(ov["last_detection_at"])),
            ("Last Correction", _fmt_ts(ov["last_correction_at"])),
        ]
        for i, (label, val) in enumerate(pairs):
            ttk.Label(
                sec.content,
                text=f"{label}:",
                font=("TkDefaultFont", 9, "bold"),
            ).grid(row=i, column=0, sticky="w", padx=(4, 8), pady=1)
            lbl = ttk.Label(sec.content, text=str(val), wraplength=420)
            lbl.grid(row=i, column=1, sticky="w", pady=1)

        # ── Retrain Status ───────────────────────────────────────────
        retrain_sec = CollapsibleFrame(
            self._detail_inner, "Training Status", initially_open=True
        )
        retrain_sec.grid(row=row, column=0, sticky="ew", pady=2, padx=4)
        row += 1

        pending = self._store.count_corrections_since_last_train()
        needs_retrain = self._store.should_retrain()
        last_train = self._store.last_train_date()
        cache = self._store.cache_stats()

        train_pairs = [
            ("Last Train", _fmt_ts(last_train)),
            (
                "Pending Corrections",
                f"{pending}" + ("  ⚠ retrain recommended" if needs_retrain else ""),
            ),
            ("Feature Cache", f"{cache.get('total_entries', 0)} entries"),
        ]
        rt_fg = "#e5c07b" if needs_retrain else "#98c379"
        for i, (label, val) in enumerate(train_pairs):
            ttk.Label(
                retrain_sec.content,
                text=f"{label}:",
                font=("TkDefaultFont", 9, "bold"),
            ).grid(row=i, column=0, sticky="w", padx=(4, 8), pady=1)
            fg = rt_fg if "retrain" in val else None
            kw = {"foreground": fg} if fg else {}
            ttk.Label(retrain_sec.content, text=str(val), **kw).grid(
                row=i, column=1, sticky="w", pady=1
            )

        # ── Correction Breakdown (global) ────────────────────────────
        breakdown = self._store.get_correction_type_breakdown()
        if breakdown:
            cb_sec = CollapsibleFrame(
                self._detail_inner, "Corrections by Type", initially_open=True
            )
            cb_sec.grid(row=row, column=0, sticky="ew", pady=2, padx=4)
            row += 1
            for i, (ctype, cnt) in enumerate(sorted(breakdown.items())):
                ttk.Label(
                    cb_sec.content,
                    text=f"{ctype}:",
                    font=("TkDefaultFont", 9, "bold"),
                ).grid(row=i, column=0, sticky="w", padx=(4, 8), pady=1)
                ttk.Label(cb_sec.content, text=str(cnt)).grid(
                    row=i, column=1, sticky="w", pady=1
                )

        # ── Training History ─────────────────────────────────────────
        history = self._store.get_training_history()
        if history:
            th_sec = CollapsibleFrame(
                self._detail_inner,
                f"Training History ({len(history)} runs)",
            )
            th_sec.grid(row=row, column=0, sticky="ew", pady=2, padx=4)
            row += 1
            # Header
            for ci, hdr in enumerate(["Date", "Train", "Val", "Accuracy", "F1 Macro"]):
                ttk.Label(
                    th_sec.content,
                    text=hdr,
                    font=("TkDefaultFont", 8, "bold"),
                    foreground="gray",
                ).grid(row=0, column=ci, sticky="w", padx=4)
            for ri, run in enumerate(history[:15], start=1):
                ts_short = _fmt_ts(run.get("trained_at"))
                vals = [
                    ts_short,
                    str(run.get("n_train", "")),
                    str(run.get("n_val", "")),
                    f"{run['accuracy']:.3f}" if run.get("accuracy") else "—",
                    f"{run['f1_macro']:.3f}" if run.get("f1_macro") else "—",
                ]
                for ci, v in enumerate(vals):
                    ttk.Label(th_sec.content, text=v, font=("TkDefaultFont", 8)).grid(
                        row=ri, column=ci, sticky="w", padx=4, pady=1
                    )

        # ── Candidate Outcomes ───────────────────────────────────────
        outcomes = self._store.count_candidate_outcomes()
        if outcomes.get("total", 0) > 0:
            co_sec = CollapsibleFrame(self._detail_inner, "VOCR Candidate Outcomes")
            co_sec.grid(row=row, column=0, sticky="ew", pady=2, padx=4)
            row += 1
            total = outcomes["total"]
            hits = outcomes["hits"]
            rate = hits / total * 100 if total else 0
            for i, (label, val) in enumerate(
                [
                    ("Total", str(total)),
                    ("Hits", str(hits)),
                    ("Misses", str(outcomes["misses"])),
                    ("Hit Rate", f"{rate:.1f}%"),
                ]
            ):
                ttk.Label(
                    co_sec.content,
                    text=f"{label}:",
                    font=("TkDefaultFont", 9, "bold"),
                ).grid(row=i, column=0, sticky="w", padx=(4, 8), pady=1)
                ttk.Label(co_sec.content, text=val).grid(
                    row=i, column=1, sticky="w", pady=1
                )

        # ── Snapshots ────────────────────────────────────────────────
        snaps = self._store.list_snapshots()
        snap_sec = CollapsibleFrame(
            self._detail_inner,
            f"Snapshots ({len(snaps)})",
        )
        snap_sec.grid(row=row, column=0, sticky="ew", pady=2, padx=4)
        row += 1
        if snaps:
            for i, snap in enumerate(snaps[:10]):
                path = snap.get("path", "")
                tag = snap.get("tag", "")
                size = snap.get("size", 0)
                ts = snap.get("created", "")
                text = f"{_fmt_ts(ts)}  {tag}  ({_fmt_bytes(size)})"
                ttk.Label(snap_sec.content, text=text, font=("TkDefaultFont", 8)).grid(
                    row=i, column=0, sticky="w", padx=4, pady=1
                )
        else:
            ttk.Label(
                snap_sec.content,
                text="No snapshots yet",
                foreground="gray",
            ).grid(row=0, column=0, sticky="w", padx=4)

        # ── Recent Corrections ───────────────────────────────────────
        recent = self._store.get_recent_corrections(limit=20)
        if recent:
            rc_sec = CollapsibleFrame(
                self._detail_inner,
                f"Recent Corrections (last {len(recent)})",
            )
            rc_sec.grid(row=row, column=0, sticky="ew", pady=2, padx=4)
            row += 1
            for ci, hdr in enumerate(["Time", "Page", "Type", "Label"]):
                ttk.Label(
                    rc_sec.content,
                    text=hdr,
                    font=("TkDefaultFont", 8, "bold"),
                    foreground="gray",
                ).grid(row=0, column=ci, sticky="w", padx=4)
            for ri, c in enumerate(recent, start=1):
                ts = _fmt_ts(c.get("corrected_at"))
                vals = [
                    ts,
                    str(c.get("page", "")),
                    c.get("correction_type", ""),
                    f"{c.get('original_label', '?')} → {c.get('corrected_label', '?')}",
                ]
                type_colors = {
                    "accept": "#98c379",
                    "relabel": "#61afef",
                    "reshape": "#c678dd",
                    "delete": "#e06c75",
                    "add": "#e5c07b",
                }
                fg = type_colors.get(c.get("correction_type", ""), None)
                for ci, v in enumerate(vals):
                    kw = {"foreground": fg} if fg and ci == 2 else {}
                    ttk.Label(
                        rc_sec.content, text=v, font=("TkDefaultFont", 8), **kw
                    ).grid(row=ri, column=ci, sticky="w", padx=4, pady=1)

    def _populate_detail(self) -> None:
        """Show detail for the selected document + optional run filter."""
        if not self._selected_doc_id:
            self._populate_overview()
            return

        self._clear_detail()
        row = 0

        # ── Document Summary ─────────────────────────────────────────
        summary = self._store.get_doc_summary(self._selected_doc_id)
        if not summary:
            self._populate_overview()
            return

        sec = CollapsibleFrame(
            self._detail_inner, "Document Summary", initially_open=True
        )
        sec.grid(row=row, column=0, sticky="ew", pady=2, padx=4)
        row += 1

        dims = ""
        if summary.get("page_width") and summary.get("page_height"):
            dims = f"  ({summary['page_width']:.0f} × {summary['page_height']:.0f} pt)"

        pairs = [
            ("Filename", summary.get("filename", "?")),
            ("Doc ID", summary.get("doc_id", "?")[:24] + "…"),
            ("Pages", f"{summary.get('page_count', '?')}{dims}"),
            ("Ingested", _fmt_ts(summary.get("ingested_at"))),
            ("Last Activity", _fmt_ts(summary.get("last_activity"))),
            ("Detections", str(summary.get("detection_count", 0))),
            ("Corrections", str(summary.get("correction_count", 0))),
            ("Groups", str(summary.get("group_count", 0))),
            ("Runs", ", ".join(summary.get("run_ids", [])[:5]) or "—"),
        ]
        for i, (label, val) in enumerate(pairs):
            ttk.Label(
                sec.content,
                text=f"{label}:",
                font=("TkDefaultFont", 9, "bold"),
            ).grid(row=i, column=0, sticky="nw", padx=(4, 8), pady=1)
            ttk.Label(sec.content, text=str(val), wraplength=400).grid(
                row=i, column=1, sticky="w", pady=1
            )

        # ── Detections per Page ──────────────────────────────────────
        det_counts = self._store.get_detection_counts_by_page(
            self._selected_doc_id, self._selected_run_id
        )
        if det_counts:
            det_sec = CollapsibleFrame(
                self._detail_inner, "Detections by Page", initially_open=True
            )
            det_sec.grid(row=row, column=0, sticky="ew", pady=2, padx=4)
            row += 1

            # Group by page
            pages: dict[int, dict[str, int]] = {}
            for r in det_counts:
                pg = r["page"]
                pages.setdefault(pg, {})[r["element_type"]] = r["count"]

            prow = 0
            for pg in sorted(pages):
                types = pages[pg]
                total = sum(types.values())
                type_summary = ", ".join(f"{t}: {c}" for t, c in sorted(types.items()))
                pg_frame = CollapsibleFrame(
                    det_sec.content,
                    f"Page {pg}  ({total} detections)",
                )
                pg_frame.grid(row=prow, column=0, sticky="ew", pady=1)
                prow += 1
                for ti, (t, c) in enumerate(sorted(types.items())):
                    ttk.Label(
                        pg_frame.content,
                        text=f"{t}:",
                        font=("TkDefaultFont", 9, "bold"),
                    ).grid(row=ti, column=0, sticky="w", padx=(4, 8), pady=1)
                    ttk.Label(pg_frame.content, text=str(c)).grid(
                        row=ti, column=1, sticky="w", pady=1
                    )

        # ── Correction Breakdown (doc-scoped) ────────────────────────
        breakdown = self._store.get_correction_type_breakdown(self._selected_doc_id)
        if breakdown:
            cb_sec = CollapsibleFrame(
                self._detail_inner,
                f"Corrections ({sum(breakdown.values())})",
                initially_open=True,
            )
            cb_sec.grid(row=row, column=0, sticky="ew", pady=2, padx=4)
            row += 1
            for i, (ctype, cnt) in enumerate(sorted(breakdown.items())):
                type_colors = {
                    "accept": "#98c379",
                    "relabel": "#61afef",
                    "reshape": "#c678dd",
                    "delete": "#e06c75",
                    "add": "#e5c07b",
                }
                fg = type_colors.get(ctype, None)
                ttk.Label(
                    cb_sec.content,
                    text=f"{ctype}:",
                    font=("TkDefaultFont", 9, "bold"),
                ).grid(row=i, column=0, sticky="w", padx=(4, 8), pady=1)
                kw = {"foreground": fg} if fg else {}
                ttk.Label(cb_sec.content, text=str(cnt), **kw).grid(
                    row=i, column=1, sticky="w", pady=1
                )

        # ── Groups ───────────────────────────────────────────────────
        page_count = summary.get("page_count", 0)
        all_groups: list[dict] = []
        for pg in range(page_count):
            groups = self._store.get_groups_for_page(self._selected_doc_id, pg)
            all_groups.extend(groups)
        if all_groups:
            grp_sec = CollapsibleFrame(
                self._detail_inner,
                f"Groups ({len(all_groups)})",
            )
            grp_sec.grid(row=row, column=0, sticky="ew", pady=2, padx=4)
            row += 1
            for ci, hdr in enumerate(["Group ID", "Label", "Page", "Members"]):
                ttk.Label(
                    grp_sec.content,
                    text=hdr,
                    font=("TkDefaultFont", 8, "bold"),
                    foreground="gray",
                ).grid(row=0, column=ci, sticky="w", padx=4)
            for ri, g in enumerate(all_groups[:30], start=1):
                vals = [
                    g.get("group_id", "?")[:12],
                    g.get("group_label", "?"),
                    str(g.get("page", "")),
                    str(len(g.get("members", []))),
                ]
                for ci, v in enumerate(vals):
                    ttk.Label(grp_sec.content, text=v, font=("TkDefaultFont", 8)).grid(
                        row=ri, column=ci, sticky="w", padx=4, pady=1
                    )

        # ── Recent Corrections (doc-scoped) ──────────────────────────
        recent = self._store.get_recent_corrections(self._selected_doc_id, limit=20)
        if recent:
            rc_sec = CollapsibleFrame(
                self._detail_inner,
                f"Recent Corrections (last {len(recent)})",
            )
            rc_sec.grid(row=row, column=0, sticky="ew", pady=2, padx=4)
            row += 1
            for ci, hdr in enumerate(["Time", "Page", "Type", "Label"]):
                ttk.Label(
                    rc_sec.content,
                    text=hdr,
                    font=("TkDefaultFont", 8, "bold"),
                    foreground="gray",
                ).grid(row=0, column=ci, sticky="w", padx=4)
            for ri, c in enumerate(recent, start=1):
                ts = _fmt_ts(c.get("corrected_at"))
                vals = [
                    ts,
                    str(c.get("page", "")),
                    c.get("correction_type", ""),
                    f"{c.get('original_label', '?')} → {c.get('corrected_label', '?')}",
                ]
                type_colors = {
                    "accept": "#98c379",
                    "relabel": "#61afef",
                    "reshape": "#c678dd",
                    "delete": "#e06c75",
                    "add": "#e5c07b",
                }
                fg = type_colors.get(c.get("correction_type", ""), None)
                for ci, v in enumerate(vals):
                    kw = {"foreground": fg} if fg and ci == 2 else {}
                    ttk.Label(
                        rc_sec.content, text=v, font=("TkDefaultFont", 8), **kw
                    ).grid(row=ri, column=ci, sticky="w", padx=4, pady=1)

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _on_snapshot(self) -> None:
        """Create a DB snapshot."""
        try:
            path = self._store.snapshot(tag="manual")
            messagebox.showinfo("Snapshot", f"Saved to:\n{path}")
            self._refresh()
        except Exception as exc:
            messagebox.showerror("Snapshot Error", str(exc))

    def _on_restore(self) -> None:
        """Restore from a snapshot."""
        snaps = self._store.list_snapshots()
        if not snaps:
            messagebox.showinfo("Restore", "No snapshots available.")
            return
        # Pick the most recent
        choices = [
            f"{_fmt_ts(s.get('created', ''))}  {s.get('tag', '')}  ({_fmt_bytes(s.get('size', 0))})"
            for s in snaps
        ]
        # Simple dialog
        from tkinter import simpledialog

        idx_str = simpledialog.askstring(
            "Restore Snapshot",
            "Enter snapshot number (1 = newest):\n\n"
            + "\n".join(f"  {i + 1}. {c}" for i, c in enumerate(choices[:10])),
            parent=self.root,
        )
        if not idx_str:
            return
        try:
            idx = int(idx_str.strip()) - 1
            snap = snaps[idx]
        except (ValueError, IndexError):
            messagebox.showerror("Restore", "Invalid selection.")
            return
        if not messagebox.askyesno(
            "Confirm Restore",
            f"This will overwrite the current database with:\n{snap.get('path', '?')}\n\nContinue?",
        ):
            return
        try:
            from pathlib import Path as _P

            self._store.restore_snapshot(_P(snap["path"]))
            messagebox.showinfo("Restore", "Database restored successfully.")
            self._refresh()
        except Exception as exc:
            messagebox.showerror("Restore Error", str(exc))
