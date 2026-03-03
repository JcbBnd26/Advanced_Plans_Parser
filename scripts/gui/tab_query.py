"""Tab – Query: Chat-style interface for querying plan documents.

Features:
- Load from last run or select a run directory
- Chat history with scrolling transcript
- Semantic search-only mode
- Page/region filters
- Export chat history to Markdown
- Cost and status indicators
"""

from __future__ import annotations

import json
import threading
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any


class QueryTab:
    """Chat-based query interface for indexed plan documents."""

    def __init__(self, notebook: ttk.Notebook, gui_state: Any) -> None:
        self.notebook = notebook
        self.state = gui_state
        self.root = notebook.winfo_toplevel()

        self.frame = ttk.Frame(notebook)
        self.frame.columnconfigure(0, weight=1)
        self.frame.rowconfigure(1, weight=1)
        notebook.add(self.frame, text="Query")

        self._engine = None
        self._chat_history: list[dict] = []
        self._busy = False

        # Shutdown safety: avoid scheduling after() onto a destroyed root
        self._closing: bool = False

        # Cooperative cancellation for background work
        self._cancel_event = threading.Event()
        self._load_gen: int = 0
        self._query_gen: int = 0

        # Default LLM settings
        self._provider = tk.StringVar(value="ollama")
        self._model = tk.StringVar(value="llama3.1:8b")
        self._page_filter = tk.StringVar(value="all")

        self._build_ui()

        # When the window is destroyed, mark as closing so background
        # threads stop scheduling UI updates.
        try:
            self.root.bind("<Destroy>", lambda e: setattr(self, "_closing", True), add="+")
        except Exception:
            pass

        # React to run completions
        self.state.subscribe("run_completed", self._on_run_completed)

    def request_cancel(self) -> None:
        """Best-effort cancel of background tasks (load/query)."""
        self._cancel_event.set()
        # Bump generations so any in-flight results are ignored
        self._load_gen += 1
        self._query_gen += 1

    # ------------------------------------------------------------------
    # Build UI
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        pad = {"padx": 6, "pady": 4}

        # ── Toolbar row ──────────────────────────────────────────────
        toolbar = ttk.Frame(self.frame)
        toolbar.grid(row=0, column=0, sticky="ew", **pad)
        toolbar.columnconfigure(4, weight=1)

        ttk.Button(toolbar, text="Load Run", command=self._load_run).grid(
            row=0, column=0, **pad
        )
        ttk.Button(toolbar, text="Load Last Run", command=self._load_last_run).grid(
            row=0, column=1, **pad
        )

        ttk.Label(toolbar, text="Provider:").grid(row=0, column=2, padx=(12, 2))
        prov_combo = ttk.Combobox(
            toolbar,
            textvariable=self._provider,
            values=["ollama", "openai", "anthropic"],
            width=10,
            state="readonly",
        )
        prov_combo.grid(row=0, column=3, **pad)

        ttk.Label(toolbar, text="Model:").grid(row=0, column=4, padx=(8, 2), sticky="e")
        ttk.Entry(toolbar, textvariable=self._model, width=18).grid(
            row=0, column=5, **pad
        )

        ttk.Label(toolbar, text="Page:").grid(row=0, column=6, padx=(8, 2))
        ttk.Entry(toolbar, textvariable=self._page_filter, width=5).grid(
            row=0, column=7, **pad
        )

        self._status_label = ttk.Label(toolbar, text="No run loaded", foreground="gray")
        self._status_label.grid(row=0, column=8, padx=(12, 4), sticky="e")

        # ── Chat transcript ──────────────────────────────────────────
        chat_frame = ttk.Frame(self.frame)
        chat_frame.grid(row=1, column=0, sticky="nsew", **pad)
        chat_frame.columnconfigure(0, weight=1)
        chat_frame.rowconfigure(0, weight=1)

        self._chat_text = tk.Text(
            chat_frame,
            wrap="word",
            state="disabled",
            font=("Consolas", 10),
            bg="#1e1e1e",
            fg="#d4d4d4",
            insertbackground="#d4d4d4",
            selectbackground="#264f78",
            relief="flat",
            padx=8,
            pady=8,
        )
        self._chat_text.grid(row=0, column=0, sticky="nsew")

        scrollbar = ttk.Scrollbar(chat_frame, command=self._chat_text.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self._chat_text.configure(yscrollcommand=scrollbar.set)

        # Tag styles for chat
        self._chat_text.tag_configure(
            "user", foreground="#569cd6", font=("Consolas", 10, "bold")
        )
        self._chat_text.tag_configure("assistant", foreground="#d4d4d4")
        self._chat_text.tag_configure(
            "source", foreground="#6a9955", font=("Consolas", 9)
        )
        self._chat_text.tag_configure(
            "meta", foreground="#808080", font=("Consolas", 9)
        )
        self._chat_text.tag_configure(
            "system", foreground="#ce9178", font=("Consolas", 9, "italic")
        )
        self._chat_text.tag_configure("error", foreground="#f44747")

        # ── Input row ────────────────────────────────────────────────
        input_frame = ttk.Frame(self.frame)
        input_frame.grid(row=2, column=0, sticky="ew", **pad)
        input_frame.columnconfigure(0, weight=1)

        self._input_entry = ttk.Entry(input_frame, font=("Consolas", 10))
        self._input_entry.grid(row=0, column=0, sticky="ew", padx=(0, 4))
        self._input_entry.bind("<Return>", lambda e: self._on_send())

        self._send_btn = ttk.Button(input_frame, text="Send", command=self._on_send)
        self._send_btn.grid(row=0, column=1, padx=2)

        self._search_btn = ttk.Button(
            input_frame, text="Search Only", command=self._on_search
        )
        self._search_btn.grid(row=0, column=2, padx=2)

        # ── Bottom bar ───────────────────────────────────────────────
        bottom = ttk.Frame(self.frame)
        bottom.grid(row=3, column=0, sticky="ew", **pad)

        ttk.Button(bottom, text="Clear Chat", command=self._clear_chat).pack(
            side="left", padx=2
        )
        ttk.Button(bottom, text="Export Chat", command=self._export_chat).pack(
            side="left", padx=2
        )
        ttk.Button(bottom, text="Show Cost", command=self._show_cost).pack(
            side="left", padx=2
        )

        self._cost_label = ttk.Label(bottom, text="", foreground="gray")
        self._cost_label.pack(side="right", padx=8)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load_run(self) -> None:
        """Browse for a run directory."""
        runs_root = Path("runs")
        d = filedialog.askdirectory(
            title="Select Run Directory",
            initialdir=str(runs_root) if runs_root.exists() else str(Path(".")),
        )
        if d:
            self._init_engine(Path(d))

    def _load_last_run(self) -> None:
        """Load the most recent run from GuiState."""
        if self.state.last_run_dir and self.state.last_run_dir.exists():
            self._init_engine(self.state.last_run_dir)
        else:
            messagebox.showinfo(
                "Query", "No recent run available. Use 'Load Run' instead."
            )

    def _on_run_completed(self) -> None:
        """Auto-notification when a pipeline run finishes."""
        self._status_label.config(
            text=f"New run available: {self.state.last_run_dir.name if self.state.last_run_dir else ''}",
            foreground="orange",
        )

    def _init_engine(self, run_dir: Path) -> None:
        """Build the query engine from a run directory."""
        self._append_system(f"Loading run: {run_dir.name}...")
        self._set_busy(True)

        # Cancel any previous load
        self._cancel_event.clear()
        self._load_gen += 1
        my_gen = self._load_gen

        def _load():
            try:
                if self._cancel_event.is_set() or self._closing or my_gen != self._load_gen:
                    return
                from plancheck.export.run_loader import load_run
                from plancheck.llm.query_engine import DocumentQueryEngine

                dr = load_run(str(run_dir))
                if self._cancel_event.is_set() or self._closing or my_gen != self._load_gen:
                    return
                engine = DocumentQueryEngine.from_document_result(
                    dr,
                    provider=self._provider.get(),
                    model=self._model.get(),
                    policy=(
                        self.state.config.llm_policy
                        if hasattr(self.state, "config")
                        else "local_only"
                    ),
                )
                n_chunks = engine.index.count
                n_pages = len(dr.pages) if dr.pages else 0
                self._safe_after(
                    0,
                    lambda: self._on_engine_ready(engine, n_pages, n_chunks, run_dir)
                    if (not self._cancel_event.is_set() and not self._closing and my_gen == self._load_gen)
                    else None,
                )
            except Exception as exc:
                self._safe_after(
                    0,
                    lambda: self._on_engine_error(exc)
                    if (not self._cancel_event.is_set() and not self._closing and my_gen == self._load_gen)
                    else None,
                )

        threading.Thread(target=_load, daemon=True).start()

    def _on_engine_ready(
        self, engine, n_pages: int, n_chunks: int, run_dir: Path
    ) -> None:
        self._engine = engine
        self._set_busy(False)
        self._status_label.config(
            text=f"{run_dir.name} — {n_pages} page(s), {n_chunks} chunks",
            foreground="green",
        )
        self._append_system(
            f"Ready. Indexed {n_chunks} chunks from {n_pages} page(s). Ask a question!"
        )

    def _on_engine_error(self, exc: Exception) -> None:
        self._set_busy(False)
        self._status_label.config(text="Load failed", foreground="red")
        self._append_error(f"Failed to load run: {exc}")

    # ------------------------------------------------------------------
    # Chat interactions
    # ------------------------------------------------------------------

    def _on_send(self) -> None:
        """Send a query to the engine."""
        question = self._input_entry.get().strip()
        if not question:
            return
        if not self._engine:
            self._append_system("No run loaded. Use 'Load Run' first.")
            return
        self._input_entry.delete(0, tk.END)
        self._append_user(question)
        self._set_busy(True)

        # Cancel any previous query and start a new generation
        self._cancel_event.clear()
        self._query_gen += 1
        my_gen = self._query_gen

        page_filt = self._parse_page_filter()

        def _query():
            try:
                if self._cancel_event.is_set() or self._closing or my_gen != self._query_gen:
                    return
                result = self._engine.query(question, page_filter=page_filt)
                if self._cancel_event.is_set() or self._closing or my_gen != self._query_gen:
                    return
                self._safe_after(
                    0,
                    lambda: self._on_answer(question, result)
                    if (not self._cancel_event.is_set() and not self._closing and my_gen == self._query_gen)
                    else None,
                )
            except Exception as exc:
                if self._cancel_event.is_set() or self._closing or my_gen != self._query_gen:
                    return
                self._safe_after(
                    0,
                    lambda: self._on_query_error(exc)
                    if (not self._cancel_event.is_set() and not self._closing and my_gen == self._query_gen)
                    else None,
                )

        threading.Thread(target=_query, daemon=True).start()

    def _safe_after(self, delay_ms: int, callback) -> None:
        """Schedule callback on the UI thread if the window still exists."""
        if self._closing:
            return
        try:
            if hasattr(self.root, "winfo_exists") and not self.root.winfo_exists():
                return
        except Exception:
            pass
        if callback is None:
            return
        try:
            self.root.after(delay_ms, callback)
        except Exception:
            pass

    def _on_search(self) -> None:
        """Semantic search without LLM."""
        question = self._input_entry.get().strip()
        if not question:
            return
        if not self._engine:
            self._append_system("No run loaded. Use 'Load Run' first.")
            return
        self._input_entry.delete(0, tk.END)
        self._append_user(f"[search] {question}")

        page_filt = self._parse_page_filter()
        results = self._engine.search_only(question, n_results=5, page_filter=page_filt)

        if not results:
            self._append_system("No results found.")
            return

        lines = []
        for r in results:
            c = r.chunk
            lines.append(
                f"  [{r.rank}] score={r.score:.3f}  page={c.page}  type={c.region_type}"
            )
            lines.append(f"      {c.text[:150]}")
        self._append_assistant("\n".join(lines))
        self._chat_history.append(
            {"role": "search", "query": question, "results": len(results)}
        )

    def _on_answer(self, question: str, result) -> None:
        self._set_busy(False)

        # Main answer
        self._append_assistant(result.text)

        # Sources
        if result.sources:
            src_lines = []
            for s in result.sources[:5]:
                page = s.get("page", "?")
                rtype = s.get("region_type", "?")
                excerpt = s.get("excerpt", "")[:100]
                src_lines.append(f"  Page {page} ({rtype}): {excerpt}")
            self._append_sources("\n".join(src_lines))

        # Meta
        if result.meta:
            m = result.meta
            tag = f"[{m.provider}/{m.model} — {m.latency_s:.1f}s, ~{m.input_tokens}+{m.output_tokens} tok]"
            if result.cached:
                tag = "[cached] " + tag
            self._append_meta(tag)

        # Track
        self._chat_history.append(
            {
                "role": "qa",
                "question": question,
                "answer": result.text,
                "sources": result.sources,
                "cached": result.cached,
            }
        )

        self._update_cost()

    def _on_query_error(self, exc: Exception) -> None:
        self._set_busy(False)
        self._append_error(f"Query error: {exc}")

    # ------------------------------------------------------------------
    # Chat display helpers
    # ------------------------------------------------------------------

    def _append_text(self, text: str, tag: str) -> None:
        self._chat_text.configure(state="normal")
        self._chat_text.insert(tk.END, text + "\n\n", tag)
        self._chat_text.configure(state="disabled")
        self._chat_text.see(tk.END)

    def _append_user(self, text: str) -> None:
        self._append_text(f"You: {text}", "user")

    def _append_assistant(self, text: str) -> None:
        self._append_text(text, "assistant")

    def _append_sources(self, text: str) -> None:
        self._append_text(f"Sources:\n{text}", "source")

    def _append_meta(self, text: str) -> None:
        self._append_text(text, "meta")

    def _append_system(self, text: str) -> None:
        self._append_text(f"[System] {text}", "system")

    def _append_error(self, text: str) -> None:
        self._append_text(f"[Error] {text}", "error")

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _clear_chat(self) -> None:
        self._chat_text.configure(state="normal")
        self._chat_text.delete("1.0", tk.END)
        self._chat_text.configure(state="disabled")
        self._chat_history.clear()
        if self._engine:
            self._engine.clear_cache()

    def _export_chat(self) -> None:
        if not self._chat_history:
            messagebox.showinfo("Query", "Nothing to export.")
            return
        path = filedialog.asksaveasfilename(
            title="Export Chat",
            defaultextension=".md",
            filetypes=[("Markdown", "*.md"), ("JSON", "*.json")],
        )
        if not path:
            return

        p = Path(path)
        if p.suffix == ".json":
            p.write_text(json.dumps(self._chat_history, indent=2), encoding="utf-8")
        else:
            lines = [f"# Query Session — {datetime.now().isoformat()}", ""]
            for entry in self._chat_history:
                if entry["role"] == "qa":
                    lines.append(f"**Q:** {entry['question']}\n")
                    lines.append(f"**A:** {entry['answer']}\n")
                    if entry.get("sources"):
                        lines.append("**Sources:**")
                        for s in entry["sources"]:
                            lines.append(
                                f"- Page {s.get('page', '?')} ({s.get('region_type', '?')})"
                            )
                    lines.append("---\n")
                elif entry["role"] == "search":
                    lines.append(
                        f"**Search:** {entry['query']} ({entry['results']} results)\n"
                    )
            p.write_text("\n".join(lines), encoding="utf-8")

        self._append_system(f"Chat exported to {p.name}")

    def _show_cost(self) -> None:
        if not self._engine:
            messagebox.showinfo("Cost", "No engine loaded.")
            return
        s = self._engine.cost_summary
        msg = (
            f"Calls: {s['call_count']}\n"
            f"Input tokens: {s['total_input_tokens']:,}\n"
            f"Output tokens: {s['total_output_tokens']:,}\n"
            f"Est. cost: ${s['total_cost_usd']:.4f}"
        )
        messagebox.showinfo("Cost Summary", msg)

    def _update_cost(self) -> None:
        if self._engine:
            s = self._engine.cost_summary
            self._cost_label.config(
                text=f"Calls: {s['call_count']} | ~${s['total_cost_usd']:.4f}"
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _parse_page_filter(self) -> int | None:
        raw = self._page_filter.get().strip().lower()
        if raw in ("all", "", "none"):
            return None
        try:
            return int(raw)
        except ValueError:
            return None

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        state = "disabled" if busy else "normal"
        self._send_btn.config(state=state)
        self._search_btn.config(state=state)
        self._input_entry.config(state=state)
        if busy:
            self._status_label.config(text="Working...", foreground="orange")
