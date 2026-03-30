"""External/optional dependency diagnostic sections.

Covers: LayoutLMv3 layout detection, sentence-transformer embeddings,
LLM semantic checks, and cross-page GNN availability.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk
from typing import Any

from ..widgets import CollapsibleFrame, LogPanel
from ..worker import PipelineWorker


# ---------------------------------------------------------------------------
# Section 5 – Layout Model (LayoutLMv3)
# ---------------------------------------------------------------------------


class LayoutModelSection(CollapsibleFrame):
    """Collapsible section for LayoutLMv3 layout detection."""

    def __init__(
        self,
        parent: tk.Widget,
        log_panel: LogPanel,
        state: Any,
        root: tk.Tk,
        **kwargs: Any,
    ) -> None:
        super().__init__(parent, "Layout Model (LayoutLMv3)", **kwargs)
        self._log = log_panel
        self._state = state
        self._root = root
        self._worker: PipelineWorker | None = None
        self._build()

    def _build(self) -> None:
        lm = self.content
        lm.columnconfigure(1, weight=1)

        ttk.Label(
            lm,
            text="Run LayoutLMv3 layout detection on the current page with a fine-tuned checkpoint.",
            foreground="gray",
        ).grid(row=0, column=0, columnspan=3, sticky="w", pady=2)

        ttk.Label(lm, text="Model:").grid(row=1, column=0, sticky="w", pady=2)
        self._layout_model_var = tk.StringVar(value=self._initial_layout_model_value())
        ttk.Entry(lm, textvariable=self._layout_model_var, width=50).grid(
            row=1, column=1, sticky="ew", padx=4
        )

        ttk.Label(
            lm,
            text="Leave blank until you have a fine-tuned checkpoint; the base model is not usable for layout inference.",
            foreground="gray",
        ).grid(row=2, column=0, columnspan=3, sticky="w", pady=2)

        layout_btns = ttk.Frame(lm)
        layout_btns.grid(row=3, column=0, columnspan=2, sticky="w", pady=4)
        ttk.Button(
            layout_btns, text="Run Layout Detection", command=self._run_layout_detection
        ).pack(side="left", padx=2)
        ttk.Button(
            layout_btns, text="Check Availability", command=self._check_layout_avail
        ).pack(side="left", padx=2)

    def _initial_layout_model_value(self) -> str:
        """Prefer a configured fine-tuned layout model; otherwise start blank."""
        configured = getattr(self._state.config, "ml_layout_model_path", "").strip()
        if configured == "microsoft/layoutlmv3-base":
            return ""
        return configured

    def _is_invalid_layout_model(self, model_name: str) -> bool:
        """Return True when the selected model is the unfine-tuned base checkpoint."""
        return model_name.strip() == "microsoft/layoutlmv3-base"

    def _check_layout_avail(self) -> None:
        self._log.clear()
        try:
            from plancheck.analysis.layout_model import is_layout_available

            avail = is_layout_available()
            if avail:
                model_name = self._layout_model_var.get().strip()
                if self._is_invalid_layout_model(model_name) or not model_name:
                    self._log.write(
                        "LayoutLMv3 dependencies are available, but you still need a fine-tuned checkpoint path.",
                        "WARNING",
                    )
                else:
                    self._log.write(
                        "LayoutLMv3 is available and a checkpoint path is configured.",
                        "SUCCESS",
                    )
            else:
                self._log.write(
                    "LayoutLMv3 NOT available. Install with: pip install 'plancheck[layout]'",
                    "WARNING",
                )
        except Exception as exc:
            self._log.write(f"Error checking availability: {exc}", "ERROR")

    def _run_layout_detection(self) -> None:
        pdf = self._state.pdf_path
        if pdf is None:
            messagebox.showwarning("No PDF", "Select a PDF in the Pipeline tab first.")
            return
        model_name = self._layout_model_var.get().strip()
        if not model_name:
            messagebox.showwarning(
                "No Model",
                "Enter a fine-tuned LayoutLMv3 model name or checkpoint path.",
            )
            return
        if self._is_invalid_layout_model(model_name):
            messagebox.showwarning(
                "Invalid Model",
                "The LayoutLMv3 base checkpoint has a random classification head. Use a fine-tuned checkpoint instead.",
            )
            return

        self._log.clear()
        self._worker = PipelineWorker(self._root, self._log)

        def target():
            from plancheck import GlyphBox, GroupingConfig, extract_tokens
            from plancheck.analysis.layout_model import (
                is_layout_available,
                predict_layout,
            )
            from plancheck.ingest import render_page_image

            if not is_layout_available():
                raise RuntimeError(
                    "LayoutLMv3 not available. "
                    "Install with: pip install 'plancheck[layout]'"
                )

            print(f"Loading page from {pdf}...")
            cfg = GroupingConfig()
            tokens, pw, ph = extract_tokens(str(pdf), 0, cfg)
            image = render_page_image(pdf, 0, resolution=150)

            print(f"Running layout detection ({model_name})...")
            preds = predict_layout(image, tokens, pw, ph, model_name_or_path=model_name)

            print(f"\nLayout predictions: {len(preds)}")
            for p in preds:
                bbox = p.bbox
                print(
                    f"  {p.label:20s} conf={p.confidence:.3f} "
                    f"bbox=({bbox[0]:.0f},{bbox[1]:.0f},{bbox[2]:.0f},{bbox[3]:.0f}) "
                    f"tokens={len(p.token_indices)}"
                )
            return preds

        def on_done(result, error, elapsed):
            if not error:
                n = len(result) if result else 0
                self._log.write(
                    f"Layout detection complete: {n} regions ({elapsed:.1f}s).",
                    "SUCCESS",
                )

        self._worker.run(target, on_done=on_done)


# ---------------------------------------------------------------------------
# Section 6 – Text Embeddings (Sentence-Transformer)
# ---------------------------------------------------------------------------


class TextEmbeddingsSection(CollapsibleFrame):
    """Collapsible section for sentence-transformer embedding tools."""

    def __init__(
        self,
        parent: tk.Widget,
        log_panel: LogPanel,
        state: Any,
        root: tk.Tk,
        **kwargs: Any,
    ) -> None:
        super().__init__(parent, "Text Embeddings (Sentence-Transformer)", **kwargs)
        self._log = log_panel
        self._state = state
        self._root = root
        self._worker: PipelineWorker | None = None
        self._build()

    def _build(self) -> None:
        em = self.content
        em.columnconfigure(1, weight=1)

        ttk.Label(
            em,
            text="Dense semantic embeddings for block text (supplements kw_* features).",
            foreground="gray",
        ).grid(row=0, column=0, columnspan=3, sticky="w", pady=2)

        ttk.Label(em, text="Model:").grid(row=1, column=0, sticky="w", pady=2)
        self._emb_model_var = tk.StringVar(value="all-MiniLM-L6-v2")
        ttk.Entry(em, textvariable=self._emb_model_var, width=50).grid(
            row=1, column=1, sticky="ew", padx=4
        )

        emb_btns = ttk.Frame(em)
        emb_btns.grid(row=2, column=0, columnspan=2, sticky="w", pady=4)
        ttk.Button(
            emb_btns, text="Check Availability", command=self._check_embeddings_avail
        ).pack(side="left", padx=2)
        ttk.Button(emb_btns, text="Test Embedding", command=self._test_embedding).pack(
            side="left", padx=2
        )

    def _check_embeddings_avail(self) -> None:
        self._log.clear()
        try:
            from plancheck.corrections.text_embeddings import is_embeddings_available

            avail = is_embeddings_available()
            if avail:
                self._log.write("sentence-transformers is available.", "SUCCESS")
            else:
                self._log.write(
                    "sentence-transformers NOT available. Install with: "
                    "pip install 'plancheck[embeddings]'",
                    "WARNING",
                )
        except Exception as exc:
            self._log.write(f"Error checking availability: {exc}", "ERROR")

    def _test_embedding(self) -> None:
        self._log.clear()
        self._worker = PipelineWorker(self._root, self._log)
        model_name = self._emb_model_var.get().strip() or "all-MiniLM-L6-v2"

        def target():
            from plancheck.corrections.text_embeddings import (
                TextEmbedder,
                is_embeddings_available,
            )

            if not is_embeddings_available():
                raise RuntimeError(
                    "sentence-transformers not available. "
                    "Install with: pip install 'plancheck[embeddings]'"
                )

            embedder = TextEmbedder(model_name=model_name)
            test_texts = [
                "GENERAL NOTES",
                "CONSTRUCTION NOTES",
                "LEGEND",
                "ABBREVIATIONS",
                "REVISION SCHEDULE",
            ]
            embeddings = embedder.embed_batch(test_texts)
            print(f"Model: {model_name}")
            print(f"Embedding dim: {embedder.embedding_dim}")
            print(f"\nSemantic similarity test:")
            import numpy as np

            for i in range(len(test_texts)):
                for j in range(i + 1, len(test_texts)):
                    sim = float(np.dot(embeddings[i], embeddings[j]))
                    print(f"  '{test_texts[i]}' vs '{test_texts[j]}': {sim:.3f}")
            return embeddings

        def on_done(result, error, elapsed):
            if not error:
                self._log.write(f"Embedding test complete ({elapsed:.1f}s).", "SUCCESS")

        self._worker.run(target, on_done=on_done)


# ---------------------------------------------------------------------------
# Section 7 – LLM Semantic Checks
# ---------------------------------------------------------------------------


class LLMSemanticChecksSection(CollapsibleFrame):
    """Collapsible section for optional LLM-assisted content analysis."""

    def __init__(
        self,
        parent: tk.Widget,
        log_panel: LogPanel,
        state: Any,
        root: tk.Tk,
        **kwargs: Any,
    ) -> None:
        super().__init__(parent, "LLM Semantic Checks", **kwargs)
        self._log = log_panel
        self._state = state
        self._root = root
        self._worker: PipelineWorker | None = None
        self._build()

    def _build(self) -> None:
        lc = self.content
        lc.columnconfigure(1, weight=1)

        ttk.Label(
            lc,
            text="Optional LLM-assisted content analysis (off by default).",
            foreground="gray",
        ).grid(row=0, column=0, columnspan=3, sticky="w", pady=2)

        ttk.Label(lc, text="Provider:").grid(row=1, column=0, sticky="w", pady=2)
        self._llm_provider_var = tk.StringVar(value="ollama")
        ttk.Combobox(
            lc,
            textvariable=self._llm_provider_var,
            width=20,
            values=["ollama", "openai", "anthropic"],
            state="readonly",
        ).grid(row=1, column=1, sticky="w", padx=4)

        ttk.Label(lc, text="Model:").grid(row=2, column=0, sticky="w", pady=2)
        self._llm_model_var = tk.StringVar(value="llama3.1:8b")
        ttk.Entry(lc, textvariable=self._llm_model_var, width=50).grid(
            row=2, column=1, sticky="ew", padx=4
        )

        llm_btns = ttk.Frame(lc)
        llm_btns.grid(row=3, column=0, columnspan=2, sticky="w", pady=4)
        ttk.Button(
            llm_btns, text="Check Availability", command=self._check_llm_avail
        ).pack(side="left", padx=2)
        ttk.Button(llm_btns, text="Run LLM Checks", command=self._run_llm_checks).pack(
            side="left", padx=2
        )

    def _check_llm_avail(self) -> None:
        self._log.clear()
        provider = self._llm_provider_var.get()
        try:
            from plancheck.checks.llm_checks import is_llm_available

            avail = is_llm_available(provider)
            if avail:
                self._log.write(f"LLM provider '{provider}' is available.", "SUCCESS")
            else:
                self._log.write(
                    f"LLM provider '{provider}' NOT available. "
                    f"Install with: pip install 'plancheck[llm]'",
                    "WARNING",
                )
        except Exception as exc:
            self._log.write(f"Error checking availability: {exc}", "ERROR")

    def _run_llm_checks(self) -> None:
        pdf = self._state.pdf_path
        if pdf is None:
            messagebox.showwarning("No PDF", "Select a PDF in the Pipeline tab first.")
            return

        provider = self._llm_provider_var.get()
        model = self._llm_model_var.get().strip()
        if not model:
            messagebox.showwarning("No Model", "Enter an LLM model name.")
            return

        self._log.clear()
        self._worker = PipelineWorker(self._root, self._log)

        def target():
            from plancheck import GroupingConfig
            from plancheck.checks.llm_checks import is_llm_available, run_llm_checks
            from plancheck.pipeline import run_pipeline

            if not is_llm_available(provider):
                raise RuntimeError(
                    f"LLM provider '{provider}' not available. "
                    f"Install with: pip install 'plancheck[llm]'"
                )

            print(f"Running pipeline on page 0...")
            cfg = GroupingConfig()
            pr = run_pipeline(pdf, 0, cfg=cfg)

            print(f"Running LLM checks ({provider}/{model})...")
            findings = run_llm_checks(
                notes_columns=pr.notes_columns,
                provider=provider,
                model=model,
            )
            print(f"\nLLM findings: {len(findings)}")
            for f in findings:
                print(f"  [{f.severity}] {f.check_id}: {f.message}")
            return findings

        def on_done(result, error, elapsed):
            if not error:
                n = len(result) if result else 0
                self._log.write(
                    f"LLM checks complete: {n} findings ({elapsed:.1f}s).", "SUCCESS"
                )

        self._worker.run(target, on_done=on_done)


# ---------------------------------------------------------------------------
# Section 8 – Cross-Page GNN
# ---------------------------------------------------------------------------


class CrossPageGNNSection(CollapsibleFrame):
    """Collapsible section for cross-page GNN inconsistency detection."""

    def __init__(
        self,
        parent: tk.Widget,
        log_panel: LogPanel,
        state: Any,
        root: tk.Tk,
        **kwargs: Any,
    ) -> None:
        super().__init__(parent, "Cross-Page GNN", **kwargs)
        self._log = log_panel
        self._state = state
        self._root = root
        self._build()

    def _build(self) -> None:
        gn = self.content
        gn.columnconfigure(1, weight=1)

        ttk.Label(
            gn,
            text="Graph neural network for cross-page inconsistency detection.",
            foreground="gray",
        ).grid(row=0, column=0, columnspan=3, sticky="w", pady=2)

        ttk.Label(gn, text="Model:").grid(row=1, column=0, sticky="w", pady=2)
        self._gnn_model_var = tk.StringVar(value="data/document_gnn.pt")
        ttk.Entry(gn, textvariable=self._gnn_model_var, width=50).grid(
            row=1, column=1, sticky="ew", padx=4
        )

        gnn_btns = ttk.Frame(gn)
        gnn_btns.grid(row=2, column=0, columnspan=2, sticky="w", pady=4)
        ttk.Button(
            gnn_btns, text="Check Availability", command=self._check_gnn_avail
        ).pack(side="left", padx=2)

    def _check_gnn_avail(self) -> None:
        self._log.clear()
        try:
            from plancheck.analysis.gnn import is_gnn_available

            avail = is_gnn_available()
            if avail:
                self._log.write(
                    "PyTorch Geometric is available (torch + torch_geometric).",
                    "SUCCESS",
                )
            else:
                self._log.write(
                    "PyTorch Geometric NOT available. Install with: "
                    "pip install 'plancheck[gnn]'",
                    "WARNING",
                )
        except Exception as exc:
            self._log.write(f"Error checking availability: {exc}", "ERROR")
