"""Tab 1 – Pipeline: PDF input, OCR stages, config, run button, embedded console.

Preserves all original GUI functionality (PDF selection, page range, OCR
toggles, tag management) and adds:
- Collapsible advanced-config section for Grouping & Geometry
- Config file Load / Save buttons (YAML / TOML)
- Embedded log console with stage-progress bar (replaces PowerShell windows)
- Cancel button for in-progress runs

Note: TOCR / VOCRPP / VOCR / Reconcile advanced knobs are intentionally
omitted from the GUI – they will be managed by the LLM layer in a future
release.
"""

from __future__ import annotations

import json
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any

from plancheck.config import GroupingConfig

from .widgets import CollapsibleFrame, ErrorPanel, LogPanel, StageProgressBar
from .worker import PipelineWorker

# ---------------------------------------------------------------------------
# Note: All advanced field lists (TOCR / VOCRPP / VOCR / Reconcile / Geometry)
# have been removed from the GUI – those knobs will be managed by the LLM
# layer in a future release.
# ---------------------------------------------------------------------------


class PipelineTab:
    """Tab 1: Pipeline configuration and execution."""

    def __init__(self, notebook: ttk.Notebook, gui_state: Any) -> None:
        self.notebook = notebook
        self.state = gui_state
        self.root = notebook.winfo_toplevel()

        self.frame = ttk.Frame(notebook)
        self.frame.columnconfigure(0, weight=1)
        self.frame.rowconfigure(0, weight=3)  # top scrollable area
        self.frame.rowconfigure(1, weight=0)  # stage bar
        self.frame.rowconfigure(2, weight=1)  # log panel
        notebook.add(self.frame, text="Pipeline")

        # PDF state
        self.pdf_files: list[Path] = []

        # Worker
        self._worker: PipelineWorker | None = None

        # Mousewheel scroll state (avoid unbind_all, which breaks other tabs)
        self._wheel_active: bool = False

        self._build_ui()

        # Subscribe to load_config event (fired by Runs tab)
        self.state.subscribe("load_config", self._on_load_config)

    # ------------------------------------------------------------------
    # Build UI
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        pad = {"padx": 10, "pady": 4}
        defaults = GroupingConfig.from_dict(self.state.config.to_dict())

        # ── Scrollable top area ──────────────────────────────────────
        self._canvas = tk.Canvas(self.frame, highlightthickness=0)
        self._scrollbar = ttk.Scrollbar(
            self.frame, orient="vertical", command=self._canvas.yview
        )
        self._inner = ttk.Frame(self._canvas)
        self._inner.columnconfigure(0, weight=1)
        self._inner.bind(
            "<Configure>",
            lambda e: self._canvas.configure(scrollregion=self._canvas.bbox("all")),
        )
        self._canvas_window = self._canvas.create_window(
            (0, 0), window=self._inner, anchor="nw"
        )
        self._canvas.configure(yscrollcommand=self._scrollbar.set)
        self._canvas.grid(row=0, column=0, sticky="nsew")
        self._scrollbar.grid(row=0, column=1, sticky="ns")

        def _on_canvas_configure(event):
            self._canvas.itemconfig(self._canvas_window, width=event.width)

        self._canvas.bind("<Configure>", _on_canvas_configure)
        self._canvas.bind("<Enter>", lambda e: setattr(self, "_wheel_active", True))
        self._canvas.bind("<Leave>", lambda e: setattr(self, "_wheel_active", False))

        # Bind once globally; handler is gated by _wheel_active
        self.root.bind_all("<MouseWheel>", self._on_mousewheel, add="+")

        row = 0

        # ── PDF File Selection ───────────────────────────────────────
        file_frame = ttk.LabelFrame(self._inner, text="PDF File", padding=10)
        file_frame.grid(row=row, column=0, sticky="ew", **pad)
        file_frame.columnconfigure(1, weight=1)

        ttk.Button(file_frame, text="Select File...", command=self._select_file).grid(
            row=0, column=0, padx=(0, 5)
        )
        self.file_label_var = tk.StringVar(value="No file selected")
        ttk.Label(file_frame, textvariable=self.file_label_var, foreground="gray").grid(
            row=0, column=1, sticky="w"
        )
        ttk.Button(file_frame, text="Clear", command=self._clear_file).grid(
            row=0, column=2, padx=(5, 0)
        )
        row += 1

        # ── Page Selection ───────────────────────────────────────────
        page_frame = ttk.LabelFrame(self._inner, text="Page Selection", padding=10)
        page_frame.grid(row=row, column=0, sticky="ew", **pad)
        page_frame.columnconfigure(2, weight=1)

        self.page_mode_var = tk.StringVar(value="all")

        ttk.Radiobutton(
            page_frame,
            text="All Pages",
            variable=self.page_mode_var,
            value="all",
            command=self._update_page_mode,
        ).grid(row=0, column=0, sticky="w", pady=2, columnspan=3)

        ttk.Radiobutton(
            page_frame,
            text="Single Page:",
            variable=self.page_mode_var,
            value="single",
            command=self._update_page_mode,
        ).grid(row=1, column=0, sticky="w", pady=2)
        self.single_page_var = tk.StringVar(value="1")
        self.single_page_entry = ttk.Entry(
            page_frame, textvariable=self.single_page_var, width=10
        )
        self.single_page_entry.grid(row=1, column=1, sticky="w", pady=2)
        self.single_page_hint = ttk.Label(page_frame, text="(1 = first page)")
        self.single_page_hint.grid(row=1, column=2, sticky="w", padx=(5, 0))

        ttk.Radiobutton(
            page_frame,
            text="Page Range:",
            variable=self.page_mode_var,
            value="range",
            command=self._update_page_mode,
        ).grid(row=2, column=0, sticky="w", pady=2)

        range_inner = ttk.Frame(page_frame)
        range_inner.grid(row=2, column=1, columnspan=2, sticky="w", pady=2)
        self.start_page_var = tk.StringVar(value="1")
        self.start_entry = ttk.Entry(
            range_inner, textvariable=self.start_page_var, width=6
        )
        self.start_entry.grid(row=0, column=0)
        ttk.Label(range_inner, text=" to ").grid(row=0, column=1)
        self.end_page_var = tk.StringVar(value="")
        self.end_entry = ttk.Entry(range_inner, textvariable=self.end_page_var, width=6)
        self.end_entry.grid(row=0, column=2)
        ttk.Label(range_inner, text="  (blank end = last page)").grid(row=0, column=3)

        self._update_page_mode()
        row += 1

        # ── Config File Actions ─────────────────────────────────────
        config_frame = ttk.LabelFrame(self._inner, text="Config File", padding=10)
        config_frame.grid(row=row, column=0, sticky="ew", **pad)
        config_frame.columnconfigure(1, weight=1)

        config_btns = ttk.Frame(config_frame)
        config_btns.grid(row=0, column=0, sticky="w")
        ttk.Button(
            config_btns,
            text="Load Config...",
            command=self._load_config_from_file,
        ).pack(side="left", padx=(0, 6))
        ttk.Button(
            config_btns,
            text="Save Config...",
            command=self._save_config_to_file,
        ).pack(side="left")

        self._config_path_var = tk.StringVar()
        ttk.Label(
            config_frame,
            textvariable=self._config_path_var,
            foreground="gray",
        ).grid(row=0, column=1, sticky="w", padx=(10, 0))
        self._refresh_config_file_label()

        row += 1

        # ── OCR Stage Toggles ───────────────────────────────────────
        stages_frame = ttk.LabelFrame(self._inner, text="Pipeline Stages", padding=10)
        stages_frame.grid(row=row, column=0, sticky="ew", **pad)
        stages_frame.columnconfigure(1, weight=1)

        # Get defaults from GroupingConfig (single source of truth)
        _defaults = defaults

        # TOCR
        self.tocr_var = tk.BooleanVar(value=_defaults.enable_tocr)
        ttk.Checkbutton(
            stages_frame,
            text="TOCR (pdfplumber text extraction)",
            variable=self.tocr_var,
        ).grid(row=0, column=0, sticky="w", pady=2)
        ttk.Label(
            stages_frame,
            text="Extract word boxes from the PDF text layer",
            foreground="gray",
        ).grid(row=0, column=1, sticky="w", padx=(10, 0))

        # VOCRPP
        self.ocr_preprocess_var = tk.BooleanVar(value=_defaults.enable_ocr_preprocess)
        ttk.Checkbutton(
            stages_frame,
            text="VOCRPP (Image Preprocessing)",
            variable=self.ocr_preprocess_var,
        ).grid(row=1, column=0, sticky="w", pady=2)
        ttk.Label(
            stages_frame,
            text="Grayscale, contrast, denoising for better OCR",
            foreground="gray",
        ).grid(row=1, column=1, sticky="w", padx=(10, 0))

        # VOCR
        self.vocr_var = tk.BooleanVar(value=_defaults.enable_vocr)
        ttk.Checkbutton(
            stages_frame,
            text="VOCR (Surya OCR extraction)",
            variable=self.vocr_var,
        ).grid(row=2, column=0, sticky="w", pady=2)
        ttk.Label(
            stages_frame,
            text="Full-page Surya visual token extraction",
            foreground="gray",
        ).grid(row=2, column=1, sticky="w", padx=(10, 0))

        # VOCR candidates
        self.enable_vocr_candidates_var = tk.BooleanVar(
            value=_defaults.enable_vocr_candidates
        )
        ttk.Checkbutton(
            stages_frame,
            text="VOCR Candidates (patch proposal stage)",
            variable=self.enable_vocr_candidates_var,
        ).grid(row=3, column=0, sticky="w", pady=2)
        ttk.Label(
            stages_frame,
            text="Detect likely symbol patches before full-page OCR refinement.",
            foreground="gray",
        ).grid(row=3, column=1, sticky="w", padx=(10, 0))

        # Reconcile
        self.ocr_reconcile_var = tk.BooleanVar(value=_defaults.enable_ocr_reconcile)
        ttk.Checkbutton(
            stages_frame,
            text="Reconcile (Symbol injection)",
            variable=self.ocr_reconcile_var,
        ).grid(row=4, column=0, sticky="w", pady=2)
        ttk.Label(
            stages_frame,
            text="Inject missing %, /, °, ± from VOCR into text layer",
            foreground="gray",
        ).grid(row=4, column=1, sticky="w", padx=(10, 0))

        # OCR DPI
        dpi_row = ttk.Frame(stages_frame)
        dpi_row.grid(row=5, column=0, columnspan=2, sticky="w", pady=(6, 2))
        ttk.Label(dpi_row, text="OCR/Preprocess DPI:").pack(side="left")
        self.ocr_dpi_var = tk.StringVar(value=str(_defaults.ocr_reconcile_resolution))
        ttk.Spinbox(
            dpi_row,
            textvariable=self.ocr_dpi_var,
            values=(120, 150, 180, 200, 220, 300, 400),
            width=8,
            state="readonly",
        ).pack(side="left", padx=(8, 0))
        ttk.Label(dpi_row, text="Render DPI:", foreground="gray").pack(
            side="left", padx=(20, 0)
        )
        self.resolution_var = tk.StringVar(value="200")
        ttk.Spinbox(
            dpi_row,
            textvariable=self.resolution_var,
            values=(72, 150, 200, 300),
            width=8,
            state="readonly",
        ).pack(side="left", padx=(8, 0))

        # Separator
        ttk.Separator(stages_frame, orient="horizontal").grid(
            row=6, column=0, columnspan=2, sticky="ew", pady=(8, 4)
        )

        # Skew correction
        self.skew_var = tk.BooleanVar(value=_defaults.enable_skew)
        ttk.Checkbutton(
            stages_frame,
            text="Deskew (Rotation correction)",
            variable=self.skew_var,
        ).grid(row=7, column=0, sticky="w", pady=2)
        ttk.Label(
            stages_frame,
            text="Correct tilted/rotated scans (slower)",
            foreground="gray",
        ).grid(row=7, column=1, sticky="w", padx=(10, 0))

        # LLM checks
        self.llm_checks_var = tk.BooleanVar(value=_defaults.enable_llm_checks)
        ttk.Checkbutton(
            stages_frame,
            text="LLM Checks (Semantic analysis)",
            variable=self.llm_checks_var,
        ).grid(row=8, column=0, sticky="w", pady=2)
        ttk.Label(
            stages_frame,
            text="AI-powered code compliance checks (requires API)",
            foreground="gray",
        ).grid(row=8, column=1, sticky="w", padx=(10, 0))

        row += 1

        # ── ML Settings ─────────────────────────────────────────────
        ml_frame = ttk.LabelFrame(self._inner, text="ML Settings", padding=10)
        ml_frame.grid(row=row, column=0, sticky="ew", **pad)
        ml_frame.columnconfigure(1, weight=1)

        self.ml_enabled_var = tk.BooleanVar(value=_defaults.ml_enabled)
        ttk.Checkbutton(
            ml_frame,
            text="Enable ML relabeling",
            variable=self.ml_enabled_var,
        ).grid(row=0, column=0, sticky="w", pady=2)
        ttk.Label(
            ml_frame,
            text="Run the trained classifier during pipeline feedback.",
            foreground="gray",
        ).grid(row=0, column=1, sticky="w", padx=(10, 0))

        self.ml_hierarchical_var = tk.BooleanVar(
            value=_defaults.ml_hierarchical_enabled
        )
        ttk.Checkbutton(
            ml_frame,
            text="Enable hierarchical title refinement",
            variable=self.ml_hierarchical_var,
            command=self._update_ml_control_state,
        ).grid(row=1, column=0, sticky="w", pady=2)
        ttk.Label(
            ml_frame,
            text="Route title-family predictions through the Stage 2 subtype model.",
            foreground="gray",
        ).grid(row=1, column=1, sticky="w", padx=(10, 0))

        ttk.Label(ml_frame, text="Stage 1 model:").grid(
            row=2, column=0, sticky="w", pady=(8, 2)
        )
        stage1_row = ttk.Frame(ml_frame)
        stage1_row.grid(row=2, column=1, sticky="ew", pady=(8, 2))
        stage1_row.columnconfigure(0, weight=1)
        self.ml_model_path_var = tk.StringVar(value=_defaults.ml_model_path)
        self.ml_model_path_var.trace_add("write", self._refresh_ml_status)
        ttk.Entry(stage1_row, textvariable=self.ml_model_path_var).grid(
            row=0, column=0, sticky="ew"
        )
        ttk.Button(
            stage1_row,
            text="Browse...",
            command=lambda: self._browse_model_path(self.ml_model_path_var),
        ).grid(row=0, column=1, padx=(6, 0))
        self._stage1_model_status_label = ttk.Label(stage1_row, foreground="gray")
        self._stage1_model_status_label.grid(
            row=1, column=0, columnspan=2, sticky="w", pady=(2, 0)
        )

        ttk.Label(ml_frame, text="Stage 2 model:").grid(
            row=3, column=0, sticky="w", pady=2
        )
        stage2_row = ttk.Frame(ml_frame)
        stage2_row.grid(row=3, column=1, sticky="ew", pady=2)
        stage2_row.columnconfigure(0, weight=1)
        self.ml_stage2_model_path_var = tk.StringVar(
            value=_defaults.ml_stage2_model_path
        )
        self.ml_stage2_model_path_var.trace_add("write", self._refresh_ml_status)
        self._stage2_path_entry = ttk.Entry(
            stage2_row,
            textvariable=self.ml_stage2_model_path_var,
        )
        self._stage2_path_entry.grid(row=0, column=0, sticky="ew")
        self._stage2_browse_button = ttk.Button(
            stage2_row,
            text="Browse...",
            command=lambda: self._browse_model_path(self.ml_stage2_model_path_var),
        )
        self._stage2_browse_button.grid(row=0, column=1, padx=(6, 0))
        self._stage2_model_status_label = ttk.Label(stage2_row, foreground="gray")
        self._stage2_model_status_label.grid(
            row=1, column=0, columnspan=2, sticky="w", pady=(2, 0)
        )

        ttk.Separator(ml_frame, orient="horizontal").grid(
            row=4,
            column=0,
            columnspan=2,
            sticky="ew",
            pady=(10, 6),
        )

        advanced_ml = CollapsibleFrame(
            ml_frame,
            title="Advanced ML Runtime",
            initially_open=False,
        )
        advanced_ml.grid(row=5, column=0, columnspan=2, sticky="ew")
        advanced_ml.content.columnconfigure(1, weight=1)

        self.ml_relabel_confidence_var = tk.StringVar(
            value=str(_defaults.ml_relabel_confidence)
        )
        self._add_labeled_entry(
            advanced_ml.content,
            row=0,
            label="Relabel confidence threshold",
            variable=self.ml_relabel_confidence_var,
            help_text="Minimum confidence before ML rewrites a detected label.",
        )

        self.ml_retrain_on_startup_var = tk.BooleanVar(
            value=_defaults.ml_retrain_on_startup
        )
        ttk.Checkbutton(
            advanced_ml.content,
            text="Auto-retrain on startup",
            variable=self.ml_retrain_on_startup_var,
        ).grid(row=1, column=0, sticky="w", pady=2)
        ttk.Label(
            advanced_ml.content,
            text="Run the retrain threshold check when the GUI launches.",
            foreground="gray",
        ).grid(row=1, column=1, sticky="w", padx=(10, 0))

        self.ml_retrain_threshold_var = tk.StringVar(
            value=str(_defaults.ml_retrain_threshold)
        )
        self._add_labeled_entry(
            advanced_ml.content,
            row=2,
            label="Retrain threshold",
            variable=self.ml_retrain_threshold_var,
            help_text="Corrections required before auto retrain becomes eligible.",
        )

        self.ml_feature_cache_enabled_var = tk.BooleanVar(
            value=_defaults.ml_feature_cache_enabled
        )
        ttk.Checkbutton(
            advanced_ml.content,
            text="Enable feature cache",
            variable=self.ml_feature_cache_enabled_var,
        ).grid(row=3, column=0, sticky="w", pady=2)
        ttk.Label(
            advanced_ml.content,
            text="Reuse encoded features across feedback passes when possible.",
            foreground="gray",
        ).grid(row=3, column=1, sticky="w", padx=(10, 0))

        self.ml_drift_enabled_var = tk.BooleanVar(value=_defaults.ml_drift_enabled)
        ttk.Checkbutton(
            advanced_ml.content,
            text="Enable drift detection",
            variable=self.ml_drift_enabled_var,
        ).grid(row=4, column=0, sticky="w", pady=2)
        ttk.Label(
            advanced_ml.content,
            text="Compare current features to stored training drift statistics.",
            foreground="gray",
        ).grid(row=4, column=1, sticky="w", padx=(10, 0))

        self.ml_drift_threshold_var = tk.StringVar(
            value=str(_defaults.ml_drift_threshold)
        )
        self._add_labeled_entry(
            advanced_ml.content,
            row=5,
            label="Drift threshold",
            variable=self.ml_drift_threshold_var,
            help_text="Higher values require a larger feature shift before flagging.",
        )

        self.ml_drift_stats_path_var = tk.StringVar(
            value=_defaults.ml_drift_stats_path
        )
        self._add_labeled_entry(
            advanced_ml.content,
            row=6,
            label="Drift stats path",
            variable=self.ml_drift_stats_path_var,
            help_text="Saved drift-detector statistics file.",
            browse_command=lambda: self._browse_any_path(
                self.ml_drift_stats_path_var,
                title="Select Drift Stats File",
            ),
        )

        optional_ml = CollapsibleFrame(
            ml_frame,
            title="Optional ML Features",
            initially_open=False,
        )
        optional_ml.grid(row=6, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        optional_ml.content.columnconfigure(1, weight=1)

        self.ml_vision_enabled_var = tk.BooleanVar(value=_defaults.ml_vision_enabled)
        ttk.Checkbutton(
            optional_ml.content,
            text="Enable vision features",
            variable=self.ml_vision_enabled_var,
        ).grid(row=0, column=0, sticky="w", pady=2)
        ttk.Label(
            optional_ml.content,
            text="Extract image features from page crops during ML feedback.",
            foreground="gray",
        ).grid(row=0, column=1, sticky="w", padx=(10, 0))

        self.ml_vision_backbone_var = tk.StringVar(value=_defaults.ml_vision_backbone)
        self._add_labeled_entry(
            optional_ml.content,
            row=1,
            label="Vision backbone",
            variable=self.ml_vision_backbone_var,
            help_text="Torchvision backbone name for image features.",
        )

        self.ml_embeddings_enabled_var = tk.BooleanVar(
            value=_defaults.ml_embeddings_enabled
        )
        ttk.Checkbutton(
            optional_ml.content,
            text="Enable text embeddings",
            variable=self.ml_embeddings_enabled_var,
        ).grid(row=2, column=0, sticky="w", pady=2)
        ttk.Label(
            optional_ml.content,
            text="Append sentence embeddings to the structured feature vector.",
            foreground="gray",
        ).grid(row=2, column=1, sticky="w", padx=(10, 0))

        self.ml_embeddings_model_var = tk.StringVar(
            value=_defaults.ml_embeddings_model
        )
        self._add_labeled_entry(
            optional_ml.content,
            row=3,
            label="Embeddings model",
            variable=self.ml_embeddings_model_var,
            help_text="Sentence-transformer model name or local path.",
        )

        self.ml_layout_enabled_var = tk.BooleanVar(value=_defaults.ml_layout_enabled)
        ttk.Checkbutton(
            optional_ml.content,
            text="Enable layout model in pipeline",
            variable=self.ml_layout_enabled_var,
        ).grid(row=4, column=0, sticky="w", pady=2)
        ttk.Label(
            optional_ml.content,
            text="Run LayoutLMv3 layout prediction during analysis when configured.",
            foreground="gray",
        ).grid(row=4, column=1, sticky="w", padx=(10, 0))

        self.ml_layout_model_path_var = tk.StringVar(
            value=_defaults.ml_layout_model_path
        )
        self._add_labeled_entry(
            optional_ml.content,
            row=5,
            label="Layout model",
            variable=self.ml_layout_model_path_var,
            help_text="Fine-tuned LayoutLMv3 checkpoint path or model name.",
        )

        self.ml_gnn_enabled_var = tk.BooleanVar(value=_defaults.ml_gnn_enabled)
        ttk.Checkbutton(
            optional_ml.content,
            text="Enable GNN post-processing",
            variable=self.ml_gnn_enabled_var,
        ).grid(row=6, column=0, sticky="w", pady=2)
        ttk.Label(
            optional_ml.content,
            text="Use the GNN model for downstream relational classification.",
            foreground="gray",
        ).grid(row=6, column=1, sticky="w", padx=(10, 0))

        self.ml_gnn_model_path_var = tk.StringVar(value=_defaults.ml_gnn_model_path)
        self._add_labeled_entry(
            optional_ml.content,
            row=7,
            label="GNN model path",
            variable=self.ml_gnn_model_path_var,
            help_text="Path to the trained GNN checkpoint.",
            browse_command=lambda: self._browse_any_path(
                self.ml_gnn_model_path_var,
                title="Select GNN Model",
            ),
        )

        self.ml_gnn_hidden_dim_var = tk.StringVar(
            value=str(_defaults.ml_gnn_hidden_dim)
        )
        self._add_labeled_entry(
            optional_ml.content,
            row=8,
            label="GNN hidden dim",
            variable=self.ml_gnn_hidden_dim_var,
            help_text="Hidden feature width expected by the GNN checkpoint.",
        )

        candidate_ml = CollapsibleFrame(
            ml_frame,
            title="VOCR Candidate ML",
            initially_open=False,
        )
        candidate_ml.grid(row=7, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        candidate_ml.content.columnconfigure(1, weight=1)

        self.vocr_cand_ml_enabled_var = tk.BooleanVar(
            value=_defaults.vocr_cand_ml_enabled
        )
        ttk.Checkbutton(
            candidate_ml.content,
            text="Enable candidate classifier",
            variable=self.vocr_cand_ml_enabled_var,
        ).grid(row=0, column=0, sticky="w", pady=2)
        ttk.Label(
            candidate_ml.content,
            text="Filter proposed OCR patches using the candidate hit/miss model.",
            foreground="gray",
        ).grid(row=0, column=1, sticky="w", padx=(10, 0))

        self.vocr_cand_classifier_path_var = tk.StringVar(
            value=_defaults.vocr_cand_classifier_path
        )
        self._add_labeled_entry(
            candidate_ml.content,
            row=1,
            label="Candidate model path",
            variable=self.vocr_cand_classifier_path_var,
            help_text="Pickle or joblib artifact for the candidate classifier.",
            browse_command=lambda: self._browse_model_path(
                self.vocr_cand_classifier_path_var
            ),
        )

        self.vocr_cand_ml_threshold_var = tk.StringVar(
            value=str(_defaults.vocr_cand_ml_threshold)
        )
        self._add_labeled_entry(
            candidate_ml.content,
            row=2,
            label="Candidate threshold",
            variable=self.vocr_cand_ml_threshold_var,
            help_text="Minimum hit probability to keep a candidate patch.",
        )

        self.vocr_cand_gnn_prior_enabled_var = tk.BooleanVar(
            value=_defaults.vocr_cand_gnn_prior_enabled
        )
        ttk.Checkbutton(
            candidate_ml.content,
            text="Blend GNN candidate prior",
            variable=self.vocr_cand_gnn_prior_enabled_var,
        ).grid(row=3, column=0, sticky="w", pady=2)
        ttk.Label(
            candidate_ml.content,
            text="Blend graph-based prior scores into candidate ranking.",
            foreground="gray",
        ).grid(row=3, column=1, sticky="w", padx=(10, 0))

        self.vocr_cand_gnn_prior_path_var = tk.StringVar(
            value=_defaults.vocr_cand_gnn_prior_path
        )
        self._add_labeled_entry(
            candidate_ml.content,
            row=4,
            label="Candidate GNN prior path",
            variable=self.vocr_cand_gnn_prior_path_var,
            help_text="Checkpoint used to produce candidate prior scores.",
            browse_command=lambda: self._browse_any_path(
                self.vocr_cand_gnn_prior_path_var,
                title="Select Candidate GNN Prior",
            ),
        )

        self.vocr_cand_gnn_prior_blend_var = tk.StringVar(
            value=str(_defaults.vocr_cand_gnn_prior_blend)
        )
        self._add_labeled_entry(
            candidate_ml.content,
            row=5,
            label="Candidate GNN blend",
            variable=self.vocr_cand_gnn_prior_blend_var,
            help_text="Blend ratio between classifier and GNN prior scores.",
        )

        llm_runtime = CollapsibleFrame(
            ml_frame,
            title="LLM Runtime",
            initially_open=False,
        )
        llm_runtime.grid(row=8, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        llm_runtime.content.columnconfigure(1, weight=1)

        self.llm_provider_var = tk.StringVar(value=_defaults.llm_provider)
        self._add_labeled_entry(
            llm_runtime.content,
            row=0,
            label="LLM provider",
            variable=self.llm_provider_var,
            help_text="Provider key used by the shared LLM client.",
        )

        self.llm_model_var = tk.StringVar(value=_defaults.llm_model)
        self._add_labeled_entry(
            llm_runtime.content,
            row=1,
            label="LLM model",
            variable=self.llm_model_var,
            help_text="Model name sent to the configured provider.",
        )

        self.llm_api_base_var = tk.StringVar(value=_defaults.llm_api_base)
        self._add_labeled_entry(
            llm_runtime.content,
            row=2,
            label="LLM API base",
            variable=self.llm_api_base_var,
            help_text="Base URL for the provider endpoint.",
        )

        self.llm_policy_var = tk.StringVar(value=_defaults.llm_policy)
        self._add_labeled_entry(
            llm_runtime.content,
            row=3,
            label="LLM policy",
            variable=self.llm_policy_var,
            help_text="Runtime policy enforced by the LLM client.",
        )

        self.llm_temperature_var = tk.StringVar(value=str(_defaults.llm_temperature))
        self._add_labeled_entry(
            llm_runtime.content,
            row=4,
            label="LLM temperature",
            variable=self.llm_temperature_var,
            help_text="Sampling temperature for LLM-backed decisions.",
        )

        self.llm_api_key_var = tk.StringVar(value=_defaults.llm_api_key)
        self._add_labeled_entry(
            llm_runtime.content,
            row=5,
            label="LLM API key",
            variable=self.llm_api_key_var,
            help_text="Optional API key for non-local providers.",
            show="*",
        )

        row += 1

        # ── Run Button ───────────────────────────────────────────────
        btn_frame = ttk.Frame(self._inner)
        btn_frame.grid(row=row, column=0, sticky="ew", **pad)
        btn_frame.columnconfigure(0, weight=1)
        btn_frame.columnconfigure(1, weight=0)

        run_style = ttk.Style()
        run_style.configure(
            "Run.TButton", font=("TkDefaultFont", 12, "bold"), padding=(20, 10)
        )

        self.run_button = ttk.Button(
            btn_frame,
            text="Run Processing",
            command=self._run_processing,
            style="Run.TButton",
        )
        self.run_button.grid(row=0, column=0, sticky="ew", pady=(0, 4), padx=(0, 8))

        self.cancel_button = ttk.Button(
            btn_frame,
            text="ABORT",
            command=self._cancel_processing,
            state="disabled",
        )
        self.cancel_button.grid(row=0, column=1, sticky="e", pady=(0, 4))

        row += 1

        # ── Error Panel (hidden by default) ──────────────────────────
        self.error_panel = ErrorPanel(self._inner)
        self.error_panel.grid(row=row, column=0, sticky="ew", **pad)

        row += 1

        # ── Stage Progress Bar with Copy Button ─────────────────────
        stage_frame = ttk.Frame(self.frame)
        stage_frame.grid(row=1, column=0, sticky="ew", padx=(10, 25), pady=(4, 2))
        stage_frame.columnconfigure(0, weight=1)

        self.stage_bar = StageProgressBar(stage_frame)
        self.stage_bar.grid(row=0, column=0, sticky="ew")

        copy_btn = ttk.Button(
            stage_frame,
            text="📋",
            width=3,
            command=self._copy_stage_times,
        )
        copy_btn.grid(row=0, column=1, padx=(6, 0), sticky="ns")

        # ── Embedded Log Console ─────────────────────────────────────
        self.log_panel = LogPanel(self.frame, height=10)
        self.log_panel.grid(row=2, column=0, sticky="nsew", padx=10, pady=(0, 6))

        self._update_ml_control_state()
        self._refresh_ml_status()

    # ------------------------------------------------------------------
    # Config helpers
    # ------------------------------------------------------------------

    def _add_labeled_entry(
        self,
        parent: ttk.Frame,
        *,
        row: int,
        label: str,
        variable: tk.StringVar,
        help_text: str = "",
        browse_command=None,
        show: str | None = None,
    ):
        """Add a labelled entry row with optional browse action."""
        ttk.Label(parent, text=f"{label}:").grid(row=row, column=0, sticky="w", pady=2)
        entry = ttk.Entry(parent, textvariable=variable, show=show)
        entry.grid(row=row, column=1, sticky="ew", pady=2)
        if browse_command is not None:
            ttk.Button(parent, text="Browse...", command=browse_command).grid(
                row=row,
                column=2,
                padx=(6, 0),
                sticky="w",
            )
        if help_text:
            ttk.Label(parent, text=help_text, foreground="gray").grid(
                row=row,
                column=3,
                sticky="w",
                padx=(10, 0),
            )
        return entry

    def _set_config_scalar(
        self,
        cfg: GroupingConfig,
        attr: str,
        variable: tk.StringVar,
        caster,
    ) -> None:
        """Apply a typed scalar value from a Tk variable when valid."""
        raw = variable.get().strip()
        if not raw:
            return
        try:
            setattr(cfg, attr, caster(raw))
        except ValueError:
            return

    def _collect_config(self) -> GroupingConfig:
        """Build a GroupingConfig from all current UI knobs + toggles."""
        cfg = GroupingConfig.from_dict(self.state.config.to_dict())
        # Master toggles
        cfg.enable_tocr = self.tocr_var.get()
        cfg.enable_vocr = self.vocr_var.get()
        cfg.enable_vocr_candidates = self.enable_vocr_candidates_var.get()
        cfg.enable_ocr_reconcile = self.ocr_reconcile_var.get()
        cfg.enable_ocr_preprocess = self.ocr_preprocess_var.get()
        cfg.enable_skew = self.skew_var.get()
        cfg.enable_llm_checks = self.llm_checks_var.get()
        cfg.ml_enabled = self.ml_enabled_var.get()
        cfg.ml_hierarchical_enabled = self.ml_hierarchical_var.get()
        cfg.ml_model_path = self.ml_model_path_var.get().strip() or cfg.ml_model_path
        cfg.ml_stage2_model_path = (
            self.ml_stage2_model_path_var.get().strip() or cfg.ml_stage2_model_path
        )
        cfg.ml_retrain_on_startup = self.ml_retrain_on_startup_var.get()
        cfg.ml_feature_cache_enabled = self.ml_feature_cache_enabled_var.get()
        cfg.ml_drift_enabled = self.ml_drift_enabled_var.get()
        cfg.ml_vision_enabled = self.ml_vision_enabled_var.get()
        cfg.ml_embeddings_enabled = self.ml_embeddings_enabled_var.get()
        cfg.ml_layout_enabled = self.ml_layout_enabled_var.get()
        cfg.ml_gnn_enabled = self.ml_gnn_enabled_var.get()
        cfg.vocr_cand_ml_enabled = self.vocr_cand_ml_enabled_var.get()
        cfg.vocr_cand_gnn_prior_enabled = self.vocr_cand_gnn_prior_enabled_var.get()

        cfg.ml_drift_stats_path = (
            self.ml_drift_stats_path_var.get().strip() or cfg.ml_drift_stats_path
        )
        cfg.ml_vision_backbone = (
            self.ml_vision_backbone_var.get().strip() or cfg.ml_vision_backbone
        )
        cfg.ml_embeddings_model = (
            self.ml_embeddings_model_var.get().strip() or cfg.ml_embeddings_model
        )
        cfg.ml_layout_model_path = (
            self.ml_layout_model_path_var.get().strip() or cfg.ml_layout_model_path
        )
        cfg.ml_gnn_model_path = (
            self.ml_gnn_model_path_var.get().strip() or cfg.ml_gnn_model_path
        )
        cfg.vocr_cand_classifier_path = (
            self.vocr_cand_classifier_path_var.get().strip()
            or cfg.vocr_cand_classifier_path
        )
        cfg.vocr_cand_gnn_prior_path = (
            self.vocr_cand_gnn_prior_path_var.get().strip()
            or cfg.vocr_cand_gnn_prior_path
        )
        cfg.llm_provider = self.llm_provider_var.get().strip() or cfg.llm_provider
        cfg.llm_model = self.llm_model_var.get().strip() or cfg.llm_model
        cfg.llm_api_base = self.llm_api_base_var.get().strip() or cfg.llm_api_base
        cfg.llm_policy = self.llm_policy_var.get().strip() or cfg.llm_policy
        cfg.llm_api_key = self.llm_api_key_var.get().strip()

        self._set_config_scalar(
            cfg,
            "ocr_reconcile_resolution",
            self.ocr_dpi_var,
            int,
        )
        self._set_config_scalar(
            cfg,
            "ml_relabel_confidence",
            self.ml_relabel_confidence_var,
            float,
        )
        self._set_config_scalar(
            cfg,
            "ml_retrain_threshold",
            self.ml_retrain_threshold_var,
            int,
        )
        self._set_config_scalar(
            cfg,
            "ml_drift_threshold",
            self.ml_drift_threshold_var,
            float,
        )
        self._set_config_scalar(
            cfg,
            "ml_gnn_hidden_dim",
            self.ml_gnn_hidden_dim_var,
            int,
        )
        self._set_config_scalar(
            cfg,
            "vocr_cand_ml_threshold",
            self.vocr_cand_ml_threshold_var,
            float,
        )
        self._set_config_scalar(
            cfg,
            "vocr_cand_gnn_prior_blend",
            self.vocr_cand_gnn_prior_blend_var,
            float,
        )
        self._set_config_scalar(
            cfg,
            "llm_temperature",
            self.llm_temperature_var,
            float,
        )

        self.state.set_config(cfg, config_file_path=self.state.config_file_path)
        return cfg

    def _apply_config(self, cfg: GroupingConfig) -> None:
        """Push a GroupingConfig into all UI controls."""
        self.state.set_config(cfg, config_file_path=self.state.config_file_path)
        self.tocr_var.set(cfg.enable_tocr)
        self.vocr_var.set(cfg.enable_vocr)
        self.enable_vocr_candidates_var.set(cfg.enable_vocr_candidates)
        self.ocr_reconcile_var.set(cfg.enable_ocr_reconcile)
        self.ocr_preprocess_var.set(cfg.enable_ocr_preprocess)
        self.skew_var.set(cfg.enable_skew)
        self.llm_checks_var.set(cfg.enable_llm_checks)
        self.ocr_dpi_var.set(str(cfg.ocr_reconcile_resolution))
        self.ml_enabled_var.set(cfg.ml_enabled)
        self.ml_hierarchical_var.set(cfg.ml_hierarchical_enabled)
        self.ml_model_path_var.set(cfg.ml_model_path)
        self.ml_stage2_model_path_var.set(cfg.ml_stage2_model_path)
        self.ml_relabel_confidence_var.set(str(cfg.ml_relabel_confidence))
        self.ml_retrain_on_startup_var.set(cfg.ml_retrain_on_startup)
        self.ml_retrain_threshold_var.set(str(cfg.ml_retrain_threshold))
        self.ml_feature_cache_enabled_var.set(cfg.ml_feature_cache_enabled)
        self.ml_drift_enabled_var.set(cfg.ml_drift_enabled)
        self.ml_drift_threshold_var.set(str(cfg.ml_drift_threshold))
        self.ml_drift_stats_path_var.set(cfg.ml_drift_stats_path)
        self.ml_vision_enabled_var.set(cfg.ml_vision_enabled)
        self.ml_vision_backbone_var.set(cfg.ml_vision_backbone)
        self.ml_embeddings_enabled_var.set(cfg.ml_embeddings_enabled)
        self.ml_embeddings_model_var.set(cfg.ml_embeddings_model)
        self.ml_layout_enabled_var.set(cfg.ml_layout_enabled)
        self.ml_layout_model_path_var.set(cfg.ml_layout_model_path)
        self.ml_gnn_enabled_var.set(cfg.ml_gnn_enabled)
        self.ml_gnn_model_path_var.set(cfg.ml_gnn_model_path)
        self.ml_gnn_hidden_dim_var.set(str(cfg.ml_gnn_hidden_dim))
        self.vocr_cand_ml_enabled_var.set(cfg.vocr_cand_ml_enabled)
        self.vocr_cand_classifier_path_var.set(cfg.vocr_cand_classifier_path)
        self.vocr_cand_ml_threshold_var.set(str(cfg.vocr_cand_ml_threshold))
        self.vocr_cand_gnn_prior_enabled_var.set(cfg.vocr_cand_gnn_prior_enabled)
        self.vocr_cand_gnn_prior_path_var.set(cfg.vocr_cand_gnn_prior_path)
        self.vocr_cand_gnn_prior_blend_var.set(str(cfg.vocr_cand_gnn_prior_blend))
        self.llm_provider_var.set(cfg.llm_provider)
        self.llm_model_var.set(cfg.llm_model)
        self.llm_api_base_var.set(cfg.llm_api_base)
        self.llm_policy_var.set(cfg.llm_policy)
        self.llm_temperature_var.set(str(cfg.llm_temperature))
        self.llm_api_key_var.set(cfg.llm_api_key)
        self._update_ml_control_state()
        self._refresh_ml_status()
        self._refresh_config_file_label()

    def _on_load_config(self) -> None:
        """Handle load_config event from Runs tab."""
        config_dict = getattr(self.state, "pending_config", None)
        if not config_dict:
            return
        cfg = GroupingConfig.from_dict(config_dict)
        self._apply_config(cfg)
        self._notify_loaded_config_validation(cfg, source="Imported config")
        self.state.pending_config = None  # Clear after applying

    def _refresh_config_file_label(self) -> None:
        """Show the current config source in the Pipeline tab."""
        path = self.state.config_file_path
        if path is None:
            self._config_path_var.set("Working config: in memory")
            return
        self._config_path_var.set(f"Working config: {path.name}")

    def _load_config_from_file(self) -> None:
        """Load a YAML or TOML config file into the Pipeline tab."""
        path_str = filedialog.askopenfilename(
            title="Load Pipeline Config",
            filetypes=[
                ("Config Files", "*.yaml *.yml *.toml"),
                ("YAML Files", "*.yaml *.yml"),
                ("TOML Files", "*.toml"),
                ("All Files", "*.*"),
            ],
            initialdir=str(Path.cwd()),
        )
        if not path_str:
            return

        path = Path(path_str)
        try:
            cfg = GroupingConfig.from_file(path)
        except Exception as exc:  # noqa: BLE001 - GUI should surface config load errors
            messagebox.showerror("Config Load Failed", str(exc))
            return

        self.state.config_file_path = path
        self._apply_config(cfg)
        self._notify_loaded_config_validation(cfg, source="Loaded config")

    def _save_config_to_file(self) -> None:
        """Save the current GUI config to YAML or TOML."""
        cfg = self._collect_config()
        if not self._confirm_save_with_validation(cfg):
            return
        initial_name = "plancheck-config.yaml"
        if self.state.config_file_path is not None:
            initial_name = self.state.config_file_path.name

        path_str = filedialog.asksaveasfilename(
            title="Save Pipeline Config",
            defaultextension=".yaml",
            initialfile=initial_name,
            filetypes=[
                ("YAML Files", "*.yaml"),
                ("YAML Files", "*.yml"),
                ("TOML Files", "*.toml"),
            ],
            initialdir=str(Path.cwd()),
        )
        if not path_str:
            return

        path = Path(path_str)
        suffix = path.suffix.lower()
        try:
            if suffix in (".yaml", ".yml"):
                path.write_text(self._config_to_yaml(cfg), encoding="utf-8")
            elif suffix == ".toml":
                path.write_text(self._config_to_toml(cfg), encoding="utf-8")
            else:
                raise ValueError(
                    "Unsupported config extension. Use .yaml, .yml, or .toml"
                )
        except Exception as exc:  # noqa: BLE001 - GUI should surface config save errors
            messagebox.showerror("Config Save Failed", str(exc))
            return

        self.state.set_config(cfg, config_file_path=path)
        self._refresh_config_file_label()

    def _browse_model_path(self, variable: tk.StringVar) -> None:
        """Choose a model artifact path for a config field."""
        path_str = filedialog.askopenfilename(
            title="Select Model Artifact",
            filetypes=[
                ("Pickle / Joblib", "*.pkl *.joblib"),
                ("All Files", "*.*"),
            ],
            initialdir=str(Path.cwd()),
        )
        if not path_str:
            return
        variable.set(self._normalize_path(Path(path_str)))

    def _browse_any_path(self, variable: tk.StringVar, *, title: str) -> None:
        """Choose any file path for a config field."""
        path_str = filedialog.askopenfilename(
            title=title,
            filetypes=[("All Files", "*.*")],
            initialdir=str(Path.cwd()),
        )
        if not path_str:
            return
        variable.set(self._normalize_path(Path(path_str)))

    def _normalize_path(self, path: Path) -> str:
        """Prefer repo-relative paths when possible for saved config values."""
        try:
            return str(path.relative_to(Path.cwd()))
        except ValueError:
            return str(path)

    def _resolve_runtime_path(self, raw_path: str) -> Path | None:
        """Resolve a potentially relative config path against the workspace."""
        path_text = raw_path.strip()
        if not path_text:
            return None
        path = Path(path_text)
        if not path.is_absolute():
            path = Path.cwd() / path
        return path

    def _collect_run_validation_messages(
        self,
        cfg: GroupingConfig,
    ) -> tuple[list[str], list[str]]:
        """Return blocking errors and non-blocking warnings for a run."""
        errors: list[str] = []
        warnings: list[str] = []

        stage1_path = self._resolve_runtime_path(cfg.ml_model_path)
        if cfg.ml_enabled and (stage1_path is None or not stage1_path.exists()):
            warnings.append(
                "ML relabeling is enabled but the Stage 1 model file was not found. "
                "The pipeline will fall back to rule-based labels until a model is trained."
            )

        if cfg.ml_hierarchical_enabled:
            stage2_path = self._resolve_runtime_path(cfg.ml_stage2_model_path)
            if stage2_path is None or not stage2_path.exists():
                warnings.append(
                    "Hierarchical routing is enabled but the Stage 2 model file was not found. "
                    "Title predictions will remain at Stage 1."
                )

        if cfg.ml_drift_enabled:
            drift_stats_path = self._resolve_runtime_path(cfg.ml_drift_stats_path)
            if drift_stats_path is None or not drift_stats_path.exists():
                warnings.append(
                    "Drift detection is enabled but the drift stats file was not found. "
                    "No drift warnings will be produced until drift stats are generated."
                )

        if cfg.vocr_cand_ml_enabled:
            candidate_model = self._resolve_runtime_path(cfg.vocr_cand_classifier_path)
            if candidate_model is None or not candidate_model.exists():
                errors.append(
                    "VOCR candidate ML is enabled but the candidate classifier file was not found. "
                    "Train or disable candidate ML before running the pipeline."
                )

        if cfg.vocr_cand_gnn_prior_enabled:
            candidate_gnn = self._resolve_runtime_path(cfg.vocr_cand_gnn_prior_path)
            if candidate_gnn is None or not candidate_gnn.exists():
                errors.append(
                    "VOCR candidate GNN prior is enabled but the prior checkpoint was not found. "
                    "Provide the checkpoint or disable the GNN prior before running."
                )

        if cfg.ml_layout_enabled:
            model_name = cfg.ml_layout_model_path.strip()
            if not model_name:
                errors.append(
                    "Layout inference is enabled but no LayoutLMv3 checkpoint is configured."
                )
            elif model_name == "microsoft/layoutlmv3-base":
                warnings.append(
                    "Layout inference is enabled with the base LayoutLMv3 checkpoint. "
                    "Use a fine-tuned checkpoint for meaningful layout predictions."
                )

        if cfg.enable_llm_checks:
            provider = cfg.llm_provider.strip().lower()
            if provider in {"openai", "anthropic"} and not cfg.llm_api_key.strip():
                errors.append(
                    "LLM checks are enabled for a hosted provider but no API key is configured."
                )

        return errors, warnings

    def _format_validation_summary(
        self,
        errors: list[str],
        warnings: list[str],
    ) -> str:
        """Format validation findings for dialogs shown in the GUI."""
        lines: list[str] = []
        if errors:
            lines.append("Blocking issues:")
            lines.extend(f"- {message}" for message in errors)
        if warnings:
            if lines:
                lines.append("")
            lines.append("Warnings:")
            lines.extend(f"- {message}" for message in warnings)
        return "\n".join(lines)

    def _notify_loaded_config_validation(self, cfg: GroupingConfig, *, source: str) -> None:
        """Show non-blocking validation feedback after importing a config."""
        errors, warnings = self._collect_run_validation_messages(cfg)
        if not errors and not warnings:
            return
        messagebox.showwarning(
            f"{source} Needs Review",
            f"{source} has ML runtime issues that should be reviewed before running:\n\n"
            f"{self._format_validation_summary(errors, warnings)}",
        )

    def _confirm_save_with_validation(self, cfg: GroupingConfig) -> bool:
        """Confirm saving a config when validation finds warnings or errors."""
        errors, warnings = self._collect_run_validation_messages(cfg)
        if not errors and not warnings:
            return True

        title = "Save Config With Warnings"
        prompt = "Save this config anyway?"
        if errors:
            title = "Save Config With Blocking Issues"
            prompt = "Save this config even though it has blocking issues?"

        return messagebox.askyesno(
            title,
            "The current pipeline config has validation findings:\n\n"
            + self._format_validation_summary(errors, warnings)
            + f"\n\n{prompt}",
        )

    def _update_ml_control_state(self) -> None:
        """Enable Stage 2 path controls only when hierarchical mode is enabled."""
        state = "normal" if self.ml_hierarchical_var.get() else "disabled"
        if hasattr(self, "_stage2_path_entry"):
            self._stage2_path_entry.configure(state=state)
        if hasattr(self, "_stage2_browse_button"):
            self._stage2_browse_button.configure(state=state)
        self._refresh_ml_status()

    def _refresh_ml_status(self, *_args) -> None:
        """Update model-path status labels based on the current UI values."""
        self._set_model_status(
            getattr(self, "_stage1_model_status_label", None),
            self.ml_model_path_var.get(),
            prefix="Stage 1",
        )

        hierarchical_enabled = self.ml_hierarchical_var.get()
        if not hierarchical_enabled:
            self._set_model_status_text(
                getattr(self, "_stage2_model_status_label", None),
                "Stage 2 model inactive until hierarchical routing is enabled.",
                "gray",
            )
            return

        self._set_model_status(
            getattr(self, "_stage2_model_status_label", None),
            self.ml_stage2_model_path_var.get(),
            prefix="Stage 2",
        )

    def _set_model_status(self, label_widget, raw_path: str, *, prefix: str) -> None:
        """Render a model-path existence message for a status label."""
        path_text = raw_path.strip()
        if not path_text:
            self._set_model_status_text(
                label_widget,
                f"{prefix} model path is empty.",
                "orange",
            )
            return

        path = Path(path_text)
        if not path.is_absolute():
            path = Path.cwd() / path

        if path.exists():
            self._set_model_status_text(
                label_widget,
                f"{prefix} model found: {path.name}",
                "green",
            )
            return

        self._set_model_status_text(
            label_widget,
            f"{prefix} model not found: {path_text}",
            "orange",
        )

    def _set_model_status_text(
        self,
        label_widget,
        text: str,
        color: str,
    ) -> None:
        """Safely configure a status label when present."""
        if label_widget is None:
            return
        label_widget.configure(text=text, foreground=color)

    def _config_scalar_to_text(self, value: Any) -> str:
        """Format a scalar config value for YAML/TOML output."""
        if isinstance(value, str):
            return json.dumps(value, ensure_ascii=False)
        if isinstance(value, bool):
            return "true" if value else "false"
        return str(value)

    def _config_to_yaml(self, cfg: GroupingConfig) -> str:
        """Serialize a flat config mapping as YAML without extra dependencies."""
        lines = ["# plancheck pipeline config"]
        for key, value in cfg.to_dict().items():
            lines.append(f"{key}: {self._config_scalar_to_text(value)}")
        lines.append("")
        return "\n".join(lines)

    def _config_to_toml(self, cfg: GroupingConfig) -> str:
        """Serialize a flat config mapping as TOML without extra dependencies."""
        lines = ["# plancheck pipeline config"]
        for key, value in cfg.to_dict().items():
            lines.append(f"{key} = {self._config_scalar_to_text(value)}")
        lines.append("")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # PDF / Page selection (preserved from original)
    # ------------------------------------------------------------------

    def _select_file(self) -> None:
        f = filedialog.askopenfilename(
            title="Select PDF File",
            filetypes=[("PDF Files", "*.pdf"), ("All Files", "*.*")],
            initialdir=str(Path("input")),
        )
        if f:
            path = Path(f)
            self.pdf_files = [path]
            self.file_label_var.set(path.name)
            # Update shared state
            self.state.pdf_path = path
            self.state.notify("pdf_changed")

    def _clear_file(self) -> None:
        self.pdf_files.clear()
        self.file_label_var.set("No file selected")
        self.state.pdf_path = None
        self.state.notify("pdf_changed")

    def _update_page_mode(self) -> None:
        mode = self.page_mode_var.get()
        self.single_page_entry.config(
            state="normal" if mode == "single" else "disabled"
        )
        self.start_entry.config(state="normal" if mode == "range" else "disabled")
        self.end_entry.config(state="normal" if mode == "range" else "disabled")

    def _parse_page_range(self) -> tuple[int, int | None]:
        mode = self.page_mode_var.get()
        if mode == "all":
            return 0, None
        if mode == "single":
            page_str = self.single_page_var.get().strip()
            if not page_str:
                raise ValueError("Please enter a page number")
            page = int(page_str)
            if page < 1:
                raise ValueError("Page number must be 1 or greater")
            return page - 1, page
        start_str = self.start_page_var.get().strip()
        end_str = self.end_page_var.get().strip()
        start = int(start_str) - 1 if start_str else 0
        if start < 0:
            start = 0
        end = int(end_str) if end_str else None
        return start, end

    def _on_mousewheel(self, event) -> None:
        if not self._wheel_active or not self._canvas.winfo_ismapped():
            return
        self._canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    # ------------------------------------------------------------------
    # Pipeline Execution (embedded – replaces subprocess)
    # ------------------------------------------------------------------

    def _run_processing(self) -> None:
        if not self.pdf_files:
            messagebox.showwarning("No Files", "Please add at least one PDF file.")
            return
        try:
            start, end = self._parse_page_range()
        except ValueError as exc:
            messagebox.showerror("Invalid Input", str(exc) or "Invalid page range")
            return
        try:
            resolution = int(self.resolution_var.get())
        except ValueError:
            resolution = 200

        cfg = self._collect_config()
        self.state.set_config(cfg, config_file_path=self.state.config_file_path)

        errors, warnings = self._collect_run_validation_messages(cfg)
        summary = self._format_validation_summary(errors, warnings)
        if errors:
            messagebox.showerror("Run Blocked", summary)
            return
        if warnings:
            proceed = messagebox.askyesno(
                "Continue With Warnings",
                "The current pipeline settings have warnings:\n\n"
                + summary
                + "\n\nContinue anyway?",
            )
            if not proceed:
                return

        runs_root = Path("runs")

        self.log_panel.clear()
        self.error_panel.clear()
        self.stage_bar.reset()
        self.run_button.config(state="disabled")
        self.cancel_button.config(state="normal")

        # Notify other tabs to close their database connections
        self.state.notify("pipeline_starting")

        self._worker = PipelineWorker(
            self.root, self.log_panel, self.stage_bar, self.error_panel
        )
        worker = self._worker

        def target():
            from ..runners.run_pdf_batch import cleanup_old_runs, run_pdf

            results = []
            for pdf_path in self.pdf_files:
                if worker and worker.cancel_event.is_set():
                    break
                run_prefix = pdf_path.stem.replace(" ", "_")[:20]
                run_dir = run_pdf(
                    pdf=pdf_path,
                    start=start,
                    end=end,
                    resolution=resolution,
                    run_root=runs_root,
                    run_prefix=run_prefix,
                    cfg=cfg,
                    cancel_event=worker.cancel_event if worker else None,
                    stage_callback=worker.post_stage if worker else None,
                )
                results.append(run_dir)
            cleanup_old_runs(runs_root, keep=50)
            return results

        def on_done(result, error, elapsed):
            self.run_button.config(state="normal")
            self.cancel_button.config(state="disabled")
            if error:
                self.error_panel.add_error(str(error), "ERROR")
            if result and not error:
                self.state.last_run_dir = (
                    result[-1] if isinstance(result, list) else result
                )
                self.state.notify("run_completed")

        self._worker.run(target, on_done=on_done)

    def _cancel_processing(self) -> None:
        if self._worker:
            self._worker.cancel()

    def _copy_stage_times(self) -> None:
        """Copy all stage times to clipboard."""
        self.stage_bar.copy_times()
