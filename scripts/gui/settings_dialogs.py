"""Settings dialogs — modal windows launched from the Settings menu.

Absorbs the Advanced ML Runtime, Optional ML Features, and LLM Runtime
collapsible sections from the Pipeline tab into standalone dialogs.
Each dialog reads/writes values through GuiState.config.
"""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk
from typing import Any

from plancheck.config import GroupingConfig


# ---------------------------------------------------------------------------
# Base class for settings dialogs
# ---------------------------------------------------------------------------


class _SettingsDialog(tk.Toplevel):
    """Base class for modal settings dialogs with OK / Cancel / Apply."""

    def __init__(
        self,
        parent: tk.Widget,
        state: Any,
        *,
        title: str,
        width: int = 600,
        height: int = 500,
    ) -> None:
        super().__init__(parent)
        self.state = state
        self.applied = False
        self.title(title)
        self.geometry(f"{width}x{height}")
        self.resizable(True, True)
        self.minsize(450, 300)

        # Modal behavior
        self.transient(parent)
        self.grab_set()

        # Main content area
        self._content = ttk.Frame(self, padding=10)
        self._content.pack(fill="both", expand=True)
        self._content.columnconfigure(1, weight=1)

        self._build_content(self._content)

        # Button row
        btn_frame = ttk.Frame(self, padding=(10, 5))
        btn_frame.pack(fill="x", side="bottom")

        ttk.Button(btn_frame, text="Cancel", command=self.destroy).pack(
            side="right", padx=(4, 0)
        )
        ttk.Button(btn_frame, text="Apply", command=self._apply).pack(
            side="right", padx=(4, 0)
        )
        ttk.Button(btn_frame, text="OK", command=self._ok).pack(
            side="right", padx=(4, 0)
        )

        # Close on Escape
        self.bind("<Escape>", lambda e: self.destroy())

        self._load_values()

    def _build_content(self, parent: ttk.Frame) -> None:
        """Override to add widgets to the dialog content area."""

    def _load_values(self) -> None:
        """Override to populate controls from state.config."""

    def _apply_values(self) -> None:
        """Override to write control values back to state.config."""

    def _ok(self) -> None:
        self._apply()
        self.destroy()

    def _apply(self) -> None:
        self._apply_values()
        self.applied = True

    def _add_labeled_entry(
        self,
        parent: ttk.Frame,
        *,
        row: int,
        label: str,
        variable: tk.StringVar,
        help_text: str = "",
        browse_command: Any = None,
        show: str | None = None,
    ) -> ttk.Entry:
        """Add a labelled entry row with optional browse action."""
        ttk.Label(parent, text=f"{label}:").grid(
            row=row, column=0, sticky="w", pady=2, padx=(0, 8)
        )
        entry = ttk.Entry(parent, textvariable=variable, show=show)
        entry.grid(row=row, column=1, sticky="ew", pady=2)
        if browse_command is not None:
            ttk.Button(parent, text="Browse…", command=browse_command).grid(
                row=row, column=2, padx=(6, 0), sticky="w"
            )
        if help_text:
            ttk.Label(parent, text=help_text, foreground="gray", wraplength=400).grid(
                row=row, column=1, columnspan=2, sticky="w", pady=(0, 2)
            )
        return entry

    def _browse_file(self, variable: tk.StringVar, *, title: str) -> None:
        """Open a file dialog and set the result on a StringVar."""
        path_str = filedialog.askopenfilename(
            title=title,
            filetypes=[("All Files", "*.*")],
            initialdir=str(Path.cwd()),
            parent=self,
        )
        if path_str:
            try:
                variable.set(str(Path(path_str).relative_to(Path.cwd())))
            except ValueError:
                variable.set(path_str)

    def _set_config_scalar(
        self,
        cfg: GroupingConfig,
        attr: str,
        variable: tk.StringVar,
        caster: type,
    ) -> None:
        """Apply a typed scalar value from a Tk variable when valid."""
        raw = variable.get().strip()
        if not raw:
            return
        try:
            setattr(cfg, attr, caster(raw))
        except ValueError:
            return


# ---------------------------------------------------------------------------
# ML Runtime Dialog (absorbs "Advanced ML Runtime" collapsible section)
# ---------------------------------------------------------------------------


class MLRuntimeDialog(_SettingsDialog):
    """Settings → ML Runtime… dialog."""

    def __init__(self, parent: tk.Widget, state: Any) -> None:
        super().__init__(parent, state, title="ML Runtime Settings", height=420)

    def _build_content(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(1, weight=1)
        row = 0

        ttk.Label(
            parent,
            text="Advanced ML Runtime Configuration",
            font=("TkDefaultFont", 10, "bold"),
        ).grid(row=row, column=0, columnspan=3, sticky="w", pady=(0, 10))
        row += 1

        self.ml_relabel_confidence_var = tk.StringVar()
        self._add_labeled_entry(
            parent,
            row=row,
            label="Relabel confidence threshold",
            variable=self.ml_relabel_confidence_var,
            help_text="Minimum confidence before ML rewrites a detected label.",
        )
        row += 1

        self.ml_retrain_on_startup_var = tk.BooleanVar()
        ttk.Checkbutton(
            parent,
            text="Auto-retrain on startup",
            variable=self.ml_retrain_on_startup_var,
        ).grid(row=row, column=0, columnspan=2, sticky="w", pady=2)
        row += 1

        self.ml_retrain_threshold_var = tk.StringVar()
        self._add_labeled_entry(
            parent,
            row=row,
            label="Retrain threshold",
            variable=self.ml_retrain_threshold_var,
            help_text="Corrections required before auto retrain becomes eligible.",
        )
        row += 1

        self.ml_feature_cache_enabled_var = tk.BooleanVar()
        ttk.Checkbutton(
            parent,
            text="Enable feature cache",
            variable=self.ml_feature_cache_enabled_var,
        ).grid(row=row, column=0, columnspan=2, sticky="w", pady=2)
        row += 1

        self.ml_drift_enabled_var = tk.BooleanVar()
        ttk.Checkbutton(
            parent,
            text="Enable drift detection",
            variable=self.ml_drift_enabled_var,
        ).grid(row=row, column=0, columnspan=2, sticky="w", pady=2)
        row += 1

        self.ml_drift_threshold_var = tk.StringVar()
        self._add_labeled_entry(
            parent,
            row=row,
            label="Drift threshold",
            variable=self.ml_drift_threshold_var,
            help_text="Higher values require a larger feature shift before flagging.",
        )
        row += 1

        self.ml_drift_stats_path_var = tk.StringVar()
        self._add_labeled_entry(
            parent,
            row=row,
            label="Drift stats path",
            variable=self.ml_drift_stats_path_var,
            help_text="Saved drift-detector statistics file.",
            browse_command=lambda: self._browse_file(
                self.ml_drift_stats_path_var, title="Select Drift Stats File"
            ),
        )
        row += 1

        self.ml_comparison_threshold_var = tk.StringVar()
        self._add_labeled_entry(
            parent,
            row=row,
            label="Comparison F1 threshold",
            variable=self.ml_comparison_threshold_var,
            help_text="Min absolute per-class F1 delta to count as improved / regressed.",
        )

    def _load_values(self) -> None:
        cfg = self.state.config
        self.ml_relabel_confidence_var.set(str(cfg.ml_relabel_confidence))
        self.ml_retrain_on_startup_var.set(cfg.ml_retrain_on_startup)
        self.ml_retrain_threshold_var.set(str(cfg.ml_retrain_threshold))
        self.ml_feature_cache_enabled_var.set(cfg.ml_feature_cache_enabled)
        self.ml_drift_enabled_var.set(cfg.ml_drift_enabled)
        self.ml_drift_threshold_var.set(str(cfg.ml_drift_threshold))
        self.ml_drift_stats_path_var.set(cfg.ml_drift_stats_path)
        self.ml_comparison_threshold_var.set(str(cfg.ml_comparison_threshold))

    def _apply_values(self) -> None:
        cfg = self.state.config
        cfg.ml_retrain_on_startup = self.ml_retrain_on_startup_var.get()
        cfg.ml_feature_cache_enabled = self.ml_feature_cache_enabled_var.get()
        cfg.ml_drift_enabled = self.ml_drift_enabled_var.get()
        cfg.ml_drift_stats_path = (
            self.ml_drift_stats_path_var.get().strip() or cfg.ml_drift_stats_path
        )
        self._set_config_scalar(
            cfg, "ml_relabel_confidence", self.ml_relabel_confidence_var, float
        )
        self._set_config_scalar(
            cfg, "ml_retrain_threshold", self.ml_retrain_threshold_var, int
        )
        self._set_config_scalar(
            cfg, "ml_drift_threshold", self.ml_drift_threshold_var, float
        )
        self._set_config_scalar(
            cfg, "ml_comparison_threshold", self.ml_comparison_threshold_var, float
        )


# ---------------------------------------------------------------------------
# ML Features Dialog (absorbs "Optional ML Features" collapsible section)
# ---------------------------------------------------------------------------


class MLFeaturesDialog(_SettingsDialog):
    """Settings → ML Features… dialog."""

    def __init__(self, parent: tk.Widget, state: Any) -> None:
        super().__init__(parent, state, title="ML Feature Settings", height=550)

    def _build_content(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(1, weight=1)
        row = 0

        ttk.Label(
            parent,
            text="Optional ML Features",
            font=("TkDefaultFont", 10, "bold"),
        ).grid(row=row, column=0, columnspan=3, sticky="w", pady=(0, 10))
        row += 1

        # Vision features
        self.ml_vision_enabled_var = tk.BooleanVar()
        ttk.Checkbutton(
            parent,
            text="Enable vision features",
            variable=self.ml_vision_enabled_var,
        ).grid(row=row, column=0, columnspan=2, sticky="w", pady=2)
        row += 1

        self.ml_vision_backbone_var = tk.StringVar()
        self._add_labeled_entry(
            parent,
            row=row,
            label="Vision backbone",
            variable=self.ml_vision_backbone_var,
            help_text="Torchvision backbone name for image features.",
        )
        row += 1

        ttk.Separator(parent, orient="horizontal").grid(
            row=row, column=0, columnspan=3, sticky="ew", pady=6
        )
        row += 1

        # Text embeddings
        self.ml_embeddings_enabled_var = tk.BooleanVar()
        ttk.Checkbutton(
            parent,
            text="Enable text embeddings",
            variable=self.ml_embeddings_enabled_var,
        ).grid(row=row, column=0, columnspan=2, sticky="w", pady=2)
        row += 1

        self.ml_embeddings_model_var = tk.StringVar()
        self._add_labeled_entry(
            parent,
            row=row,
            label="Embeddings model",
            variable=self.ml_embeddings_model_var,
            help_text="Sentence-transformer model name or local path.",
        )
        row += 1

        ttk.Separator(parent, orient="horizontal").grid(
            row=row, column=0, columnspan=3, sticky="ew", pady=6
        )
        row += 1

        # Layout model
        self.ml_layout_enabled_var = tk.BooleanVar()
        ttk.Checkbutton(
            parent,
            text="Enable layout model in pipeline",
            variable=self.ml_layout_enabled_var,
        ).grid(row=row, column=0, columnspan=2, sticky="w", pady=2)
        row += 1

        self.ml_layout_model_path_var = tk.StringVar()
        self._add_labeled_entry(
            parent,
            row=row,
            label="Layout model",
            variable=self.ml_layout_model_path_var,
            help_text="Fine-tuned LayoutLMv3 checkpoint path or model name.",
        )
        row += 1

        ttk.Separator(parent, orient="horizontal").grid(
            row=row, column=0, columnspan=3, sticky="ew", pady=6
        )
        row += 1

        # GNN
        self.ml_gnn_enabled_var = tk.BooleanVar()
        ttk.Checkbutton(
            parent,
            text="Enable GNN post-processing",
            variable=self.ml_gnn_enabled_var,
        ).grid(row=row, column=0, columnspan=2, sticky="w", pady=2)
        row += 1

        self.ml_gnn_model_path_var = tk.StringVar()
        self._add_labeled_entry(
            parent,
            row=row,
            label="GNN model path",
            variable=self.ml_gnn_model_path_var,
            help_text="Path to the trained GNN checkpoint.",
            browse_command=lambda: self._browse_file(
                self.ml_gnn_model_path_var, title="Select GNN Model"
            ),
        )
        row += 1

        self.ml_gnn_hidden_dim_var = tk.StringVar()
        self._add_labeled_entry(
            parent,
            row=row,
            label="GNN hidden dim",
            variable=self.ml_gnn_hidden_dim_var,
            help_text="Hidden feature width expected by the GNN checkpoint.",
        )
        row += 1

        self.ml_gnn_patience_var = tk.StringVar()
        self._add_labeled_entry(
            parent,
            row=row,
            label="GNN early-stop patience",
            variable=self.ml_gnn_patience_var,
            help_text="Stop GNN training after N epochs without val improvement (0 = disabled).",
        )

    def _load_values(self) -> None:
        cfg = self.state.config
        self.ml_vision_enabled_var.set(cfg.ml_vision_enabled)
        self.ml_vision_backbone_var.set(cfg.ml_vision_backbone)
        self.ml_embeddings_enabled_var.set(cfg.ml_embeddings_enabled)
        self.ml_embeddings_model_var.set(cfg.ml_embeddings_model)
        self.ml_layout_enabled_var.set(cfg.ml_layout_enabled)
        self.ml_layout_model_path_var.set(cfg.ml_layout_model_path)
        self.ml_gnn_enabled_var.set(cfg.ml_gnn_enabled)
        self.ml_gnn_model_path_var.set(cfg.ml_gnn_model_path)
        self.ml_gnn_hidden_dim_var.set(str(cfg.ml_gnn_hidden_dim))
        self.ml_gnn_patience_var.set(str(cfg.ml_gnn_patience))

    def _apply_values(self) -> None:
        cfg = self.state.config
        cfg.ml_vision_enabled = self.ml_vision_enabled_var.get()
        cfg.ml_vision_backbone = (
            self.ml_vision_backbone_var.get().strip() or cfg.ml_vision_backbone
        )
        cfg.ml_embeddings_enabled = self.ml_embeddings_enabled_var.get()
        cfg.ml_embeddings_model = (
            self.ml_embeddings_model_var.get().strip() or cfg.ml_embeddings_model
        )
        cfg.ml_layout_enabled = self.ml_layout_enabled_var.get()
        cfg.ml_layout_model_path = (
            self.ml_layout_model_path_var.get().strip() or cfg.ml_layout_model_path
        )
        cfg.ml_gnn_enabled = self.ml_gnn_enabled_var.get()
        cfg.ml_gnn_model_path = (
            self.ml_gnn_model_path_var.get().strip() or cfg.ml_gnn_model_path
        )
        self._set_config_scalar(
            cfg, "ml_gnn_hidden_dim", self.ml_gnn_hidden_dim_var, int
        )
        self._set_config_scalar(
            cfg, "ml_gnn_patience", self.ml_gnn_patience_var, int
        )


# ---------------------------------------------------------------------------
# LLM Configuration Dialog (absorbs "LLM Runtime" collapsible section)
# ---------------------------------------------------------------------------


class LLMConfigDialog(_SettingsDialog):
    """Settings → LLM Configuration… dialog."""

    def __init__(self, parent: tk.Widget, state: Any) -> None:
        super().__init__(parent, state, title="LLM Configuration", height=400)

    def _build_content(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(1, weight=1)
        row = 0

        ttk.Label(
            parent,
            text="LLM Runtime Configuration",
            font=("TkDefaultFont", 10, "bold"),
        ).grid(row=row, column=0, columnspan=3, sticky="w", pady=(0, 10))
        row += 1

        self.llm_provider_var = tk.StringVar()
        self._add_labeled_entry(
            parent,
            row=row,
            label="LLM provider",
            variable=self.llm_provider_var,
            help_text="Provider key (ollama, openai, anthropic).",
        )
        row += 1

        self.llm_model_var = tk.StringVar()
        self._add_labeled_entry(
            parent,
            row=row,
            label="LLM model",
            variable=self.llm_model_var,
            help_text="Model name sent to the configured provider.",
        )
        row += 1

        self.llm_api_base_var = tk.StringVar()
        self._add_labeled_entry(
            parent,
            row=row,
            label="LLM API base",
            variable=self.llm_api_base_var,
            help_text="Base URL for the provider endpoint.",
        )
        row += 1

        self.llm_policy_var = tk.StringVar()
        self._add_labeled_entry(
            parent,
            row=row,
            label="LLM policy",
            variable=self.llm_policy_var,
            help_text="Runtime policy enforced by the LLM client.",
        )
        row += 1

        self.llm_temperature_var = tk.StringVar()
        self._add_labeled_entry(
            parent,
            row=row,
            label="LLM temperature",
            variable=self.llm_temperature_var,
            help_text="Sampling temperature for LLM-backed decisions.",
        )
        row += 1

        self.llm_api_key_var = tk.StringVar()
        self._add_labeled_entry(
            parent,
            row=row,
            label="LLM API key",
            variable=self.llm_api_key_var,
            help_text="Optional API key for non-local providers.",
            show="*",
        )

    def _load_values(self) -> None:
        cfg = self.state.config
        self.llm_provider_var.set(cfg.llm_provider)
        self.llm_model_var.set(cfg.llm_model)
        self.llm_api_base_var.set(cfg.llm_api_base)
        self.llm_policy_var.set(cfg.llm_policy)
        self.llm_temperature_var.set(str(cfg.llm_temperature))
        self.llm_api_key_var.set(cfg.llm_api_key)

    def _apply_values(self) -> None:
        cfg = self.state.config
        cfg.llm_provider = self.llm_provider_var.get().strip() or cfg.llm_provider
        cfg.llm_model = self.llm_model_var.get().strip() or cfg.llm_model
        cfg.llm_api_base = self.llm_api_base_var.get().strip() or cfg.llm_api_base
        cfg.llm_policy = self.llm_policy_var.get().strip() or cfg.llm_policy
        cfg.llm_api_key = self.llm_api_key_var.get().strip()
        self._set_config_scalar(
            cfg, "llm_temperature", self.llm_temperature_var, float
        )


# ---------------------------------------------------------------------------
# General Settings Dialog
# ---------------------------------------------------------------------------


class GeneralSettingsDialog(_SettingsDialog):
    """Settings → General… dialog for app-level preferences."""

    def __init__(self, parent: tk.Widget, state: Any) -> None:
        super().__init__(parent, state, title="General Settings", height=300)

    def _build_content(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(1, weight=1)
        row = 0

        ttk.Label(
            parent,
            text="Application Preferences",
            font=("TkDefaultFont", 10, "bold"),
        ).grid(row=row, column=0, columnspan=2, sticky="w", pady=(0, 10))
        row += 1

        # Default tab visibility
        ttk.Label(
            parent,
            text="Default Tab Visibility",
            font=("TkDefaultFont", 9, "bold"),
        ).grid(row=row, column=0, columnspan=2, sticky="w", pady=(6, 2))
        row += 1

        self._show_database_var = tk.BooleanVar()
        ttk.Checkbutton(
            parent,
            text="Show Database Inspector tab by default",
            variable=self._show_database_var,
        ).grid(row=row, column=0, columnspan=2, sticky="w", pady=2)
        row += 1

        self._show_diagnostics_var = tk.BooleanVar()
        ttk.Checkbutton(
            parent,
            text="Show Diagnostics tab by default",
            variable=self._show_diagnostics_var,
        ).grid(row=row, column=0, columnspan=2, sticky="w", pady=2)
        row += 1

        ttk.Separator(parent, orient="horizontal").grid(
            row=row, column=0, columnspan=2, sticky="ew", pady=10
        )
        row += 1

        ttk.Label(
            parent,
            text="Additional settings will be added in future releases.",
            foreground="gray",
        ).grid(row=row, column=0, columnspan=2, sticky="w")

    def _load_values(self) -> None:
        self._show_database_var.set(
            self.state.tab_visibility.get("database", True)
        )
        self._show_diagnostics_var.set(
            self.state.tab_visibility.get("diagnostics", True)
        )

    def _apply_values(self) -> None:
        self.state.tab_visibility["database"] = self._show_database_var.get()
        self.state.tab_visibility["diagnostics"] = self._show_diagnostics_var.get()
