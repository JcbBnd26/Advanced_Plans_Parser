import tkinter as tk
from pathlib import Path

from plancheck.config import GroupingConfig
from scripts.gui import tab_pipeline as tab_pipeline_module
from scripts.gui.gui import GuiState
from scripts.gui.tab_pipeline import PipelineTab


class _DummyLabel:
    def __init__(self) -> None:
        self.kwargs: dict[str, str] = {}

    def configure(self, **kwargs) -> None:
        self.kwargs.update(kwargs)


def _make_tab(interp: tk.Tcl) -> PipelineTab:
    tab = PipelineTab.__new__(PipelineTab)
    tab.state = GuiState()
    tab.tocr_var = tk.BooleanVar(master=interp, value=True)
    tab.vocr_var = tk.BooleanVar(master=interp, value=True)
    tab.enable_vocr_candidates_var = tk.BooleanVar(master=interp, value=True)
    tab.ocr_reconcile_var = tk.BooleanVar(master=interp, value=True)
    tab.ocr_preprocess_var = tk.BooleanVar(master=interp, value=True)
    tab.skew_var = tk.BooleanVar(master=interp, value=False)
    tab.llm_checks_var = tk.BooleanVar(master=interp, value=False)
    tab.ocr_dpi_var = tk.StringVar(master=interp, value="300")
    tab.ml_enabled_var = tk.BooleanVar(master=interp, value=True)
    tab.ml_hierarchical_var = tk.BooleanVar(master=interp, value=False)
    tab.ml_model_path_var = tk.StringVar(
        master=interp,
        value="data/element_classifier.pkl",
    )
    tab.ml_stage2_model_path_var = tk.StringVar(
        master=interp,
        value="data/title_subtype_classifier.pkl",
    )
    tab.ml_relabel_confidence_var = tk.StringVar(master=interp, value="0.8")
    tab.ml_retrain_on_startup_var = tk.BooleanVar(master=interp, value=False)
    tab.ml_retrain_threshold_var = tk.StringVar(master=interp, value="50")
    tab.ml_feature_cache_enabled_var = tk.BooleanVar(master=interp, value=True)
    tab.ml_drift_enabled_var = tk.BooleanVar(master=interp, value=False)
    tab.ml_drift_threshold_var = tk.StringVar(master=interp, value="0.3")
    tab.ml_drift_stats_path_var = tk.StringVar(
        master=interp,
        value="data/drift_stats.json",
    )
    tab.ml_vision_enabled_var = tk.BooleanVar(master=interp, value=False)
    tab.ml_vision_backbone_var = tk.StringVar(master=interp, value="resnet18")
    tab.ml_embeddings_enabled_var = tk.BooleanVar(master=interp, value=False)
    tab.ml_embeddings_model_var = tk.StringVar(
        master=interp,
        value="all-MiniLM-L6-v2",
    )
    tab.ml_layout_enabled_var = tk.BooleanVar(master=interp, value=False)
    tab.ml_layout_model_path_var = tk.StringVar(
        master=interp,
        value="microsoft/layoutlmv3-base",
    )
    tab.ml_gnn_enabled_var = tk.BooleanVar(master=interp, value=False)
    tab.ml_gnn_model_path_var = tk.StringVar(
        master=interp,
        value="data/gnn_model.pt",
    )
    tab.ml_gnn_hidden_dim_var = tk.StringVar(master=interp, value="64")
    tab.ml_gnn_patience_var = tk.StringVar(master=interp, value="20")
    tab.ml_comparison_threshold_var = tk.StringVar(master=interp, value="0.005")
    tab.vocr_cand_ml_enabled_var = tk.BooleanVar(master=interp, value=False)
    tab.vocr_cand_classifier_path_var = tk.StringVar(
        master=interp,
        value="data/candidate_classifier.pkl",
    )
    tab.vocr_cand_ml_threshold_var = tk.StringVar(master=interp, value="0.3")
    tab.vocr_cand_gnn_prior_enabled_var = tk.BooleanVar(master=interp, value=False)
    tab.vocr_cand_gnn_prior_path_var = tk.StringVar(
        master=interp,
        value="data/gnn_candidate_prior.pt",
    )
    tab.vocr_cand_gnn_prior_blend_var = tk.StringVar(master=interp, value="0.25")
    tab.llm_provider_var = tk.StringVar(master=interp, value="ollama")
    tab.llm_model_var = tk.StringVar(master=interp, value="llama3.1:8b")
    tab.llm_api_base_var = tk.StringVar(
        master=interp,
        value="http://localhost:11434",
    )
    tab.llm_policy_var = tk.StringVar(master=interp, value="local_only")
    tab.llm_temperature_var = tk.StringVar(master=interp, value="0.1")
    tab.llm_api_key_var = tk.StringVar(master=interp, value="")
    tab._config_path_var = tk.StringVar(master=interp, value="")
    tab._stage1_model_status_label = _DummyLabel()
    tab._stage2_model_status_label = _DummyLabel()
    tab._stage2_path_entry = _DummyLabel()
    tab._stage2_browse_button = _DummyLabel()
    return tab


def test_collect_config_preserves_existing_ml_fields() -> None:
    interp = tk.Tcl()
    tab = _make_tab(interp)
    tab.state.set_config(
        GroupingConfig(
            ml_hierarchical_enabled=True,
            ml_stage2_model_path="data/custom_stage2.pkl",
            ml_layout_enabled=True,
            vocr_cand_ml_enabled=True,
        )
    )

    tab.tocr_var.set(False)
    tab.ml_hierarchical_var.set(True)
    tab.ml_stage2_model_path_var.set("data/gui_stage2.pkl")
    tab.ml_layout_enabled_var.set(True)
    tab.ml_drift_enabled_var.set(True)
    tab.ml_drift_threshold_var.set("0.45")
    tab.vocr_cand_ml_enabled_var.set(True)
    tab.vocr_cand_ml_threshold_var.set("0.42")
    tab.llm_provider_var.set("openai")

    cfg = tab._collect_config()

    assert cfg.enable_tocr is False
    assert cfg.ml_hierarchical_enabled is True
    assert cfg.ml_stage2_model_path == "data/gui_stage2.pkl"
    assert cfg.ml_layout_enabled is True
    assert cfg.ml_drift_enabled is True
    assert cfg.ml_drift_threshold == 0.45
    assert cfg.vocr_cand_ml_enabled is True
    assert cfg.vocr_cand_ml_threshold == 0.42
    assert cfg.llm_provider == "openai"
    assert tab.state.config.ml_stage2_model_path == "data/gui_stage2.pkl"


def test_apply_config_updates_ml_controls() -> None:
    interp = tk.Tcl()
    tab = _make_tab(interp)

    cfg = GroupingConfig(
        enable_tocr=False,
        enable_vocr_candidates=False,
        ml_enabled=False,
        ml_hierarchical_enabled=True,
        ml_model_path="data/stage1.pkl",
        ml_stage2_model_path="data/stage2.pkl",
        ml_retrain_on_startup=True,
        ml_drift_enabled=True,
        ml_drift_threshold=0.4,
        ml_layout_enabled=True,
        ml_layout_model_path="data/layout-model",
        vocr_cand_ml_enabled=True,
        vocr_cand_ml_threshold=0.55,
        llm_provider="openai",
    )

    tab._apply_config(cfg)

    assert tab.tocr_var.get() is False
    assert tab.enable_vocr_candidates_var.get() is False
    assert tab.ml_enabled_var.get() is False
    assert tab.ml_hierarchical_var.get() is True
    assert tab.ml_model_path_var.get() == "data/stage1.pkl"
    assert tab.ml_stage2_model_path_var.get() == "data/stage2.pkl"
    assert tab.ml_retrain_on_startup_var.get() is True
    assert tab.ml_drift_enabled_var.get() is True
    assert tab.ml_drift_threshold_var.get() == "0.4"
    assert tab.ml_layout_enabled_var.get() is True
    assert tab.ml_layout_model_path_var.get() == "data/layout-model"
    assert tab.vocr_cand_ml_enabled_var.get() is True
    assert tab.vocr_cand_ml_threshold_var.get() == "0.55"
    assert tab.llm_provider_var.get() == "openai"


def test_collect_run_validation_messages_warns_for_missing_runtime_files(
    tmp_path,
    monkeypatch,
) -> None:
    interp = tk.Tcl()
    tab = _make_tab(interp)
    monkeypatch.chdir(tmp_path)

    cfg = GroupingConfig(
        ml_enabled=True,
        ml_model_path="data/missing_stage1.pkl",
        ml_hierarchical_enabled=True,
        ml_stage2_model_path="data/missing_stage2.pkl",
        ml_drift_enabled=True,
        ml_drift_stats_path="data/missing_drift.json",
        vocr_cand_ml_enabled=True,
        vocr_cand_classifier_path="data/missing_candidate.pkl",
        vocr_cand_gnn_prior_enabled=True,
        vocr_cand_gnn_prior_path="data/missing_prior.pt",
        ml_layout_enabled=True,
        ml_layout_model_path="microsoft/layoutlmv3-base",
        enable_llm_checks=True,
        llm_provider="openai",
        llm_api_key="",
    )

    errors, warnings = tab._collect_run_validation_messages(cfg)

    assert len(errors) == 3
    assert len(warnings) == 4
    assert any("Stage 1 model file was not found" in item for item in warnings)
    assert any("Stage 2 model file was not found" in item for item in warnings)
    assert any("drift stats file was not found" in item for item in warnings)
    assert any("base LayoutLMv3 checkpoint" in item for item in warnings)
    assert any("candidate classifier file was not found" in item for item in errors)
    assert any("prior checkpoint was not found" in item for item in errors)
    assert any("no API key is configured" in item for item in errors)


def test_collect_run_validation_messages_blocks_empty_layout_checkpoint(
    tmp_path,
    monkeypatch,
) -> None:
    interp = tk.Tcl()
    tab = _make_tab(interp)
    monkeypatch.chdir(tmp_path)

    cfg = GroupingConfig(
        ml_enabled=False,
        ml_layout_enabled=True,
        ml_layout_model_path="",
    )

    errors, warnings = tab._collect_run_validation_messages(cfg)

    assert warnings == []
    assert errors == [
        "Layout inference is enabled but no LayoutLMv3 checkpoint is configured."
    ]


def test_confirm_save_with_validation_prompts_on_blocking_issues(
    tmp_path,
    monkeypatch,
) -> None:
    interp = tk.Tcl()
    tab = _make_tab(interp)
    monkeypatch.chdir(tmp_path)
    calls: list[tuple[str, str]] = []

    def _fake_askyesno(title: str, message: str) -> bool:
        calls.append((title, message))
        return False

    monkeypatch.setattr(tab_pipeline_module.messagebox, "askyesno", _fake_askyesno)

    cfg = GroupingConfig(
        enable_llm_checks=True,
        llm_provider="openai",
        llm_api_key="",
    )

    proceed = tab._confirm_save_with_validation(cfg)

    assert proceed is False
    assert calls
    assert calls[0][0] == "Save Config With Blocking Issues"
    assert "Blocking issues:" in calls[0][1]
    assert "no API key is configured" in calls[0][1]


def test_notify_loaded_config_validation_shows_review_warning(
    tmp_path,
    monkeypatch,
) -> None:
    interp = tk.Tcl()
    tab = _make_tab(interp)
    monkeypatch.chdir(tmp_path)
    calls: list[tuple[str, str]] = []

    def _fake_showwarning(title: str, message: str) -> None:
        calls.append((title, message))

    monkeypatch.setattr(
        tab_pipeline_module.messagebox,
        "showwarning",
        _fake_showwarning,
    )

    cfg = GroupingConfig(
        ml_enabled=True,
        ml_model_path="data/missing_stage1.pkl",
    )

    tab._notify_loaded_config_validation(cfg, source="Loaded config")

    assert calls
    assert calls[0][0] == "Loaded config Needs Review"
    assert "Warnings:" in calls[0][1]
    assert "Stage 1 model file was not found" in calls[0][1]
