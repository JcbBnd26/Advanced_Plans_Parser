import tkinter as tk

from plancheck.config import GroupingConfig
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
        )
    )

    tab.tocr_var.set(False)
    tab.ml_hierarchical_var.set(True)
    tab.ml_stage2_model_path_var.set("data/gui_stage2.pkl")

    cfg = tab._collect_config()

    assert cfg.enable_tocr is False
    assert cfg.ml_hierarchical_enabled is True
    assert cfg.ml_stage2_model_path == "data/gui_stage2.pkl"
    assert tab.state.config.ml_stage2_model_path == "data/gui_stage2.pkl"


def test_apply_config_updates_ml_controls() -> None:
    interp = tk.Tcl()
    tab = _make_tab(interp)

    cfg = GroupingConfig(
        enable_tocr=False,
        ml_enabled=False,
        ml_hierarchical_enabled=True,
        ml_model_path="data/stage1.pkl",
        ml_stage2_model_path="data/stage2.pkl",
    )

    tab._apply_config(cfg)

    assert tab.tocr_var.get() is False
    assert tab.ml_enabled_var.get() is False
    assert tab.ml_hierarchical_var.get() is True
    assert tab.ml_model_path_var.get() == "data/stage1.pkl"
    assert tab.ml_stage2_model_path_var.get() == "data/stage2.pkl"
