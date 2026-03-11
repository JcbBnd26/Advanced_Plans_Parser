import tkinter as tk

from plancheck.config import GroupingConfig
from scripts.gui.gui import GuiState
from scripts.gui.tab_diagnostics import LayoutModelSection, MLCalibrationSection


def test_calibration_section_resolves_stage1_paths() -> None:
    interp = tk.Tcl()
    state = GuiState()
    state.set_config(
        GroupingConfig(
            ml_model_path="data/custom_stage1.pkl",
            ml_stage2_model_path="data/custom_stage2.pkl",
        )
    )

    section = MLCalibrationSection.__new__(MLCalibrationSection)
    section._state = state
    section._calibration_target_var = tk.StringVar(master=interp, value="stage1")

    name, model_path, jsonl_path = section._resolve_calibration_target()

    assert name == "Stage 1"
    assert str(model_path) == "data\\custom_stage1.pkl"
    assert str(jsonl_path) == "data\\training_data.jsonl"


def test_calibration_section_resolves_stage2_paths() -> None:
    interp = tk.Tcl()
    state = GuiState()
    state.set_config(
        GroupingConfig(
            ml_model_path="data/custom_stage1.pkl",
            ml_stage2_model_path="data/custom_stage2.pkl",
        )
    )

    section = MLCalibrationSection.__new__(MLCalibrationSection)
    section._state = state
    section._calibration_target_var = tk.StringVar(master=interp, value="stage2")

    name, model_path, jsonl_path = section._resolve_calibration_target()

    assert name == "Stage 2"
    assert str(model_path) == "data\\custom_stage2.pkl"
    assert str(jsonl_path) == "data\\training_data_stage2.jsonl"


def test_layout_section_starts_blank_for_base_checkpoint() -> None:
    section = LayoutModelSection.__new__(LayoutModelSection)
    section._state = type(
        "_State",
        (),
        {"config": GroupingConfig(ml_layout_model_path="microsoft/layoutlmv3-base")},
    )()

    assert section._initial_layout_model_value() == ""


def test_layout_section_accepts_custom_checkpoint() -> None:
    section = LayoutModelSection.__new__(LayoutModelSection)
    section._state = type(
        "_State",
        (),
        {"config": GroupingConfig(ml_layout_model_path="models/layout-ft")},
    )()

    assert section._initial_layout_model_value() == "models/layout-ft"
    assert section._is_invalid_layout_model("models/layout-ft") is False
    assert section._is_invalid_layout_model("microsoft/layoutlmv3-base") is True
