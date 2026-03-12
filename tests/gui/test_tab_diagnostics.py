import tkinter as tk
from pathlib import Path

from plancheck.config import GroupingConfig
from scripts.gui.gui import GuiState
from scripts.gui.tab_diagnostics import (LayoutModelSection,
                                         MLCalibrationSection,
                                         _build_ml_runtime_summary)


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


def test_build_ml_runtime_summary_reports_enabled_modes(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "stage1.pkl").write_text("s1", encoding="utf-8")
    (data_dir / "stage2.pkl").write_text("s2", encoding="utf-8")
    (data_dir / "drift.json").write_text("{}", encoding="utf-8")
    (data_dir / "candidate.pkl").write_text("cand", encoding="utf-8")

    cfg = GroupingConfig(
        ml_hierarchical_enabled=True,
        ml_model_path="data/stage1.pkl",
        ml_stage2_model_path="data/stage2.pkl",
        ml_drift_enabled=True,
        ml_drift_stats_path="data/drift.json",
        vocr_cand_ml_enabled=True,
        vocr_cand_classifier_path="data/candidate.pkl",
        ml_layout_enabled=True,
        ml_layout_model_path="models/layout-ft",
    )

    summary = _build_ml_runtime_summary(
        cfg,
        pending_corrections=60,
        db_present=True,
    )

    assert summary["Routing"] == "Hierarchical title refinement"
    assert summary["Stage 1 model"] == "Ready (stage1.pkl)"
    assert summary["Stage 2 model"] == "Ready (stage2.pkl)"
    assert summary["Drift detection"] == "Enabled (threshold 0.30, stats ready)"
    assert summary["Retrain readiness"] == "60/50 pending — recommended"
    assert summary["Candidate ML"] == "Enabled (candidate.pkl)"
    assert summary["Layout runtime"] == "models/layout-ft"


def test_calibration_section_owns_reliability_diagram_renderer() -> None:
    assert hasattr(MLCalibrationSection, "_draw_reliability_diagram")
