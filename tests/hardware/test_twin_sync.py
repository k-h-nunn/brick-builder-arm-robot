"""The twin log schema, the scene-state distance, and one real-versus-twin run over the bus."""

import numpy as np
import pytest

from brick_builder.hardware.twin_sync import (
    TWIN_LOG_COLUMNS,
    TwinLog,
    comparison_table,
    discrepancy_report,
    scene_distance,
)


def filled_log(n=12, source="virtual_bus"):
    log = TwinLog({"source": source, "perturbation": "nominal"})
    for k in range(n):
        log.append(
            tick=k, t=0.02 * k,
            joint_measured=np.full(6, 0.1 * k), joint_target=np.full(6, 0.1 * k),
            joint_rmse=1e-3 * k, tool_pos_err=1e-4 * k, prediction_rmse=2e-4,
            bus_rtt_s=0.004, tool_measured=np.zeros(3),
        )
    return log


def test_the_log_has_one_fixed_column_set_whatever_a_row_omits():
    log = filled_log()
    table = log.table()
    assert set(table) == set(TWIN_LOG_COLUMNS)
    for name, width in TWIN_LOG_COLUMNS.items():
        assert table[name].shape == ((len(log),) if width == 1 else (len(log), width))
    assert np.isnan(table["scene_distance"]).all()
    assert not np.isnan(table["joint_rmse"]).any()


def test_an_unknown_column_is_a_mistake_not_a_silent_drop():
    log = TwinLog()
    with pytest.raises(KeyError):
        log.append(tick=0, nonsense=1.0)


def test_the_log_round_trips_through_meta_json_and_ticks_npz(tmp_path):
    log = filled_log()
    log.write(tmp_path / "run")
    assert (tmp_path / "run" / "meta.json").exists()
    assert (tmp_path / "run" / "ticks.npz").exists()
    back = TwinLog.read(tmp_path / "run")
    assert len(back) == len(log)
    assert back.meta["source"] == "virtual_bus"
    for name in TWIN_LOG_COLUMNS:
        a, b = log.table()[name], back.table()[name]
        assert np.array_equal(a, b, equal_nan=True)


def test_the_scalar_columns_also_write_as_plain_csv(tmp_path):
    path = filled_log().to_csv(tmp_path / "run.csv")
    lines = path.read_text().splitlines()
    assert lines[0].split(",")[0] == "tick"
    assert len(lines) == 13


def test_the_summary_ignores_the_ticks_a_metric_was_not_measured_on():
    summary = filled_log().summary()
    assert summary["ticks"] == 12
    assert summary["joint_rmse_mean"] == pytest.approx(np.mean(1e-3 * np.arange(12)))
    assert np.isnan(summary["scene_distance_mean"])


def test_the_report_is_a_markdown_table_with_a_row_per_measured_metric(tmp_path):
    markdown, figure = discrepancy_report(filled_log(), tmp_path / "fig.png")
    assert "| metric | unit | mean | p95 |" in markdown
    assert "open-loop joint RMSE" in markdown
    assert "one-step prediction RMSE" in markdown
    assert figure.exists()


def test_the_comparison_table_prints_each_run_against_the_baseline():
    runs = {"nominal": filled_log(), "backlash 3x": filled_log()}
    table = comparison_table(runs)
    assert "| perturbation |" in table
    assert "backlash 3x" in table
    assert "(1.0x)" in table


def test_two_identical_scene_states_are_zero_apart_and_a_moved_one_is_not():
    from brick_builder.perception.ideal_scene_model import IdealSceneModel
    from brick_builder.hardware.real_world import SimulatedCamera
    from brick_builder.hardware.twin_sync import _study_scene, PERTURBATIONS

    scene = _study_scene(PERTURBATIONS[0])
    camera = SimulatedCamera(scene, model=IdealSceneModel())
    try:
        first = camera.observe(0.0)
        second = camera.observe(0.02)
        assert scene_distance(first, second) == pytest.approx(0.0, abs=1e-9)
    finally:
        camera.close()


@pytest.mark.slow
def test_a_perturbed_far_end_moves_a_metric_the_nominal_one_leaves_alone():
    from brick_builder.hardware.twin_sync import PERTURBATIONS, run_against_virtual_bus

    nominal = run_against_virtual_bus(PERTURBATIONS[0]).summary()
    gain = run_against_virtual_bus(
        [p for p in PERTURBATIONS if p.name == "servo gain P=32"][0]).summary()
    assert nominal["joint_rmse_mean"] < 1e-3          # under 0.06 deg: the twin tracks it
    assert gain["joint_rmse_mean"] > 5 * nominal["joint_rmse_mean"]
    assert gain["tool_pos_err_mean"] > 5 * nominal["tool_pos_err_mean"]
