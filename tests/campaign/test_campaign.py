"""The loop's bookkeeping: what each round measures, how a round is scored, and resuming.

No simulator here. Episodes are fabricated records, which is enough to check the summaries, the
CSV round trip, the columns each round asks for and that a finished phase is not run twice.
"""

from __future__ import annotations

import json

from brick_builder.campaign.campaign import (
    Campaign, CampaignConfig, _read_csv, _write_csv, summarize_records,
)
from brick_builder.campaign.tasks import BrickRecord, EpisodeRecord


def _brick(*, on_target: bool, lateral: float = 0.1) -> BrickRecord:
    return BrickRecord(index=0, cell=(4, 3), seated=on_target, at_target_cell=on_target,
                       connected=on_target, lateral_mm=lateral, depth_error_mm=0.0,
                       tilt_deg=0.0)


def _records(task: str, inverse: str, wins: int, n: int = 10, *, predictor: str = "analytic",
             round_index: int = 0) -> list[EpisodeRecord]:
    out = []
    for seed in range(n):
        record = EpisodeRecord(task=task, seed=seed, phase="evaluate", round_index=round_index,
                               inverse=inverse, predictor=predictor, inverse_version="v1",
                               predictor_version="v1", source="ideal",
                               median_tracking_mm=1.0 + 0.1 * seed,
                               median_prediction_mm=0.2 + 0.01 * seed,
                               damage_state="intact", vision_ticks=200, wall_s=30.0)
        record.bricks = [_brick(on_target=seed < wins, lateral=0.05 + 0.01 * seed)]
        record.ticks_per_step = {"Align": 12 + seed, "Approach": 30}
        if seed >= wins:
            record.failure = "seated on the wrong cell"
        out.append(record)
    return out


def _cfg(tmp_path, **kw) -> CampaignConfig:
    base = dict(name="test", root=str(tmp_path), rounds=2, episodes=1,
                tasks=("placement", "pick_and_place"), eval_seeds=tuple(range(4)))
    base.update(kw)
    return CampaignConfig(**base)


# ---------------------------------------------------------------------------------- summarising ---


def test_a_group_is_scored_with_its_wilson_interval() -> None:
    out = summarize_records(_records("placement", "analytic", 8))
    row = out["placement | analytic | F analytic"]
    assert row["seated_on_target"] == "8/10"
    assert row["success_rate"] == 0.8
    assert row["wilson_low"] < 0.8 < row["wilson_high"]
    assert row["episodes"] == 10
    assert row["failures"] == "seated on the wrong cell=2"
    assert row["ticks_align"] == 16.5
    assert row["lateral_mm_median"] > 0.0


def test_the_lateral_error_only_counts_bricks_that_landed() -> None:
    """A brick that missed its cell has no meaningful distance to it, so it is left out."""
    records = _records("placement", "analytic", 2)
    row = summarize_records(records)["placement | analytic | F analytic"]
    assert row["lateral_mm_max"] < 0.07


def test_the_two_inverses_and_the_two_prediction_rules_are_separate_groups() -> None:
    out = summarize_records(_records("placement", "analytic", 9)
                            + _records("placement", "learned", 6)
                            + _records("placement", "learned", 6, predictor="learned"))
    assert set(out) == {"placement | analytic | F analytic", "placement | learned | F analytic",
                        "placement | learned | F learned"}


def test_a_build_needs_every_brick_on_its_cell() -> None:
    record = EpisodeRecord(task="build_2", seed=0, phase="evaluate", round_index=0,
                           inverse="analytic", predictor="analytic", inverse_version="",
                           predictor_version="", source="ideal")
    record.bricks = [_brick(on_target=True), _brick(on_target=False)]
    row = summarize_records([record])["build_2 | analytic | F analytic"]
    assert row["seated_on_target"] == "0/1"
    assert row["bricks_on_target"] == "1/2"


# ------------------------------------------------------------------------------------ the files ---


def test_the_episode_csv_round_trips(tmp_path) -> None:
    records = _records("build_2", "learned", 7)
    for record in records:
        record.bricks.append(_brick(on_target=True, lateral=0.2))
    path = _write_csv(tmp_path / "evaluate.csv", records)
    back = _read_csv(path)
    assert len(back) == len(records)
    assert summarize_records(back).keys() == summarize_records(records).keys()
    assert (summarize_records(back)["build_2 | learned | F analytic"]["seated_on_target"]
            == summarize_records(records)["build_2 | learned | F analytic"]["seated_on_target"])


def test_a_finished_phase_is_not_run_again(tmp_path) -> None:
    campaign = Campaign(_cfg(tmp_path), log=lambda line: None)
    assert campaign._done("r1.perform") is None
    campaign._finish("r1.perform", episodes=4, wall_s=12.0)
    assert campaign._done("r1.perform")["episodes"] == 4

    again = Campaign(_cfg(tmp_path), log=lambda line: None)
    assert again._done("r1.perform")["episodes"] == 4
    assert json.loads(campaign.state_path.read_text())["phases"]["r1.perform"]["done"]


def test_the_campaign_never_points_at_the_shipped_weights_for_its_own_pair(tmp_path) -> None:
    campaign = Campaign(_cfg(tmp_path), log=lambda line: None)
    inverse, predictor, version = campaign.deployed_paths()
    assert inverse is None and predictor is None and version == ""
    assert campaign.weights.is_relative_to(campaign.dir)
    shipped, _ = campaign.shipped_paths()
    assert not shipped.is_relative_to(campaign.dir)


# ---------------------------------------------------------------------------------- the columns ---


def test_round_zero_measures_both_inverses_on_every_task(tmp_path) -> None:
    campaign = Campaign(_cfg(tmp_path), log=lambda line: None)
    columns = campaign.columns_for(0, "shipped", tmp_path / "inverse.npz",
                                   tmp_path / "predictor.npz")
    inverses = {c["inverse"] for c in columns}
    assert inverses == {"analytic", "learned"}
    full = [c for c in columns if set(c["tasks"]) == set(campaign.cfg.tasks)]
    assert len(full) == 2, "both inverses are measured on every task in the baseline round"
    paired = [c for c in columns if c["predictor"] == "analytic"]
    assert all(c["tasks"] == ("placement",) for c in paired)


def test_a_later_round_does_not_re_measure_the_fixed_analytic_baseline(tmp_path) -> None:
    campaign = Campaign(_cfg(tmp_path, rounds=3), log=lambda line: None)
    columns = campaign.columns_for(1, "r1", tmp_path / "inverse.npz", tmp_path / "predictor.npz")
    assert all(c["inverse"] == "learned" for c in columns)
    tasks = [t for c in columns for t in c["tasks"]]
    assert set(tasks) == set(campaign.cfg.tasks)


def test_the_last_round_rechecks_the_analytic_baseline_on_one_task(tmp_path) -> None:
    campaign = Campaign(_cfg(tmp_path, rounds=2), log=lambda line: None)
    columns = campaign.columns_for(2, "r2", tmp_path / "inverse.npz", tmp_path / "predictor.npz")
    analytic = [c for c in columns if c["inverse"] == "analytic"]
    assert analytic and analytic[0]["tasks"] == ("placement",)


def test_the_prediction_rules_are_paired_on_one_task(tmp_path) -> None:
    campaign = Campaign(_cfg(tmp_path), log=lambda line: None)
    columns = campaign.columns_for(1, "r1", tmp_path / "inverse.npz", tmp_path / "predictor.npz")
    paired = [c for c in columns if "placement" in c["tasks"]]
    assert {c["predictor"] for c in paired} == {"analytic", "learned"}
    assert [c["tasks"] for c in paired if c["predictor"] == "analytic"] == [("placement",)]
