"""The loop's bookkeeping: what each round measures, how a round is scored, and resuming.

No simulator here. Episodes are fabricated records, which is enough to check the summaries, the
CSV round trip, the columns each round asks for and that a finished phase is not run twice.
"""

from __future__ import annotations

import json

from brick_builder.campaign.campaign import (
    Campaign, CampaignConfig, RoundRecord, _read_csv, _write_csv, summarize_records,
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


# -------------------------------------------------------------------------------- would it ship ---


def test_would_ship_reads_the_seated_key_the_scores_actually_carry(tmp_path) -> None:
    """`seated_fraction` was written against a `seated_on_target_cell` key the campaign's own
    score rows never set (they set `seated_on_target`), so the gate always compared 0/0 and
    passed. A round behind on its baseline has to read as not meeting it."""
    campaign = Campaign(_cfg(tmp_path, rounds=1), log=lambda line: None)
    baseline = summarize_records(_records("placement", "analytic", 9))
    campaign._put_round(RoundRecord(index=0, version="shipped", scores=baseline))

    behind = summarize_records(_records("placement", "learned", 5, predictor="learned",
                                        round_index=1))
    ship = campaign.would_ship(1, behind)
    assert ship["tasks"]["placement"]["meets"] is False
    assert ship["ship"] is False

    ahead = summarize_records(_records("placement", "learned", 10, predictor="learned",
                                       round_index=1))
    ship = campaign.would_ship(1, ahead)
    assert ship["tasks"]["placement"]["learned"] != ship["tasks"]["placement"]["analytic"]
    assert ship["tasks"]["placement"]["meets"] is True
    assert ship["ship"] is True


def test_would_ship_carries_the_round_s_own_version_before_the_record_is_saved(tmp_path) -> None:
    """`would_ship` used to read `self._round(index).version` -- the round's own record, not yet
    written when `score_round` calls it -- and got back the empty default. `score_round` now
    passes the version it already has in hand."""
    campaign = Campaign(_cfg(tmp_path, rounds=1), log=lambda line: None)
    baseline = summarize_records(_records("placement", "analytic", 9))
    campaign._put_round(RoundRecord(index=0, version="shipped", scores=baseline))

    scores = summarize_records(_records("placement", "learned", 9, predictor="learned",
                                        round_index=1))
    assert campaign.would_ship(1, scores)["version"] == ""
    assert campaign.would_ship(1, scores, version="r1")["version"] == "r1"


def test_redeploy_returns_the_round_s_own_version_not_the_shared_pointer(tmp_path) -> None:
    """A later round's redeploy moves the campaign's shared `deployed.json` pointer onto its own
    version; an earlier round whose own redeploy already finished must not read that shared
    pointer back and relabel itself with it."""
    campaign = Campaign(_cfg(tmp_path, rounds=2), log=lambda line: None)
    campaign._finish("r1.deploy", version="r1")
    campaign.weights.mkdir(parents=True, exist_ok=True)
    (campaign.weights / "deployed.json").write_text(json.dumps({"version": "r5"}))

    class _Result:
        version = "r5"

    assert campaign.redeploy(1, _Result()) == "r1"


# ------------------------------------------------------------------------- offline action error ---


def test_offline_action_error_only_reads_data_through_its_own_round(tmp_path, monkeypatch) -> None:
    """A resumed campaign's later rounds leave their `data/r*` directories on disk; an earlier
    round's offline metric must not be scored against ticks logged after it."""
    campaign = Campaign(_cfg(tmp_path, rounds=3), log=lambda line: None)
    for index in (1, 2, 3):
        directory = campaign.data / f"r{index}"
        directory.mkdir(parents=True)
        (directory / "ep_0.npz").write_bytes(b"")

    seen: dict = {}

    def fake_load(dirs):
        seen["dirs"] = sorted(dirs)
        return "the loaded data"

    def fake_metrics(data, inverse, predictor, *, seed):
        assert data == "the loaded data"
        return {"action_rmse": (0.1, 0.2, 0.3), "analytic_rms": (0.4, 0.5, 0.6),
                "n_held_out": 7}

    monkeypatch.setattr("brick_builder.action.learn.dataset.load", fake_load)
    monkeypatch.setattr("brick_builder.action.learn.evaluate.inverse_metrics", fake_metrics)
    monkeypatch.setattr("brick_builder.action.models.mlp.Mlp.load",
                        classmethod(lambda cls, path: path))

    class _Result:
        ticks = 1000
        inverse = "inverse.npz"
        predictor = "predictor.npz"

    out = campaign._offline_action_error(2, _Result())
    assert seen["dirs"] == [str(campaign.data / "r1"), str(campaign.data / "r2")]
    assert out["action_held_out_ticks"] == 7
