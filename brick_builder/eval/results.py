"""Shared episode record, imported by envs, rl and eval so the three agree on what an episode outcome is."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class FailureClass(str, Enum):
    """Why an episode failed, labelled from simulator state (PLAN §6.4)."""

    WRONG_CELL = "wrong_cell"            # seated, but on the wrong studs
    PARTIAL_SEAT = "partial_seat"        # some studs engaged, brick wedged or tilted
    JAM = "jam"                          # no progress under load
    STRUCTURE_DAMAGE = "structure_damage"  # an existing connection broke or a neighbour moved
    SLIP = "slip"                        # brick moved in, or fell from, the gripper
    OVERLOAD_TRIP = "overload_trip"      # emulated servo overload protection fired
    TIMEOUT = "timeout"
    KINEMATIC = "kinematic"              # joint limit or unreachable target


@dataclass
class EpisodeResult:
    """One finished episode. `params` carries the sweep/randomization values the episode ran under.

    `peak_load` and `structure_intact` default to `None` ("not reported"), like
    `seat_depth_error` already did -- rl-eval-export-2. Neither used to have a `None` state:
    they defaulted to 0.0 / True, so an environment that silently stopped populating them (a
    bug, or a task for which the field is not meaningful) made `brick_builder.eval.report`'s safety
    checks read "0% load, 100% intact" and pass vacuously instead of flagging that nothing was
    actually measured. `brick_builder.eval.report.check_acceptance` now fails loud with a clear
    "not reported" row whenever either field is `None` on any episode in a batch, rather than
    silently treating missing data as a perfect score.

    `damage_state` and `peak_wall_stress_Pa` come from
    `brick_builder.bricks.damage.DamageMonitor`, which `brick_builder.scene.compose.build_scene`
    attaches by default. `damage_state` is one of
    `brick_builder.bricks.damage.STATES` -- `intact`, `marked`, `loosened`, `yielded`, `cracked`,
    `broken` -- and `peak_wall_stress_Pa` is the highest wall-bending stress any brick in the
    scene saw during the episode. Both default to `None` for the same reason as the two above: a
    scene with the monitor switched off reports nothing rather than reporting `intact`.
    """

    success: bool
    steps: int
    seat_depth_error: float | None = None   # metres; None when the task has no seat
    peak_load: float | None = None          # max |servo load| seen, as a fraction of full duty; None = not reported
    structure_intact: bool | None = None    # None = not reported
    failure: FailureClass | None = None
    ret: float = 0.0                        # undiscounted return
    params: dict = field(default_factory=dict)
    damage_state: str | None = None         # worst DamageModel state any brick reached; None = not reported
    peak_wall_stress_Pa: float | None = None  # highest wall-bending stress any brick saw; None = not reported
