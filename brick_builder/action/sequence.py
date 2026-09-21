"""The ideal sequence: what the twin did, frame by frame, and what it saw while doing it.

A single goal scene state cannot describe a task. There are endlessly many ways to end up with a
brick on a cell, and most of them knock the other bricks over on the way, so "match this picture"
is not an instruction. What the controller tracks instead is a *sequence*: the scene state the
twin saw at every vision tick and the action it took from there.

    (S*_0, a*_0), (S*_1, a*_1), ... , (S*_T, .)

`S*_k` is a scene state in the same representation the camera produces -- slot states, read off
the wrist view, so it is relative to the arm and carries no world coordinates. `a*_k` is what the
tool targets did during that tick: the five 20 ms commands the driver was given, and the net
change in tool position and yaw across them.

The recording is feed-forward and happens once, before the run, in a nominal world: the plate
where the task says it is, the brick where the supply row says. Nothing in it is corrected,
because in that world there is nothing to correct. Everything the real run does differently is
feedback on top of it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..perception.scene_model import SceneState


@dataclass
class Command:
    """One 20 ms write to the servo bus, exactly as the twin issued it."""

    pos: np.ndarray
    spin: float
    grip: float
    c_vert: float = 1.0
    c_lat: float = 1.0


@dataclass
class ReferenceFrame:
    """One vision tick of the ideal sequence: what was seen, and what was done next."""

    k: int
    t: float
    step: str
    """The step's title, as it appears on screen."""
    step_index: int
    """Steps are counted, not named, so a title may repeat without two steps merging."""

    state: SceneState
    """`S*_k` -- the twin's slot states for this tick. Carries its own optional label map, which
    the viewer reads and the controller does not."""

    bodies: dict = field(default_factory=dict)
    """What is identifiable in THIS frame, as `title -> body`. Per frame, not per step: a step
    can begin with the arm over the baseplate and end with it over a brick on the table, and a
    reference set fixed at the step's first frame goes on lining up on whatever was in view then.

    That was a real bug: the approach kept correcting itself against the baseplate all the way
    across to the supply area, so the brick it was reaching for was never the thing it lined up
    on."""

    label: object = None
    """Whatever a viewer asked to be kept for this tick -- a label image, say. The controller
    never looks inside it; `record_ideal` stores what its caller's hook returns and no more."""

    commands: list[Command] = field(default_factory=list)
    """`a*_k` as issued: the control-rate writes that took the twin from this tick to the next."""

    action: np.ndarray = field(default_factory=lambda: np.zeros(4))
    """`a*_k` as a number: net change of (tool x, y, z, yaw) over this tick."""

    @property
    def travel(self) -> float:
        """How far the tool target moved during this tick [m] -- how fast the reference is going,
        which is what decides how far behind the real arm is allowed to be."""
        return float(np.linalg.norm(self.action[:3]))


@dataclass
class IdealSequence:
    """The whole recording, plus what each step needs in order to be tracked."""

    frames: list[ReferenceFrame]
    bodies_by_step: dict[int, dict[str, str]]
    """Per step: the objects visible in that step's first reference frame, as `title -> body`.
    They are what the loop lines itself up on, and they change along the sequence -- the loop
    sees loose bricks on a table at the start and a baseplate at the end."""

    held_body: str | None = None
    """The brick in the hand during this step, if any. Never a thing to line up *on*: it rides
    with the camera. Its blob says how far off-centre it is sitting in the jaws."""

    held_by_step: dict[int, str | None] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.frames)

    def step_starts(self) -> dict[int, int]:
        """First frame index of each step."""
        out: dict[int, int] = {}
        for i, f in enumerate(self.frames):
            out.setdefault(f.step_index, i)
        return out

    def titles(self) -> list[str]:
        seen: dict[int, str] = {}
        for f in self.frames:
            seen.setdefault(f.step_index, f.step)
        return [seen[i] for i in sorted(seen)]
