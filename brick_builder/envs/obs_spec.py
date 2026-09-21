"""The tagged observation registry every BrickBuilder environment builds its spaces from.

Every entry a policy's *actor* half can see must name
its real-hardware source: a servo register, forward kinematics of the read joint positions, an
echo of a command we ourselves issued, a perception token, a skill-FSM state, or plain task
context (a goal the task hands the policy, not something measured). An entry with no source, or
with a privileged simulator-only source, cannot be an actor entry -- that is the mechanism that
keeps a policy from learning to rely on information the real robot could never give it. The
*critic* half may additionally see ``"sim"``-sourced ground truth.

This module holds one flat, importable registry (``register``/``get``/``all_entries``), plus the
small amount of bookkeeping every environment needs to turn a list of entry names into a
gymnasium ``Box`` and back: :class:`Layout` (name -> slice, used every tick to write into a flat
observation vector) and :func:`markdown_table` (used both by tests, so an untagged or renamed
entry cannot slip past them, and by ``docs/guide/04-environments.md``, so the documentation is
generated from the same registry the code uses rather than typed out by hand and left to drift).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import gymnasium as gym
import numpy as np

# ------------------------------------------------------------------------------- the registry ---

_SOURCE_RE = re.compile(r"^(register:.+|fk|command_echo|task|token|fsm|sim)$")
"""What a source tag is allowed to look like. ``register:<address/name>`` names a servo register
address the way design report 02 section 0 numbers them (e.g. ``register:56`` for
``Present_Position``); the others are the fixed vocabulary of section 1."""


@dataclass(frozen=True)
class ObsEntry:
    """One named slot in an observation vector.

    ``history`` is how many consecutive ticks are stacked (oldest first) into this entry's slot;
    section 1 says proprioceptive actor entries carry a 4-tick history, so the shared entries
    below use ``history=4`` and task-specific single-shot entries (a goal, a token, a phase flag)
    use the default of 1.
    """

    name: str
    size: int
    source: str
    scope: str = "actor"          # "actor" (the real robot could see this) or "critic" (sim only)
    history: int = 1
    note: str = ""

    def __post_init__(self) -> None:
        if self.size <= 0:
            raise ValueError(f"{self.name!r}: size must be positive, got {self.size}")
        if self.history <= 0:
            raise ValueError(f"{self.name!r}: history must be positive, got {self.history}")
        if self.scope not in ("actor", "critic"):
            raise ValueError(f"{self.name!r}: scope must be 'actor' or 'critic', got {self.scope!r}")
        if not self.source or not _SOURCE_RE.match(self.source):
            raise ValueError(
                f"{self.name!r}: source {self.source!r} is not one of register:<addr>, fk, "
                "command_echo, task, token, fsm, sim"
            )
        if self.scope == "actor" and self.source == "sim":
            raise ValueError(
                f"{self.name!r}: an actor entry cannot use the privileged 'sim' source -- "
                "the real robot has no way to read simulator ground truth"
            )

    @property
    def width(self) -> int:
        """Flattened length this entry occupies in an observation vector: ``size * history``."""
        return self.size * self.history


_REGISTRY: dict[str, ObsEntry] = {}


def register(entry: ObsEntry) -> ObsEntry:
    """Add ``entry`` to the shared registry and return it, so call sites can do
    ``FOO = register(ObsEntry(...))`` at module scope.

    Registering the same name twice is only an error if the definitions disagree -- re-importing
    a module that calls ``register`` at import time (which every task module does) must not
    explode the second time pytest imports it.
    """
    existing = _REGISTRY.get(entry.name)
    if existing is not None and existing != entry:
        raise ValueError(
            f"observation entry {entry.name!r} is already registered as {existing!r}; "
            f"cannot redefine it as {entry!r}"
        )
    _REGISTRY[entry.name] = entry
    return entry


def get(name: str) -> ObsEntry:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(f"no observation entry named {name!r}; call obs_spec.register() first") from None


def all_entries() -> dict[str, ObsEntry]:
    """A copy of the whole registry, name -> entry."""
    return dict(_REGISTRY)


def check_sources(names: list[str], scope: str | None = None) -> None:
    """Raise if any of ``names`` is unregistered, or (for ``scope="actor"``) is critic-only.

    This is the test hook spec section 1 asks for: "a test fails if an actor entry has no
    source". Every :class:`ObsEntry` validates its own source at construction time, so by the
    time it is in the registry it already has one; what is left to check here is that an
    *actor* observation never smuggles in a critic-only (``"sim"``-sourced) entry. A *critic*
    observation is unrestricted: it is meant to see at least what the actor sees, plus
    privileged extras, so a mix of actor- and critic-scoped names is exactly right for it.
    """
    for name in names:
        entry = get(name)
        if scope == "actor" and entry.scope != "actor":
            raise ValueError(f"actor observation includes non-actor entry {name!r} ({entry.scope})")


# ------------------------------------------------------------------------------------ layout ---


@dataclass(frozen=True)
class Layout:
    """Where each of ``names`` lands inside one flat observation vector."""

    names: tuple[str, ...]
    slices: dict[str, slice]
    size: int

    def pack(self, values: dict[str, np.ndarray]) -> np.ndarray:
        """Assemble the flat vector from a ``{name: array}`` mapping.

        Every name in :attr:`names` must be present in ``values`` with exactly ``entry.width``
        numbers (after flattening) -- a task that forgets to compute one of its own entries fails
        loudly here instead of silently observing zeros.
        """
        out = np.zeros(self.size, dtype=np.float32)
        for name in self.names:
            sl = self.slices[name]
            v = np.asarray(values[name], dtype=np.float32).reshape(-1)
            if v.size != sl.stop - sl.start:
                raise ValueError(
                    f"observation entry {name!r}: expected {sl.stop - sl.start} numbers, got {v.size}"
                )
            out[sl] = v
        return out


def make_layout(names: list[str]) -> Layout:
    """Build a :class:`Layout` for ``names`` in the given order. Raises on an unregistered name
    or on a name repeated twice (which would silently overwrite its own slice)."""
    seen: set[str] = set()
    slices: dict[str, slice] = {}
    offset = 0
    for name in names:
        if name in seen:
            raise ValueError(f"observation entry {name!r} listed twice")
        seen.add(name)
        entry = get(name)
        slices[name] = slice(offset, offset + entry.width)
        offset += entry.width
    return Layout(names=tuple(names), slices=slices, size=offset)


def make_space(names: list[str]) -> gym.spaces.Box:
    """The ``gymnasium.spaces.Box`` for a list of entry names, in the order given."""
    size = make_layout(names).size
    return gym.spaces.Box(low=-np.inf, high=np.inf, shape=(size,), dtype=np.float32)


# --------------------------------------------------------------------------------- doc tables ---


def markdown_table(names: list[str]) -> str:
    """A markdown table of ``names``' entries, for ``docs/guide/04-environments.md``.

    Generated straight from the registry so the documentation cannot describe a size, source or
    history that the code does not actually have; ``tests/envs`` asserts the guide page contains
    exactly this text for each environment's actor/critic observation, which is what "the docs
    cannot drift" means in practice.
    """
    lines = [
        "| name | numbers | history | source | scope | note |",
        "|---|---|---|---|---|---|",
    ]
    for name in names:
        e = get(name)
        note = e.note or ""
        lines.append(f"| `{e.name}` | {e.size} | {e.history} | `{e.source}` | {e.scope} | {note} |")
    return "\n".join(lines)


# ------------------------------------------------------------------- shared proprioceptive set ---
#
# Every ArmEnv exposes these: the six servos' own registers, forward kinematics of the read
# positions, and echoes of the pose/grip/compliance we ourselves last commanded. Nothing here
# depends on any particular task.

JOINT_POS = register(ObsEntry(
    "joint_pos", 6, "register:56 (Present_Position)", "actor", history=4,
    note="ServoBus.read().position, one number per joint in servo-ID order",
))
JOINT_VEL = register(ObsEntry(
    "joint_vel", 6, "register:58 (Present_Speed)", "actor", history=4,
    note="ServoBus.read().speed",
))
JOINT_LOAD = register(ObsEntry(
    "joint_load", 6, "register:60 (Present_Load)", "actor", history=4,
    note="ServoBus.read().load -- commanded PWM duty in [-1, 1], not a torque",
))
TCP_POS = register(ObsEntry(
    "tcp_pos", 3, "fk", "actor", history=4,
    note="ArmKinematics.fk() of the read joint_pos, position only",
))
TCP_ROT6 = register(ObsEntry(
    "tcp_rot6", 6, "fk", "actor", history=4,
    note="ArmKinematics.fk() of the read joint_pos: first two columns of the rotation matrix "
    "(continuous 6D rotation representation, Zhou et al. 2019)",
))
CMD_POSE = register(ObsEntry(
    "cmd_pose", 5, "command_echo", "actor",
    note="our own last commanded TCP target: x, y, z, tilt, spin",
))
CMD_GRIP = register(ObsEntry(
    "cmd_grip", 1, "command_echo", "actor",
    note="our own last commanded gripper joint angle",
))
CMD_COMPLIANCE = register(ObsEntry(
    "cmd_compliance", 2, "command_echo", "actor",
    note="our own last commanded c_vert, c_lat torque-limit fractions",
))

PROPRIO_ENTRIES: tuple[str, ...] = (
    "joint_pos", "joint_vel", "joint_load", "tcp_pos", "tcp_rot6",
    "cmd_pose", "cmd_grip", "cmd_compliance",
)
"""Every ``ArmEnv`` actor observation starts with these, in this order."""

# ------------------------------------------------------------------------------ critic-only ---

READ_STALE = register(ObsEntry(
    "read_stale", 6, "sim", "critic",
    note="per servo: this tick's sync-read reply was lost and the previous reading was reused",
))
WRITE_DROPPED = register(ObsEntry(
    "write_dropped", 1, "sim", "critic",
    note="this tick's sync-write was lost on the emulated bus (BusEmulator.tick().write_lost)",
))
BUS_TIMING = register(ObsEntry(
    "bus_timing", 2, "sim", "critic",
    note="this tick's command_latency_s, read_skew_s from BusEmulator",
))
IK_FAILED = register(ObsEntry(
    "ik_failed", 1, "sim", "critic",
    note="this tick's IK did not converge; the previous joint goal was kept",
))
DR_PARAMS = register(ObsEntry(
    "dr_params", 17, "sim", "critic",
    note="this episode's dynamics.* (8) and command.* (9) randomize.sample() draw, in RANGES order",
))

CRITIC_ONLY_ENTRIES: tuple[str, ...] = (
    "read_stale", "write_dropped", "bus_timing", "ik_failed", "dr_params",
)
"""Appended after the actor entries (mirrored) to build every ``ArmEnv`` critic observation."""
