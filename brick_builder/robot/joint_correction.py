"""Host-side outer correction on the joint angles the servos report, at the 50 Hz control rate.

The STS3215 firmware is a proportional law with nothing behind it.  It commands a duty proportional
to the position error, so it leaves a standing error wherever something pushes back -- gravity, dry
friction in the gear train (about 0.05 N*m against a 9.4 N*m/rad gain, a dead zone of roughly +-0.32
degrees per joint), and its own dead band, inside which it commands nothing at all.  Gravity has a
feedforward term already; the other two do not, and cannot have one, because where inside a dead
zone a joint stops depends on which way it came from.

What removes those is integral action, and the host is where it belongs: the firmware has an I term
(register 22) but changing it writes EPROM and re-tunes every joint at once, while a host-side trim
can be frozen the instant the arm leans on something.  This module is that trim -- one integrator
per joint, clamped, freezable, resettable -- and nothing else.  It reads the angles the bus reports
and writes the angles the bus is asked to hold; it never touches MuJoCo, a scene or a model.

It is only worth running once the encoder is where the hardware puts it.  On the output shaft the
trim sees the link, so what it corrects is what the tool actually does.  On the motor side it sees
a number that already tracks its command, and correcting it does nothing: measured over the same
workspace grid, 3.62 mm of tool error untrimmed and 3.87 mm trimmed.  Measure before switching it
on -- with nothing systematic left to remove, an integrator is a way of turning quantization and
read noise into motion (see :func:`measure_settling`).

:meth:`JointCorrection.update` is written to be transcribed into C99 as it stands: fixed-size
arrays, one explicit loop, no allocation, no numpy call inside the loop, no branch that depends on
anything but the numbers passed in.

The rest of the module is the measurement this correction exists for: :func:`measure_settling`
drives the arm to a grid of tool-down poses and reports how far the tool really ends up from where
it was sent, how much of that gap the encoders can see, and how long it takes to stop moving.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

CONTROL_HZ = 50.0
"""Rate the outer correction runs at [Hz] -- the project's own control tick."""

CONTROL_DT = 1.0 / CONTROL_HZ

TICKS_PER_REV = 4096
"""Encoder counts per output-shaft turn, so one count is 0.0879 degrees."""


# ------------------------------------------------------------------------------- the correction ---


@dataclass(frozen=True)
class JointCorrectionConfig:
    """Tuning for :class:`JointCorrection`.  Everything angular is in encoder counts."""

    ki: float = 1.0
    """Integral gain [1/s]: the correction grows by ``ki * error`` every second.

    Measured, not guessed.  Swept over the workspace grid in :func:`measure_settling` with the
    firmware dead band at 2 and at 7 counts: 0.5 is still creeping in at four seconds, 1.0 reaches
    the lowest mean error of anything tried (0.36 mm and 0.44 mm against 0.51 mm and 1.12 mm
    untrimmed), and 2.0 and 4.0 give most of that back as a limit cycle -- at 4.0 the tool buzzes
    four times as far as it does at 1.0.  The report has the whole sweep.
    """

    clamp_ticks: float = 40.0
    """Largest correction any one joint may accumulate [counts], i.e. 3.5 degrees.

    Sized off what the correction actually needs and what it must not be allowed to do.  Settled,
    the trim is small: over the workspace grid its median is 2.2 counts with no firmware dead band
    and 5.1 counts with a 7-count one, and the 95th percentile is under 6.  It reaches the clamp
    *transiently* on 5 of 49 poses, during the first half second of a step that crosses the whole
    workspace -- which is the windup this clamp exists to bound, and bounding it is why the trimmed
    arm arrives without overshooting.  Forty is also small enough that a runaway -- the arm pressing
    on something with the freeze input wrongly left clear -- cannot walk a joint anywhere dangerous
    before anyone notices.
    """

    deadzone_ticks: float = 1.0
    """Error below which nothing is integrated [counts].

    One count is the quantization floor: below it the reading carries no information about the
    sign of the error, so integrating it only produces a one-count hunt.
    """

    dt: float = CONTROL_DT
    """Seconds per :meth:`JointCorrection.update` call."""

    ticks_per_rev: int = TICKS_PER_REV


class JointCorrection:
    """One clamped integrator per joint, between the commanded joint angles and the servo bus.

    ::

        corr = JointCorrection(5)
        for tick in range(n):
            target = corr.update(q_desired, readings.position[:5], freeze=pressing)
            bus.write(np.concatenate([target, [grip]]))

    ``update`` returns ``desired + correction``.  A caller that adds its own feedforward term --
    every controller in this project carries a gravity one -- should send
    ``feedforward(desired) + corr.correction`` instead, so the integrator sees the plain tracking
    error and does not spend itself re-deriving a term that is already there.

    ``freeze`` is the input that makes this safe to run during a press or a grip: while it is set
    the correction is held exactly where it is, so an error the arm is *meant* to have -- a tool
    pushed into a brick, jaws squeezing past a face -- does not wind the trim up and then release
    it into a lurch when the contact ends.

    The gripper is normally left out (construct with ``n_joints=5``): its command is a squeeze past
    the object's face, so its position error is large, deliberate and permanent.
    """

    def __init__(self, n_joints: int = 5, cfg: JointCorrectionConfig = JointCorrectionConfig()) -> None:
        self.n = int(n_joints)
        self.cfg = cfg
        self.tick = 2.0 * math.pi / cfg.ticks_per_rev
        # Everything the update loop touches, in radians and as plain floats: a C99 port is
        # `float state[N]` and the same three branches.
        self._clamp = cfg.clamp_ticks * self.tick
        self._deadzone = cfg.deadzone_ticks * self.tick
        self._gain = cfg.ki * cfg.dt
        self.correction = np.zeros(self.n)
        self._target = np.zeros(self.n)

    def reset(self) -> None:
        """Drop the accumulated trim.  Call it whenever the arm is re-homed or an episode ends."""
        for i in range(self.n):
            self.correction[i] = 0.0
            self._target[i] = 0.0

    def update(self, desired, measured, freeze: bool = False) -> np.ndarray:
        """One 50 Hz tick: integrate ``desired - measured`` and return the angles to send.

        The anti-windup is the clamp itself.  The correction *is* the integrator state, so holding
        the state at its limit is exactly "stop integrating in the direction that is already
        saturated" -- there is no separate output to compare it against and no way for the two to
        disagree.  ``freeze`` skips the integration entirely rather than zeroing the input, so the
        trim the arm had before it touched anything is still there when it lets go.

        The returned array is the same buffer every call -- nothing here allocates, which is the
        point of writing it this way.  Copy it if you need to keep a tick's answer.
        """
        clamp = self._clamp
        deadzone = self._deadzone
        gain = self._gain
        state, out = self.correction, self._target
        for i in range(self.n):
            want = float(desired[i])
            state_i = state[i]
            if not freeze:
                error = want - float(measured[i])
                if error > deadzone:
                    state_i += gain * (error - deadzone)
                elif error < -deadzone:
                    state_i += gain * (error + deadzone)
                if state_i > clamp:
                    state_i = clamp
                elif state_i < -clamp:
                    state_i = -clamp
                state[i] = state_i
            out[i] = want + state_i
        return out


# ------------------------------------------------------------------ measuring what it is worth ---


@dataclass(frozen=True)
class SettlingSample:
    """One pose: where the tool was sent, where it stopped, and what the encoders thought."""

    target: np.ndarray
    error_mm: float
    """Distance from the commanded TCP to where the tool really ended up [mm]."""
    encoder_gap_mm: float
    """Distance from forward kinematics of the *read* joint angles to the true TCP [mm].

    This is the part of the error the robot cannot see, so it is the number that decides whether
    a correction on the encoders can do anything about it at all.
    """
    settle_s: float
    """Time until the tool stayed within 1 mm of where it finally stopped [s]."""
    chatter_mm: float
    """Peak-to-peak tool motion over the last second, once everything has settled [mm]."""


def workspace_grid(
    reaches=(0.13, 0.16, 0.19, 0.22),
    heights=(0.03, 0.05, 0.07),
    bearings=(-0.6, -0.3, 0.0, 0.3, 0.6),
) -> list[tuple[np.ndarray, float]]:
    """Tool-down TCP targets over the reachable workspace: ``(position, tool spin)`` pairs.

    ``reaches`` is the horizontal distance from the base axis and ``bearings`` the direction in
    the horizontal plane, so the grid sweeps the arm's forward arc rather than one line out of it.
    The tool spin is set equal to the bearing, which is the pose a real pick-and-place uses: the
    tool faces along the arm and ``wrist_roll`` stays near the middle of its range.

    The bounds are the tool-down workspace, measured rather than assumed.  With the tool pointing
    straight down this arm reaches from about 0.13 m to about 0.24 m and only up to about 0.09 m
    high: a target at 0.10 m has no tool-down solution at all, and one at 0.12 m has a solution
    that folds the fixed jaw into the shoulder link.  The plate sits at 0.17 m and the carry
    height puts the TCP around 0.06 m, so the grid brackets the work the arm really does.
    Unreachable and self-colliding combinations are dropped by :func:`measure_settling`.
    """
    return [
        (np.array([r * math.cos(a), r * math.sin(a), z]), a)
        for r in reaches
        for a in bearings
        for z in heights
    ]


def _arm_world(arm_cfg):
    """A bare arm on the table with nothing to collide with, plus its kinematics."""
    from brick_builder.robot.kinematics import ArmKinematics
    from brick_builder.robot.patch_so101 import build_arm_spec

    spec = build_arm_spec(arm_cfg)
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data, ArmKinematics(model)


def _gravity_feedforward(kin, q5: np.ndarray, kp5: np.ndarray) -> np.ndarray:
    """The command that puts the arm at ``q5``, not ``q5`` itself.

    The same two fixed-point passes ``ArmEnv`` and ``ArmDriver`` run, repeated here rather than
    imported so this measurement does not depend on an environment being constructible.
    """
    comp = q5.copy()
    for _ in range(2):
        comp = q5 + kin.gravity_torque(comp) / kp5
    return comp


def measure_settling(
    targets=None,
    *,
    arm_cfg=None,
    servo_cfg=None,
    correction: JointCorrectionConfig | None = None,
    seconds: float = 4.0,
    seed: int = 0,
) -> list[SettlingSample]:
    """Drive the arm to each target in turn and report how well it arrives.

    One control tick is one ``ServoBus.write``, ten 2 ms physics steps and one ``ServoBus.read``,
    which is the loop every other controller in this project runs.  The command is a step: the arm
    starts from the previous pose and is told the new one once, so what is measured is where a
    proportional servo stops rather than how well a trajectory generator creeps up on the answer.

    ``correction`` switches the outer trim on; ``None`` is the bare arm.  Targets that inverse
    kinematics cannot reach are skipped.
    """
    from brick_builder.robot.patch_so101 import ArmConfig
    from brick_builder.robot.servo_model import ServoBus, ServoConfig
    from brick_builder.scene.compose import tool_down_ik

    arm_cfg = ArmConfig() if arm_cfg is None else arm_cfg
    servo_cfg = ServoConfig() if servo_cfg is None else servo_cfg
    targets = workspace_grid() if targets is None else list(targets)

    model, data, kin = _arm_world(arm_cfg)
    probe = mujoco.MjData(model)          # scratch state for the self-collision check
    bus = ServoBus(model, data, cfg=servo_cfg, rng=np.random.default_rng(seed))
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tcp_pinch")
    kp5 = bus.kp[:5]
    grip = 0.5 * float(model.jnt_range[bus.joint_ids[5], 0] + model.jnt_range[bus.joint_ids[5], 1])
    corr = None if correction is None else JointCorrection(5, correction)

    n_ticks = int(round(seconds * CONTROL_HZ))
    trace = np.zeros((n_ticks, 3))
    out: list[SettlingSample] = []
    q_goal = np.array([0.0, -0.5, 0.9, 1.2, 0.0])
    for target, spin in targets:
        try:
            q5 = tool_down_ik(kin, target, spin, q0=q_goal)
        except RuntimeError:
            continue
        # Check the solve really landed on the target before blaming the servo for the gap, and
        # that the pose does not fold the tool into the arm -- near the inner edge of the
        # tool-down workspace the fixed jaw meets the shoulder link, and a jammed arm measures
        # the collision, not the servo.
        if float(np.linalg.norm(kin.fk(q5)[0] - target)) > 1e-4:
            continue
        probe.qpos[bus.qpos_adr[:5]] = q5
        mujoco.mj_forward(model, probe)
        if probe.ncon:
            continue
        q_goal = q5
        if corr is not None:
            corr.reset()
        feed = _gravity_feedforward(kin, q5, kp5)
        reading = bus.read()
        for tick in range(n_ticks):
            goal = feed.copy()
            if corr is not None:
                corr.update(q5, reading.position[:5])
                goal = goal + corr.correction
            bus.write(np.concatenate([goal, [grip]]))
            for _ in range(10):
                bus.substep()
                mujoco.mj_step(model, data)
            reading = bus.read()
            trace[tick] = data.site_xpos[site]

        final = trace[-1]
        settle = 0.0
        for tick in range(n_ticks):
            if float(np.linalg.norm(trace[tick] - final)) > 0.001:
                settle = (tick + 1) / CONTROL_HZ
        tail = trace[-int(CONTROL_HZ):]
        out.append(SettlingSample(
            target=np.asarray(target, dtype=float).copy(),
            error_mm=1e3 * float(np.linalg.norm(final - target)),
            encoder_gap_mm=1e3 * float(np.linalg.norm(kin.fk(reading.position[:5])[0] - final)),
            settle_s=settle,
            chatter_mm=1e3 * float(np.max(tail.max(axis=0) - tail.min(axis=0))),
        ))
    return out


def summarize(samples) -> dict:
    """Mean, median, 95th percentile and worst of each field, plus the sample count."""
    if not samples:
        return {"n": 0}
    out = {"n": len(samples)}
    for field in ("error_mm", "encoder_gap_mm", "settle_s", "chatter_mm"):
        values = np.array([getattr(s, field) for s in samples], dtype=float)
        out[field] = {
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "p95": float(np.percentile(values, 95)),
            "worst": float(values.max()),
            "min": float(values.min()),
        }
    return out


def random_servo(seed: int, level: float = 1.0):
    """An ``(ArmConfig, ServoConfig)`` pair drawn from the project's own randomization ranges.

    Uses :mod:`brick_builder.scene.randomize`'s sampler for the ``dynamics`` and ``command`` groups and
    reads the arm-side numbers back out of the sample, because :func:`randomize.apply` writes into
    a built :class:`~brick_builder.scene.compose.Scene` and this measurement runs on a bare arm.  The
    gain is carried as a P value so that ``servo_kp`` stays the only place the law is written down:
    the sampled ``dynamics.kp`` is inverted through it.
    """
    from brick_builder.robot.patch_so101 import ArmConfig, BAM_ERROR_GAIN, BAM_ERROR_GAIN_RATIO, BAM_KT, BAM_R
    from brick_builder.scene import randomize

    rng = np.random.default_rng(seed)
    sample = randomize.sample(rng, level=level, groups=("dynamics", "command"))
    vin = float(sample["dynamics.vin"])
    p_gain = float(sample["dynamics.kp"]) / (
        (BAM_KT / BAM_R) * vin * BAM_ERROR_GAIN * BAM_ERROR_GAIN_RATIO
    )
    arm_cfg = ArmConfig(
        p_gain=p_gain,
        vin=vin,
        torque_limit_nm=float(sample["dynamics.forcerange"]) * 7.4 / vin,
        backlash_deg=float(sample["dynamics.backlash_deg"]),
    )
    return arm_cfg, randomize.servo_config(sample)


# ------------------------------------------------------------------------------------ figure ---


def figure_settling_error(
    out_path: str | Path = "docs/media/robot/settling_error.png",
    *,
    correction: JointCorrectionConfig | None = None,
    seeds=(0, 1, 2),
    targets=None,
):
    """Tool settling error over the workspace grid, before and after the encoder correction.

    Two panels: the distribution of the final error, and how much of it the robot can see (forward
    kinematics of its own readings against the true tool).  The second is the point -- with the
    encoder on the motor side the gap is as large as the error itself, so nothing on the robot
    knows; with it on the output shaft the error collapses and what is left is visible.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from brick_builder.robot.servo_model import ServoConfig

    correction = JointCorrectionConfig() if correction is None else correction
    motor_side = ServoConfig(encoder_on_output=False)
    series = {}
    series["motor-side encoder"] = measure_settling(targets=targets, servo_cfg=motor_side)
    series["output-shaft encoder"] = measure_settling(targets=targets)
    series["output shaft, trimmed"] = measure_settling(targets=targets, correction=correction)
    randomized = []
    for seed in seeds:
        arm_cfg, servo_cfg = random_servo(seed)
        randomized += measure_settling(
            targets=targets, arm_cfg=arm_cfg, servo_cfg=servo_cfg, seed=seed
        )
    series["output shaft, randomized"] = randomized

    labels = list(series)
    colours = ["#b0413e", "#1f6feb", "#2f9e70", "#e0a458"]
    fig, axes = plt.subplots(1, 2, figsize=(16.0, 6.6), dpi=110)

    data = [[s.error_mm for s in series[k]] for k in labels]
    parts = axes[0].boxplot(data, widths=0.6, patch_artist=True,
                            tick_labels=[k.replace(", ", ",\n") for k in labels])
    for patch, colour in zip(parts["boxes"], colours):
        patch.set_facecolor(colour)
        patch.set_alpha(0.55)
    top = max(max(values) for values in data)
    for k, values in enumerate(data):
        spread = np.linspace(-0.18, 0.18, len(values))
        axes[0].scatter(np.full(len(values), k + 1) + spread, values, s=9, color=colours[k],
                        alpha=0.45, zorder=3)
        axes[0].annotate(f"mean\n{np.mean(values):.2f} mm", (k + 1, top),
                         textcoords="offset points", xytext=(0, 6), ha="center", fontsize=12)
    axes[0].set_ylabel("tool settling error [mm]", fontsize=13)
    axes[0].set_title("Tool Position Error", fontsize=14)
    axes[0].grid(axis="y", alpha=0.3)
    axes[0].tick_params(labelsize=12)
    axes[0].set_ylim(0.0, top * 1.35)

    for label, colour in zip(labels, colours):
        samples = series[label]
        axes[1].scatter([s.error_mm for s in samples], [s.encoder_gap_mm for s in samples],
                        s=30, color=colour, alpha=0.75, label=label)
    limit = max(top, max(s.encoder_gap_mm for k in labels for s in series[k])) * 1.1
    axes[1].plot([0.0, limit], [0.0, limit], color="0.6", lw=1.2, ls="--",
                 label="Equal Error")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("tool settling error [mm]", fontsize=13)
    axes[1].set_ylabel("Encoder Kinematics Error [mm]", fontsize=13)
    axes[1].set_title("Encoder Kinematics Error", fontsize=14)
    axes[1].legend(fontsize=11, loc="lower right")
    axes[1].grid(alpha=0.3, which="both")
    axes[1].tick_params(labelsize=12)

    fig.suptitle(
        f"Tool settling error over {len(series[labels[0]])} tool-down poses, 0.13-0.22 m reach "
        f"(trim: ki={correction.ki:g}/s, clamp {correction.clamp_ticks:g} counts)",
        fontsize=15,
    )
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path
