"""Servo system identification: step, ramp and chirp on one joint, fitted to the servo law.

Runs against the virtual bus today and against the arm the day it arrives -- the same code, the
same three tests, the same fit.  What it identifies:

``kp`` [N*m/rad]
    Slope of the static droop line.  The joint is commanded to hold at a set of angles whose
    gravity torque is known from the arm model, and the steady-state position error is read back
    over the bus; ``1/kp`` is the slope of error against torque and ``tau_friction/kp`` is the
    intercept.  The identified law predicts ``kp = (kt/R) * Vin * 0.166 * ratio * P``, so the fit
    is also a check on the firmware's ``0.166`` factor (guide 03).
``friction`` [N*m]
    Dry friction in the gear train, from half the difference between a hold approached from
    below and the same hold approached from above.
``peak_speed`` [rad/s]
    Fastest the joint moves during a step.  A *lower* bound on the firmware's target slew, not an
    estimate of it: the link's own inertia keeps it below the slew rate at these amplitudes.
``lost_motion`` [deg]
    Width of the hysteresis loop a slow triangular ramp traces out: gear free play, the firmware's
    dead band and twice the friction droop, together.  With ``kp`` and ``friction`` already fitted,
    the droop part is known and what is left is free play plus dead band.
``dead_band`` [encoder ticks] and ``backlash`` [deg] -- attempted, validated to fail
    ``fit_dead_band_ticks`` tries to split the ramp's hysteresis in two using ``Present_Load``:
    inside the firmware's dead band (registers 26/27) the firmware commands nothing at all, so the
    duty cycle should read exactly zero while the position does not move, which free play carries
    no such tell for (duty stays driven the whole time the output lags the motor).  ``--validate``
    shows the pocket is shorter than one 20 ms control tick at these gains, so a sync-read almost
    never lands inside it: the split is close to 100 % wrong on every randomized draw, even though
    ``dead_band + play`` (combined, not split) recovers well.  Kept and reported rather than
    deleted, because a validated negative result is still the answer to "which parameters are not
    identifiable from the bus and why" -- see the guide.  Fill the dead band from the self-test's
    direct register read (26/27) instead, and get free play by subtracting that from the combined
    figure.

Against the virtual bus every one of these has a known answer, so the script prints the recovered
value beside it and the error.  Guide 08 records what the recovery measured.

    python scripts/hardware/servo_sysid.py --virtual --joint shoulder_lift
    python scripts/hardware/servo_sysid.py --port /dev/ttyACM0 --joint shoulder_lift

Validation
----------
``--validate N`` draws ``N`` sets of the simulator's servo/joint parameters from
``brick_builder.scene.randomize`` (the ``dynamics`` group plus the firmware dead band field,
perturbed inside the same ranges training uses), runs the identification through the virtual bus
for each, and reports true/fitted/error for every parameter -- the evidence that this kit recovers
parameters it did not already know, not just the one nominal case:

    python scripts/hardware/servo_sysid.py --validate 5 --joint shoulder_lift
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from brick_builder.hardware.bus import Bus, BusSettings
from brick_builder.hardware.real_world import kinematic_arm
from brick_builder.hardware.sts3215 import DEG_PER_TICK, JOINT_NAMES, RAD_PER_TICK
from brick_builder.robot.patch_so101 import ArmConfig, BAM_FRICTION_BASE, servo_kp

HOLD_TICKS = 60
"""Control ticks per static hold -- 1.2 s, which is past the 0.5 s settling time of guide 03."""

SETTLE_TICKS = 20
"""Ticks at the end of a hold that the steady-state error is averaged over."""


@dataclass
class ServoFit:
    """What one joint's identification produced, in engineering units."""

    joint: str
    kp_nm_per_rad: float
    friction_nm: float
    peak_speed_rad_s: float
    lost_motion_deg: float
    deadband_and_play_deg: float
    """Lost motion with the fitted friction droop taken out: free play plus the firmware dead
    band, before the two are split by :func:`fit_dead_band_ticks`."""
    dead_band_ticks: float
    """Firmware position dead band (registers 26/27), from the zero-``Present_Load`` pocket at
    the ramp's turning point.  0 when no such pocket is found (dead band at or near 0)."""
    backlash_deg: float
    """What is left of :attr:`deadband_and_play_deg` after the dead band above is taken out --
    the TOTAL (not half-range) mechanical free play, clipped at 0."""
    droop_points: int
    droop_residual_deg: float

    def report(self, known: dict | None = None) -> str:
        rows = [
            ("kp", "N*m/rad", self.kp_nm_per_rad, None if known is None else known.get("kp")),
            ("friction", "N*m", self.friction_nm, None if known is None else known.get("friction")),
            ("peak step speed", "rad/s", self.peak_speed_rad_s, None),
            ("lost motion", "deg", self.lost_motion_deg, None),
            ("dead band + play", "deg", self.deadband_and_play_deg,
             None if known is None else known.get("deadband_and_play")),
            ("dead band", "ticks", self.dead_band_ticks,
             None if known is None else known.get("dead_band_ticks")),
            ("free play (total)", "deg", self.backlash_deg,
             None if known is None else known.get("backlash_deg")),
        ]
        lines = [f"system identification, joint {self.joint}", "",
                 "| quantity | unit | fitted | known | error |", "|---|---|---|---|---|"]
        for name, unit, value, truth in rows:
            if truth is None:
                lines.append(f"| {name} | {unit} | {value:.4g} | - | - |")
            else:
                err = 100.0 * abs(value - truth) / max(abs(truth), 1e-9)
                lines.append(f"| {name} | {unit} | {value:.4g} | {truth:.4g} | {err:.1f} % |")
        lines.append("")
        lines.append(f"droop line: {self.droop_points} points, "
                     f"residual {self.droop_residual_deg:.3f} deg")
        return "\n".join(lines)


# ----------------------------------------------------------------------------- the three tests ---


def hold(bus: Bus, goal: np.ndarray, ticks: int = HOLD_TICKS) -> np.ndarray:
    """Command one pose for ``ticks`` control ticks and return every reading."""
    out = np.zeros((ticks, len(goal)))
    for k in range(ticks):
        out[k] = bus.cycle(goal).position
    return out


def step_test(bus: Bus, home: np.ndarray, index: int, amplitude: float,
              ticks: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """One step on one joint.  Returns the commanded angle and the reading, per tick."""
    hold(bus, home, HOLD_TICKS)
    goal = home.copy()
    goal[index] = home[index] + amplitude
    pos = np.zeros(ticks)
    speed = np.zeros(ticks)
    for k in range(ticks):
        state = bus.cycle(goal)
        pos[k] = state.position[index]
        speed[k] = state.speed[index]
    return pos, speed


def ramp_test(bus: Bus, home: np.ndarray, index: int, amplitude: float,
              ticks: int = 240) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A slow triangular ramp: up to ``+amplitude``, back down, so the hysteresis shows.

    Returns commanded angle, read angle and ``Present_Load`` (PWM duty, signed), all per tick --
    the load is what :func:`fit_dead_band_ticks` reads the firmware's dead band off.
    """
    hold(bus, home, HOLD_TICKS)
    command = np.concatenate([
        np.linspace(0.0, amplitude, ticks // 2),
        np.linspace(amplitude, 0.0, ticks - ticks // 2),
    ])
    reading = np.zeros(command.size)
    load = np.zeros(command.size)
    goal = home.copy()
    for k, value in enumerate(command):
        goal[index] = home[index] + value
        state = bus.cycle(goal)
        reading[k] = state.position[index]
        load[k] = state.load[index]
    return home[index] + command, reading, load


def chirp_test(bus: Bus, home: np.ndarray, index: int, amplitude: float,
               f0: float = 0.2, f1: float = 4.0, seconds: float = 6.0,
               dt: float = 0.02) -> tuple[np.ndarray, np.ndarray]:
    """A linear frequency sweep.  Returns the commanded angle and the reading, per tick.

    Not fitted here -- the droop line identifies the gain far more directly -- but recorded,
    because the amplitude ratio against frequency is what a later second-order fit needs and it
    costs six seconds to collect.
    """
    n = int(seconds / dt)
    t = np.arange(n) * dt
    phase = 2.0 * math.pi * (f0 * t + 0.5 * (f1 - f0) / seconds * t ** 2)
    command = amplitude * np.sin(phase)
    reading = np.zeros(n)
    goal = home.copy()
    for k, value in enumerate(command):
        goal[index] = home[index] + value
        reading[k] = bus.cycle(goal).position[index]
    return home[index] + command, reading


# ----------------------------------------------------------------------------------- the fits ---


def fit_droop(torques: np.ndarray, errors: np.ndarray) -> tuple[float, float, float]:
    """Least squares ``error = tau / kp + offset``.  Returns (kp, offset, residual).

    ``errors`` is the *friction-free* droop -- the mean of a hold approached from below and the
    same hold approached from above, which cancels the dry-friction dead zone because the joint
    stops on opposite sides of it each way.  The offset that is left is a calibration error, not
    friction, and a non-zero one is worth reporting rather than absorbing.
    """
    design = np.stack([torques, np.ones_like(torques)], axis=1)
    (slope, offset), *_ = np.linalg.lstsq(design, errors, rcond=None)
    if slope <= 0.0:
        raise ValueError("droop slope is not positive: the joint did not sag with the load")
    kp = 1.0 / slope
    residual = float(np.sqrt(np.mean((design @ [slope, offset] - errors) ** 2)))
    return kp, float(offset), residual


def fit_lost_motion(command: np.ndarray, reading: np.ndarray) -> float:
    """Width of the hysteresis loop [rad]: the command gap between the two branches at mid-range.

    Both branches are interpolated onto the same reading axis and subtracted over the middle half
    of the travel, where neither end is being approached and both branches are straight.
    """
    half = command.size // 2
    up_cmd, up_read = command[:half], reading[:half]
    down_cmd, down_read = command[half:][::-1], reading[half:][::-1]
    lo = max(up_read.min(), down_read.min())
    hi = min(up_read.max(), down_read.max())
    if not hi > lo:
        return 0.0
    grid = np.linspace(lo + 0.25 * (hi - lo), hi - 0.25 * (hi - lo), 32)
    up = np.interp(grid, up_read, up_cmd)
    down = np.interp(grid, down_read, down_cmd)
    return float(np.mean(np.abs(up - down)))


def fit_dead_band_ticks(command: np.ndarray, load: np.ndarray, *,
                        load_tol: float = 5e-4) -> float:
    """Attempt at the firmware dead band [encoder ticks], from a zero-``Present_Load`` pocket.

    Inside registers 26/27 the firmware commands nothing at all (``servo_model.py``'s
    ``_update_overload`` sets ``duty = 0.0`` exactly), so in principle ``Present_Load`` reads zero
    for as long as the position error stays inside the dead band, which free play does not share
    (a joint taking up mechanical slack is still driven at the ordinary gain).  In practice
    (``--validate``) the pocket is shorter than one 20 ms control tick at the gains this arm runs
    at, so a sync-read almost never lands inside it and this comes back 0 far more often than the
    dead band actually is.  Kept for what it demonstrates -- read the module docstring's
    ``dead_band``/``backlash`` section before trusting a number out of it.

    Returns 0.0 when no pocket is found in this ramp's samples.
    """
    peak = int(np.argmax(command))
    quiet = np.flatnonzero(np.abs(load) <= load_tol)
    # Only the pocket touching the turning point is the dead band; a quiet tick anywhere else in
    # the ramp (there should not be one, but a noisy read is not impossible) is not it.
    quiet = quiet[np.abs(quiet - peak) <= max(len(command) // 8, 4)]
    if quiet.size == 0:
        return 0.0
    width_rad = float(command[quiet].max() - command[quiet].min())
    return max(0.5 * width_rad / RAD_PER_TICK, 0.0)


def identify(bus: Bus, joint: str = "shoulder_lift", *, kin=None, amplitude: float = 0.35,
             angles: np.ndarray | None = None) -> ServoFit:
    """Run all three tests on one joint and fit the servo law to them."""
    if kin is None:
        _, _, kin = kinematic_arm()
    index = JOINT_NAMES.index(joint)
    if index >= 5:
        raise ValueError("the gripper has no gravity torque to fit a droop line against")
    angles = np.linspace(-1.1, 0.2, 6) if angles is None else np.asarray(angles, dtype=float)

    torques, errors, halves = [], [], []
    for angle in angles:
        goal = np.zeros(6)
        both = []
        for approach in (-0.25, +0.25):
            goal[index] = float(angle) + approach
            hold(bus, goal, SETTLE_TICKS + 10)
            goal[index] = float(angle)
            reading = hold(bus, goal)[-SETTLE_TICKS:, index]
            # ``gravity_torque`` is the torque needed to *hold* the pose, and a proportional servo
            # only makes it by sitting that far off its target: commanded minus measured is
            # ``tau / kp``, which is why ``ArmDriver`` adds exactly that as a feed-forward.
            both.append(float(angle) - float(np.mean(reading)))
        torques.append(float(kin.gravity_torque(np.eye(5)[index] * float(angle))[index]))
        # Approaching from below and from above stops the joint on opposite sides of the dry
        # friction dead zone: the mean is the elastic droop and half the difference is the
        # friction, which is the only way to separate the two from position readings alone.
        errors.append(0.5 * (both[0] + both[1]))
        halves.append(0.5 * abs(both[0] - both[1]))
    torques = np.asarray(torques)
    errors = np.asarray(errors)
    sign = np.sign(np.mean(torques)) or 1.0
    kp, _, residual = fit_droop(sign * torques, sign * errors)
    friction = kp * float(np.mean(halves))

    home = np.zeros(6)
    home[index] = float(angles[len(angles) // 2])
    _, speed = step_test(bus, home, index, amplitude)
    rate_limit = float(np.max(np.abs(speed)))

    command, reading, load = ramp_test(bus, home, index, 0.25)
    lost_motion = fit_lost_motion(command, reading)
    droop_part = 2.0 * friction / kp
    deadband_and_play = max(lost_motion - droop_part, 0.0)
    dead_band_ticks = fit_dead_band_ticks(command, load)
    backlash_deg = max(math.degrees(deadband_and_play) - 2.0 * dead_band_ticks * DEG_PER_TICK, 0.0)
    return ServoFit(
        joint=joint,
        kp_nm_per_rad=kp,
        friction_nm=friction,
        peak_speed_rad_s=rate_limit,
        lost_motion_deg=math.degrees(lost_motion),
        deadband_and_play_deg=math.degrees(deadband_and_play),
        dead_band_ticks=dead_band_ticks,
        backlash_deg=backlash_deg,
        droop_points=len(angles),
        droop_residual_deg=math.degrees(residual),
    )


# ------------------------------------------------------------------------------ the known answer ---


def known_values(arm: ArmConfig = ArmConfig(), deadband_counts: float = 0.0) -> dict:
    """What the virtual bus is built from -- the numbers the fit has to recover."""
    play_deg = 2.0 * arm.backlash_deg
    deadband_deg = deadband_counts * 360.0 / 4096.0
    return {
        "kp": servo_kp(arm.p_gain, arm.vin),
        "friction": BAM_FRICTION_BASE,
        "deadband_and_play": play_deg + 2.0 * deadband_deg,
        "dead_band_ticks": deadband_counts,
        "backlash_deg": play_deg,
    }


def virtual_arm(arm: ArmConfig = ArmConfig(), *, seed: int = 0, sample: dict | None = None,
                servo_cfg=None):
    """A virtual bus with a bare arm behind it -- no table, no bricks, gravity on.

    A thin wrapper around :func:`brick_builder.calib.servo_bench.bare_virtual_arm`, which every
    bring-up script (this one, the self-test, the settling-error map) shares; see its docstring
    for what ``sample`` and ``servo_cfg`` do.
    """
    from brick_builder.calib.servo_bench import bare_virtual_arm

    return bare_virtual_arm(arm, seed=seed, sample=sample, servo_cfg=servo_cfg)


# ------------------------------------------------------------------------------- validation ---


def validate_recovery(joint: str = "shoulder_lift", n: int = 5, *, seed0: int = 1000) -> tuple[
        str, list[dict]]:
    """Draw ``n`` perturbed servo/joint parameter sets and recover them through the virtual bus.

    Each draw samples the ``dynamics`` randomization group (``dynamics.kp``,
    ``dynamics.frictionloss``, ``dynamics.backlash_deg``, ...) plus the firmware dead band
    (``command.error_deadband_ticks``) at ``level=1.0`` -- the full published range, the same one
    training draws from -- applies it to a bare-arm virtual bus with the command path otherwise
    clean (no delay jitter, no read noise: a validation run whose baseline already differed from
    its own truth by simulator noise could not say whether a recovery error came from the fit or
    from the noise), and runs :func:`identify` on it.  Returns a markdown table (one row per
    draw x parameter) and the raw per-draw records.
    """
    from brick_builder.robot.servo_model import ServoConfig
    from brick_builder.scene.randomize import sample_command, sample_dynamics

    records: list[dict] = []
    for k in range(int(n)):
        rng = np.random.default_rng(seed0 + k)
        dyn = sample_dynamics(rng, level=1.0)
        cmd = sample_command(rng, level=1.0)
        servo_cfg = ServoConfig(action_delay_jitter_s=0.0, read_noise_ticks=0.0,
                                vin=float(dyn["dynamics.vin"]),
                                error_deadband_ticks=float(cmd["command.error_deadband_ticks"]))
        true_backlash_deg = 2.0 * float(dyn["dynamics.backlash_deg"])
        true_dead_band_ticks = float(cmd["command.error_deadband_ticks"])
        truth = {
            "kp": float(dyn["dynamics.kp"]),
            "friction": float(dyn["dynamics.frictionloss"]),
            "deadband_and_play_deg": true_backlash_deg + 2.0 * true_dead_band_ticks * DEG_PER_TICK,
            "backlash_deg": true_backlash_deg,
            "dead_band_ticks": true_dead_band_ticks,
        }
        scene = vbus = None
        record = {"draw": k, "seed": seed0 + k, "truth": truth}
        try:
            scene, vbus = virtual_arm(sample=dyn, seed=seed0 + k, servo_cfg=servo_cfg)
            vbus.start()
            bus = Bus(vbus.port, BusSettings(timeout_s=0.5))
            try:
                bus.enable_torque(check=False)
                fit = identify(bus, joint, kin=scene.kin)
            finally:
                bus.close()
            record["fitted"] = {
                "kp": fit.kp_nm_per_rad, "friction": fit.friction_nm,
                "deadband_and_play_deg": fit.deadband_and_play_deg,
                "backlash_deg": fit.backlash_deg, "dead_band_ticks": fit.dead_band_ticks,
            }
        except Exception as exc:  # noqa: BLE001 -- one bad draw must not sink the other n-1
            record["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            if vbus is not None:
                vbus.stop()
        records.append(record)

    lines = [f"servo system identification, recovery over {n} randomized draws, joint {joint}", "",
             "| draw | parameter | true | fitted | error |", "|---|---|---|---|---|"]
    for record in records:
        if "error" in record:
            lines.append(f"| {record['draw']} | - | - | - | FAILED: {record['error']} |")
            continue
        for name in ("kp", "friction", "deadband_and_play_deg", "backlash_deg", "dead_band_ticks"):
            truth = record["truth"][name]
            fitted = record["fitted"][name]
            err = (100.0 * abs(fitted - truth) / max(abs(truth), 1e-9) if abs(truth) > 1e-9
                   else (0.0 if abs(fitted) < 1e-9 else float("inf")))
            err_str = "0.0 %" if truth == 0.0 and fitted == 0.0 else (
                f"{err:.1f} %" if math.isfinite(err) else "n/a (true = 0)")
            lines.append(f"| {record['draw']} | {name} | {truth:.4g} | {fitted:.4g} | "
                         f"{err_str} |")
    ok = [r for r in records if "error" not in r]
    lines.append("")
    if ok:
        for name in ("kp", "friction", "deadband_and_play_deg", "backlash_deg", "dead_band_ticks"):
            errs = [100.0 * abs(r["fitted"][name] - r["truth"][name]) / max(abs(r["truth"][name]), 1e-9)
                    for r in ok if not (r["truth"][name] == 0.0 and r["fitted"][name] == 0.0)]
            if errs:
                lines.append(f"mean |error|, {name}: {float(np.mean(errs)):.1f} % "
                             f"over {len(errs)} of {n} draws")
    lines.append(f"failed draws: {n - len(ok)} of {n}")
    lines.append("")
    lines.append(
        "note: `dead band` and `free play (total)` are one bus-behavioural attempt to split "
        "`dead band + play` in two (the zero-`Present_Load` pocket at the ramp's turning point, "
        "fit_dead_band_ticks) -- validated here to fail (the pocket is shorter than one 20 ms "
        "control tick at these gains, so the sync-read almost never lands inside it). `dead band "
        "+ play` (combined) is the quantity this kit actually recovers from bus behaviour; fill "
        "`arm.servo_dead_band_counts` from the self-test's direct register read (26/27) instead, "
        "and get free play by subtracting that from the combined figure -- see guide 08."
    )
    return "\n".join(lines) + "\n", records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", help="serial port of the servo bus")
    parser.add_argument("--virtual", action="store_true",
                        help="run against a virtual bus with a simulated arm behind it")
    parser.add_argument("--joint", default="shoulder_lift", choices=list(JOINT_NAMES[:5]))
    parser.add_argument("--baud", type=int, default=1_000_000)
    parser.add_argument("--out", help="write the fit (or, with --validate, the records) to this "
                                      "JSON file")
    parser.add_argument("--validate", type=int, metavar="N",
                        help="skip the single-fit run: draw N randomized servo/joint parameter "
                             "sets and report the recovery error of each, against the virtual bus")
    parser.add_argument("--seed0", type=int, default=1000, help="first --validate draw's seed")
    args = parser.parse_args(argv)
    if args.validate is not None:
        table, records = validate_recovery(args.joint, args.validate, seed0=args.seed0)
        print(table)
        if args.out:
            Path(args.out).write_text(json.dumps(records, indent=2) + "\n")
        return 0
    if not args.port and not args.virtual:
        parser.error("give --port, --virtual to run against the simulator, or --validate N")

    scene = vbus = None
    if args.virtual:
        scene, vbus = virtual_arm()
        vbus.start()
        bus = Bus(vbus.port, BusSettings(timeout_s=0.5))
    else:
        bus = Bus(args.port, BusSettings(baud=args.baud))
    try:
        bus.enable_torque(check=not args.virtual)
        fit = identify(bus, args.joint, kin=None if scene is None else scene.kin)
    finally:
        bus.close()
        if vbus is not None:
            vbus.stop()
    print(fit.report(known_values() if args.virtual else None))
    if args.out:
        Path(args.out).write_text(json.dumps(asdict(fit), indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
