"""Record a breakable DUPLO 2x4 being struck on the baseplate, at 1920x1080, in one take.

Usage:

    python scripts/make_damage_video.py
    python scripts/make_damage_video.py --probe            # the physics only, no render
    python scripts/make_damage_video.py --quality draft --out /tmp/check.mp4

The arm is the striker: the closed jaws are driven at a wall of a breakable 2x4 standing on
the baseplate, three times, at 0.1, 0.3 and 0.6 m/s of commanded tool speed.  Nothing in the
video is staged - a piece comes away only when its weld passes ``F/F_limit + M/M_limit >= 1``
in :class:`~brick_builder.bricks.breakable.BreakableBricks`.

The take is real time and continuous, and it ends on three seconds of the aftermath, because
the release itself resolves in one 2 ms step and a 30 fps frame cannot hold it.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Iterator, Sequence

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from brick_builder.bricks import materials as mat
from brick_builder.bricks.snap_api import SnapParams
from brick_builder.bricks.structures import Placement
from brick_builder.control.pick_place import CONTROL_DT, ArmDriver, cell_world
from brick_builder.demo.video import (
    CamKey,
    CloseUp,
    Recorder,
    ReadoutSpec,
    Tick,
    quality_by_name,
    studio_look,
)
from brick_builder.scene.compose import SceneConfig, build_scene

TITLE = "Brick Builder Arm Robot"

SPEEDS = (0.1, 0.3, 0.6)
"""Commanded tool speeds at the wall [m/s] - the three the damage guide's collision case uses."""

BRICKS = ("struct_0", "struct_1", "struct_2")
"""One brick per pass, standing in a row on the plate's edge.  A struck brick shifts on its
studs, so a second pass at the same one would be a pass at a brick that has already moved; a
row of three keeps the only difference between the passes the speed."""

STRIKE_Z = 0.010
"""Height of the TCP above the plate's top face during a strike [m].

The fixed jaw's pad reaches 8 mm below the TCP, so this lands it on the bottom few millimetres
of the wall - about 15 mm below the roof, the lever the guide works its wall thresholds at.
The brick stands on the plate's edge row, so the struck face has nothing in front of it."""

STANDOFF = {0.1: 0.045, 0.3: 0.085, 0.6: 0.150}
"""Runway each pass needs to be at its speed when it arrives [m]: the servos take about
0.2 s to come up to 0.6 m/s.  The fastest pass therefore gets the brick nearest the base,
which is what keeps its runway inside the arm's reach."""

COLUMNS = {0.1: 5, 0.3: 3, 0.6: 1}
"""Plate column each pass's brick stands in, by speed."""

PUSH_S = 0.092
"""How long the commanded tool pose keeps travelling after it has reached the wall face [s].

The same rule for all three passes, so the only thing that differs between them is the speed:
the command runs on for 92 ms, which carries it 9 mm past the face at 0.1 m/s and 55 mm past it
at 0.6 m/s.  A tool that stopped at the surface would be a touch rather than a collision."""


# ==================================================================== the readouts ===

READOUTS = (
    ReadoutSpec(key="stress_mpa", label="Wall stress [MPa]", unit="", lo=0.0, hi=None,
                fmt="{:.0f}", references=(44.0, 70.0), plot=True),
    ReadoutSpec(key="force_n", label="Contact force [N]", unit="", lo=0.0, hi=None,
                fmt="{:.1f}"),
    ReadoutSpec(key="util_pct", label="Weld utilisation [%]", unit="", lo=0.0, hi=None,
                fmt="{:.0f}"),
    ReadoutSpec(key="state", label="Damage state", unit="", lo=0.0, hi=1.0, text=True),
    ReadoutSpec(key="pieces", label="Pieces released", unit="", lo=0.0, hi=None,
                fmt="{:.0f}"),
)

SUBTITLE = ("Brick damage | breakable DUPLO 2x4 | striker: arm tool | "
            "speeds: 0.1, 0.3, 0.6 m/s")


# ======================================================================== the scene ===


def damage_scene(quality=None) -> tuple:
    """Arm, baseplate and one breakable 2x4 seated on it, with the monitor at every step."""

    cfg = SceneConfig(
        held=None,
        placed=[Placement(h=4, w=2, x=COLUMNS[v], y=0, z=0, colour="red")
                for v in SPEEDS],
        breakable=True,
        damage_monitor_every=1,
        snap=SnapParams(snap_every=1),
        table=True,
    )
    scene = build_scene(cfg)
    if quality is not None:
        studio_look(scene.model, quality)
    return scene, dict(zip(SPEEDS, BRICKS))


def wall_face(scene, brick: str) -> tuple[float, float]:
    """World x of the brick and y of its struck wall's outer face: the 2-stud end."""

    spec = scene.cfg.spec
    bid = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_BODY, brick)
    centre = scene.data.xpos[bid]
    return float(centre[0]), float(centre[1]) - (4 * spec.pitch - spec.clearance) / 2.0


def readings(scene) -> dict:
    """What the monitor and the weld bookkeeping have on the row, this tick.

    Every number is the monitor's own running peak, because a rigid-body impact resolves in a
    single 2 ms step and a 30 fps frame would otherwise show nothing at all."""

    monitor = scene.monitor
    util = float(scene.breakable.peak_utilisation.max()) if scene.breakable else 0.0
    return {
        "stress_mpa": monitor.peak_wall_stress_Pa / 1e6,
        "force_n": max(monitor.damage.brick(n).peak_lateral_N for n in BRICKS),
        "util_pct": 100.0 * util,
        "state": monitor.state,
        "pieces": float(scene.breakable.n_released) if scene.breakable else 0.0,
    }


# ======================================================================= the motion ===


def strike_motion(scene, bricks: dict, *, lead_seconds: float = 3.0,
                  hold_seconds: float = 3.0) -> Iterator[Tick]:
    """One pass at each brick in the row, then the aftermath.  One tick per 50 Hz control step."""

    driver = ArmDriver(scene)
    grip = float(driver.grip_range[0])          # jaws closed: the tool is one compact block
    gripper = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_BODY, "gripper")
    z = scene.plate_top + STRIKE_Z
    contact_speed: dict[float, float] = {}
    per_strike: list[dict] = []

    def tool_speed() -> float:
        return float(np.linalg.norm(scene.data.cvel[gripper, 3:6]))

    def go(pos, phase: str, ticks: int) -> Iterator[Tick]:
        for _ in range(ticks):
            driver.step(pos, 0.0, grip)
            yield Tick(scene=scene, phase=phase, data=readings(scene))

    def sweep(pos_of, phase: str, ticks: int, watch: float | None) -> Iterator[Tick]:
        """Walk the commanded pose along, recording the tool's speed as contact starts.

        Contact is read off the monitor rather than off the contact list: the lateral load it
        keeps on each brick is a running peak measured every physics step, so the first tick on
        which it moves is the tick the tool arrived, however short the impact was."""

        load = readings(scene)["force_n"]
        for k in range(ticks):
            before = tool_speed()
            driver.step(pos_of(k), 0.0, grip)
            now = readings(scene)
            if watch is not None and watch not in contact_speed and now["force_n"] > load + 0.5:
                contact_speed[watch] = before
            load = max(load, now["force_n"])
            yield Tick(scene=scene, phase=phase, data=now)

    first_x, first_face = wall_face(scene, bricks[SPEEDS[0]])
    start0 = np.array([first_x, first_face - STANDOFF[SPEEDS[0]], z])
    yield from go(start0, f"Strike {SPEEDS[0]:.1f} m/s",
                  max(int(round(lead_seconds / CONTROL_DT)), 1))

    x = face = 0.0
    for speed in SPEEDS:
        phase = f"Strike {speed:.1f} m/s"
        standoff = STANDOFF[speed]
        x, face = wall_face(scene, bricks[speed])
        start = np.array([x, face - standoff, z])
        # Cross to this pass's runway well short of a quarter of a metre a second, so the
        # traverse is never itself a strike.
        yield from sweep(lambda k, s=start: s, phase,
                         int(round(max(standoff / 0.22, 0.8) / CONTROL_DT)), None)
        run = standoff + speed * PUSH_S
        ticks = max(int(round(run / (speed * CONTROL_DT))), 1)
        before = dict(readings(scene))
        yield from sweep(
            lambda k, s=start, v=speed: s + np.array([0.0, v * CONTROL_DT * (k + 1), 0.0]),
            phase, ticks, speed,
        )
        # The hold is watched too: at 0.6 m/s the tool is still catching its command up when
        # the runway ends, so the blow itself lands after the sweep.
        yield from sweep(lambda k, e=start + np.array([0.0, run, 0.0]): e, phase,
                         int(round(0.4 / CONTROL_DT)), speed)
        after = dict(readings(scene))
        per_strike.append({
            "speed_m_s": speed,
            "contact_speed_m_s": contact_speed.get(speed, float("nan")),
            "stress_MPa": after["stress_mpa"],
            "force_N": after["force_n"],
            "util_pct": after["util_pct"],
            "state": after["state"],
            "pieces": after["pieces"],
            "pieces_this_pass": after["pieces"] - before["pieces"],
        })

    # The release resolves in one step, so the take ends on the aftermath: the tool draws back
    # and the camera holds on the piece lying free of the brick.
    away = np.array([x, face - 0.055, z + 0.095])
    for _ in range(max(int(round(hold_seconds / CONTROL_DT)), 1)):
        driver.step(away, 0.0, grip)
        yield Tick(scene=scene, phase="Aftermath", data=readings(scene))

    strike_motion.report = {"strikes": per_strike, "ik_failures": driver.ik_failures}


# ========================================================================= the video ===

LOOKAT_DZ = -0.014
"""Where the camera is aimed, relative to the plate's top [m].  Below it, so the row of bricks
sits in the clear upper half of the frame instead of behind the data strip."""

DISTANCE = 0.25
AZIMUTH = 118.0
"""Round to the struck side: the tool comes at the bricks' far end walls, so a camera on the
other side would watch the blow through the brick."""

ELEVATION = -14.0


def damage_film(out: Path, *, quality_name: str = "hd1080", stills: int = 6,
                stills_dir: Path | None = None, lead_seconds: float = 3.0,
                hold_seconds: float = 3.0):
    quality = quality_by_name(quality_name)
    scene, bricks = damage_scene(quality)
    bid = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_BODY, bricks[0.6])
    centre = scene.data.xpos[bid].copy()
    lookat = (float(centre[0]) + 0.016, float(centre[1]), scene.plate_top + LOOKAT_DZ)

    # The three passes and their traverses come to about 4.9 s between the lead-in and the
    # hold; the stills are spread over that, so all of them land inside the take.
    length = lead_seconds + 4.9 + hold_seconds
    still_times = list(np.linspace(0.02 * length, 0.99 * length, stills)) if stills else []

    with Recorder(
        scene, quality=quality,
        shot=[CamKey(0.0, lookat, DISTANCE, AZIMUTH, ELEVATION)],
        title=TITLE, subtitle=SUBTITLE, readouts=READOUTS,
        pip_camera=lambda tick: CloseUp(lookat=(lookat[0], lookat[1],
                                                scene.plate_top + 0.008),
                                        distance=0.115, azimuth=AZIMUTH - 22.0,
                                        elevation=-10.0),
        pip_label=lambda phase: "Struck wall",
        phase_name=lambda phase: phase,
        title_seconds=lead_seconds, min_span=0.10, trace_seconds=1e6,
    ) as recorder:
        result = recorder.run(
            strike_motion(scene, bricks, lead_seconds=lead_seconds,
                          hold_seconds=hold_seconds),
            out, still_times=still_times, stills_dir=stills_dir,
        )
    return result, getattr(strike_motion, "report", {}), scene


# ========================================================================== driving ===


def probe(lead_seconds: float = 0.5, hold_seconds: float = 1.0) -> dict:
    """The same motion with nothing rendered: what the strikes measure."""

    scene, bricks = damage_scene()
    for _ in strike_motion(scene, bricks, lead_seconds=lead_seconds,
                           hold_seconds=hold_seconds):
        pass
    report = dict(getattr(strike_motion, "report", {}))
    report["events"] = [vars(e) for e in scene.breakable.events]
    report["final_state"] = scene.monitor.state
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="make_damage_video.py")
    parser.add_argument("--out", default="media_out/hd/"
                        "brick_builder_arm_robot_brick_damage_1080p.mp4")
    parser.add_argument("--quality", default="hd1080", choices=("draft", "hd1080"))
    parser.add_argument("--stills", type=int, default=6)
    parser.add_argument("--stills-dir", default=None)
    parser.add_argument("--lead", type=float, default=3.0)
    parser.add_argument("--hold", type=float, default=3.0)
    parser.add_argument("--probe", action="store_true")
    args = parser.parse_args(argv)

    if args.probe:
        report = probe()
        for row in report["strikes"]:
            print(f"  {row['speed_m_s']:.1f} m/s commanded, "
                  f"{row['contact_speed_m_s']:.3f} m/s at contact: "
                  f"{row['stress_MPa']:7.1f} MPa  {row['force_N']:6.1f} N  "
                  f"util {row['util_pct']:6.1f} %  {row['state']:8s}  "
                  f"{row['pieces']:.0f} released")
        print("events:", report["events"])
        print("ik failures:", report["ik_failures"])
        return 0

    out = Path(args.out)
    stills_dir = Path(args.stills_dir) if args.stills_dir else out.parent / "stills"
    result, report, scene = damage_film(
        out, quality_name=args.quality, stills=args.stills,
        stills_dir=stills_dir if args.stills else None,
        lead_seconds=args.lead, hold_seconds=args.hold,
    )
    print(f"master {result.master} {result.frames} frames {result.seconds:.1f}s "
          f"{result.width}x{result.height}")
    for row in report.get("strikes", []):
        print(f"  {row['speed_m_s']:.1f} m/s commanded, "
              f"{row['contact_speed_m_s']:.3f} m/s at contact: "
              f"{row['stress_MPa']:7.1f} MPa  {row['force_N']:6.1f} N  "
              f"util {row['util_pct']:6.1f} %  {row['state']:8s}  "
              f"{row['pieces']:.0f} released")
    for event in scene.breakable.events:
        print(f"  released {event.piece} at t={event.time:.3f}s "
              f"utilisation {event.utilisation:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
