"""The press bench: a virtual force gauge for the snap-fit model.

The bench is the small rig every acceptance number in PLAN section 6.3 is measured on. It is
deliberately not the robot: there is no arm, no servo model and no gripper, just a stiff ram on
a vertical slide that carries a brick down onto a baseplate and reports how much force that
took, sample by sample, against how deep the studs actually went.

The rig
-------

    world
     +- plate            a baseplate, welded to the world
     +- ram              slide along z, stiff position actuator          <- the force gauge
         +- ram_tilt     hinge about x, position actuator ("the wrist")  (only when tilt is on)
             +- brick    rigidly carried, gripped at `cfg.grip`

The gauge reading is the ram slide's actuator force, sign-flipped so that pressing down is
positive. Gravity is off by default, exactly as a bench force gauge is tared before a
measurement: it keeps the reading equal to the insertion force instead of the insertion force
plus a constant 0.12 N of brick weight. Tests that are *about* gravity (the stability and
creep checks) build their scenes with `build_stack_scene` instead, which has it on.

The wrist hinge matters for rocking and peeling and for nothing else, and it is driven rather
than free: a brick hanging off a frictionless pivot is a pendulum, and the first contact simply
spins it over. It is the wrist a real arm has, and `run` records the torque it is supplying
alongside the vertical force, because part of the point of rocking is to move load onto it.

Where the ram grips the brick is a free parameter (`cfg.grip`, in brick-local coordinates), and
it is the difference between a flat press and a rocking press. A flat press grips the middle of
the top face and pushes every stud in at once. A rocking press grips the leading bottom edge,
tips the brick by a couple of degrees, lets the leading row go in first, and then turns the
wrist back to level so the brick rolls down about that edge. Gripping on the bottom face is not
a detail: tilting about a point one brick-height above the sockets slides them 0.7 mm sideways,
which is more than the socket clearance, and the brick lands on the stud tops instead.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import mujoco
import numpy as np

from brick_builder.bricks.generator import add_baseplate, add_brick
from brick_builder.bricks.snap_api import SnapModel, SnapParams
from brick_builder.bricks.spec import DUPLO, BrickSpec

# Slab thickness the generator gives a baseplate, repeated here so the bench can work out
# where "flush" is without compiling first. Kept in step with generator._BASEPLATE_THICKNESS.
_PLATE_SLAB_FACTOR = 2.0


@dataclass(frozen=True)
class BenchConfig:
    """Everything about one bench run that is not the snap model itself."""

    nx: int = 2
    ny: int = 4
    plate_nx: int = 4
    plate_ny: int = 6
    spec: BrickSpec = DUPLO
    timestep: float = 0.002
    gravity: bool = False
    tilt: bool = False                 # add the passive float + hinge joints
    tilt_deg: float = 0.0              # initial hinge angle, about the rig's x axis
    grip: tuple[float, float, float] | None = None  # grip point in brick-local x, y, z;
                                       # None = centre of the brick's top face
    offset: tuple[float, float] = (0.0, 0.0)       # lateral misalignment of the whole rig
    yaw_deg: float = 0.0               # brick rotated about its own z
    start_gap: float = 0.0015          # brick bottom starts this far above the stud tips
    ram_kp: float = 2.0e4
    ram_armature: float = 0.05         # lead-screw inertia; lets the gauge be stiff at dt 2 ms
    tilt_damping: float = 0.02         # viscous damping on the tilt hinge, N·m·s/rad
    tilt_armature: float = 1.0e-3      # wrist rotor inertia
    tilt_kp: float = 200.0             # wrist position gain, N·m/rad
    friction: float = 0.35             # brick-on-brick sliding friction
    solref: tuple[float, float] = (0.004, 1.0)
    solimp: tuple[float, float, float] = (0.95, 0.999, 0.0005)


@dataclass
class BenchTrace:
    """What one bench run recorded, one row per physics step."""

    t: np.ndarray                  # (n,) seconds
    force: np.ndarray              # (n,) newtons, positive = ram pushing down
    depth: np.ndarray              # (n,) nominal insertion depth of the brick, metres
    depths: np.ndarray             # (n, n_socket) per-socket depth reported by the snap model
    n_engaged: np.ndarray          # (n,) sockets currently connected
    ram: np.ndarray                # (n,) ram slide position
    tilt: np.ndarray               # (n,) hinge angle, radians (0 when the rig has no hinge)
    wrist_torque: np.ndarray | None = None   # (n,) N·m at the tilt hinge, None without one
    warnings: int = 0              # total MuJoCo warning count raised during the run
    meta: dict = field(default_factory=dict)

    @property
    def peak_force(self) -> float:
        """Largest downward (pressing) force in the run."""

        return float(np.max(self.force)) if self.force.size else 0.0

    @property
    def peak_pull(self) -> float:
        """Largest upward (pulling) force in the run, as a positive number."""

        return float(-np.min(self.force)) if self.force.size else 0.0

    def peak_pull_above(self, depth: float) -> float:
        """Largest upward force recorded while the brick was still deeper than `depth`.

        A straight pull-off is not one number once the force law has a bump in it: the brick
        first has to break away from full depth, and then, on the way out, it has to be dragged
        back over whatever the curve does at shallower depths. `peak_pull` is the larger of the
        two; this is the break-away part on its own, which is the holding force of a seated
        connection and the number the acceptance test is about.
        """

        mask = self.depth >= depth
        return float(-np.min(self.force[mask])) if np.any(mask) else 0.0

    def at_depths(self, grid: np.ndarray) -> np.ndarray:
        """Force interpolated onto a depth grid, using only the part of the run that went in.

        The run is trimmed at the deepest point so the interpolation sees a monotone depth
        axis; a press that overshoots and bounces would otherwise fold back on itself.
        """

        stop = int(np.argmax(self.depth)) + 1
        d, f = self.depth[:stop], self.force[:stop]
        order = np.argsort(d)
        return np.interp(grid, d[order], f[order])


class PressBench:
    """Builds the rig, runs commanded ram motions, and hands back force-against-depth traces."""

    def __init__(self, snap: SnapModel, cfg: BenchConfig = BenchConfig()) -> None:
        self.cfg = cfg
        self.snap = snap
        spec = cfg.spec
        world = mujoco.MjSpec()
        opt = world.option
        opt.timestep = cfg.timestep
        opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
        opt.impratio = 10.0
        opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
        opt.iterations = 100
        opt.ls_iterations = 50
        opt.noslip_iterations = 0
        if not cfg.gravity:
            opt.gravity = [0.0, 0.0, 0.0]
        world.default.geom.solref = list(cfg.solref)
        world.default.geom.solimp = list(cfg.solimp) + [0.5, 2.0]
        world.default.geom.friction = [cfg.friction, 0.005, 0.0001]

        plate = add_baseplate(world, "plate", cfg.plate_nx, cfg.plate_ny, pos=(0.0, 0.0, 0.0))
        self.plate_top = spec.roof * _PLATE_SLAB_FACTOR
        self.stud_tip_z = self.plate_top + spec.stud_h

        grip = cfg.grip if cfg.grip is not None else (0.0, 0.0, spec.height)
        ox, oy, oz = grip
        theta = math.radians(cfg.tilt_deg)
        # Where the grip point has to start so that the lowest corner of the brick's bottom
        # face clears the stud tips by `start_gap`, whatever the tilt and the grip offset are.
        half_x, half_y = cfg.nx * spec.pitch / 2.0, cfg.ny * spec.pitch / 2.0
        corners = np.array(
            [
                [sx * half_x - ox, sy * half_y - oy, -oz]
                for sx in (-1.0, 1.0)
                for sy in (-1.0, 1.0)
            ]
        )
        rot = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, math.cos(theta), -math.sin(theta)],
                [0.0, math.sin(theta), math.cos(theta)],
            ]
        )
        lowest = float(np.min((corners @ rot.T)[:, 2]))
        z_attach = self.stud_tip_z + cfg.start_gap - lowest

        # The grip point sits `grip` away from the brick centre, so the rig has to be
        # shifted by the same amount to keep the brick itself over the plate's stud grid.
        ram = world.worldbody.add_body(
            name="ram", pos=[cfg.offset[0] + ox, cfg.offset[1] + oy, z_attach]
        )
        rj = ram.add_joint(name="ram_z", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[0, 0, 1])
        rj.armature = cfg.ram_armature
        ram.mass = 1e-4
        ram.inertia = [1e-9, 1e-9, 1e-9]
        ram.ipos = [0.0, 0.0, 0.0]
        ram.explicitinertial = True
        parent = ram
        if cfg.tilt:
            tiltb = ram.add_body(name="ram_tilt", pos=[0.0, 0.0, 0.0])
            tj = tiltb.add_joint(
                name="ram_tilt", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[1, 0, 0]
            )
            tj.damping = [cfg.tilt_damping, 0.0, 0.0]
            tj.armature = cfg.tilt_armature
            _tiny_inertial(tiltb)
            parent = tiltb

        yaw = math.radians(cfg.yaw_deg)
        quat = (math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0))
        brick = add_brick(
            world,
            "brick",
            cfg.nx,
            cfg.ny,
            "red",
            pos=(-ox, -oy, -oz),
            quat=quat,
            spec=spec,
            free=False,
            parent=parent,
        )

        act = world.add_actuator(name="ram_act")
        act.set_to_position(kp=cfg.ram_kp, kv=2.0 * math.sqrt(cfg.ram_kp * cfg.ram_armature))
        act.target = "ram_z"
        act.trntype = mujoco.mjtTrn.mjTRN_JOINT
        if cfg.tilt:
            tact = world.add_actuator(name="tilt_act")
            tact.set_to_position(
                kp=cfg.tilt_kp, kv=2.0 * math.sqrt(cfg.tilt_kp * cfg.tilt_armature)
            )
            tact.target = "ram_tilt"
            tact.trntype = mujoco.mjtTrn.mjTRN_JOINT

        self.bricks = [brick]
        self.plates = [plate]
        snap.declare(world, self.bricks, self.plates)
        self.spec_world = world
        self.model = world.compile()
        self.data = mujoco.MjData(self.model)
        self._ram_qadr = self.model.jnt_qposadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "ram_z")
        ]
        self._tilt_qadr = None
        if cfg.tilt:
            self._tilt_qadr = self.model.jnt_qposadr[
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "ram_tilt")
            ]
            self.data.qpos[self._tilt_qadr] = theta
        self.brick_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "brick")
        snap.bind(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self.data.ctrl[0] = self.data.qpos[self._ram_qadr]
        if cfg.tilt:
            self.data.ctrl[1] = theta
        self.snap_micros: float = 0.0
        # Distance from the grip point to the far end of the stud grid. When the grip sits
        # on the leading row, this is the lever the wrist rotation swings the trailing row
        # down through, and it sets the rotation rate that matches a given press speed.
        self.rock_lever = abs((cfg.ny - 1) / 2.0 * spec.pitch - grip[1])

    # -- running -----------------------------------------------------------

    def nominal_depth(self) -> float:
        """Insertion depth of the brick as a single number, from the brick's own pose.

        Measured the same way the snap model measures one socket: along the brick's z axis,
        from the bottom-face plane to the height of the plate's stud tips. At `stud_h` the
        brick is flush on the plate.
        """

        xpos = self.data.xpos[self.brick_body]
        z = self.data.xmat[self.brick_body].reshape(3, 3)[:, 2]
        return float(np.dot(np.array([0.0, 0.0, self.stud_tip_z]) - xpos, z))

    def seated(self, tolerance: float = 0.0003) -> bool:
        """True once every socket of the brick is within `tolerance` of full depth."""

        dep = self.snap.stud_depths("brick")
        return bool(dep.size and np.min(dep) >= self.cfg.spec.stud_h - tolerance)

    def run(
        self,
        command: Callable[[float], float],
        steps: int,
        stop: Callable[[], bool] | None = None,
        tilt_command: Callable[[float], float] | None = None,
    ) -> BenchTrace:
        """Step the sim `steps` times, with `command(t)` giving the ram's target position.

        `stop` is checked after every step; the run ends early when it returns True and the
        trace is trimmed. That is how a press run stops at the moment the brick seats instead
        of continuing to squash it into the plate, which would make the "peak press force" a
        measure of the rig's stiffness rather than of the snap model.

        `tilt_command(t)` drives the wrist hinge when the rig has one; left out, the wrist
        holds whatever angle it was last told to hold.
        """

        m, d = self.model, self.data
        t = np.empty(steps)
        force = np.empty(steps)
        depth = np.empty(steps)
        n_socket = len(self.snap.stud_depths("brick"))
        depths = np.empty((steps, n_socket))
        n_eng = np.empty(steps, dtype=np.int32)
        ram = np.empty(steps)
        tilt = np.zeros(steps)
        torque = np.zeros(steps)
        warn0 = int(d.warning.number.sum())
        micros = 0.0
        used = steps
        for i in range(steps):
            d.ctrl[0] = command(d.time)
            if tilt_command is not None and self._tilt_qadr is not None:
                d.ctrl[1] = tilt_command(d.time)
            mujoco.mj_step(m, d)
            t0 = _perf()
            self.snap.update(m, d)
            micros += _perf() - t0
            t[i] = d.time
            force[i] = -d.actuator_force[0]
            depth[i] = self.nominal_depth()
            dep = self.snap.stud_depths("brick")
            depths[i] = dep
            n_eng[i] = int(np.count_nonzero(dep > 0.0))
            ram[i] = d.qpos[self._ram_qadr]
            if self._tilt_qadr is not None:
                tilt[i] = d.qpos[self._tilt_qadr]
                torque[i] = d.actuator_force[1]
            if stop is not None and stop():
                used = i + 1
                break
        self.snap_micros = micros * 1e6 / max(used, 1)
        sl = slice(0, used)
        return BenchTrace(
            t=t[sl],
            force=force[sl],
            depth=depth[sl],
            depths=depths[sl],
            n_engaged=n_eng[sl],
            ram=ram[sl],
            tilt=tilt[sl],
            wrist_torque=torque[sl],
            warnings=int(d.warning.number.sum()) - warn0,
            meta={"snap_us_per_step": self.snap_micros, "steps": used},
        )

    def press(
        self,
        speed: float = 0.002,
        travel: float | None = None,
        until_seated: bool = True,
    ) -> BenchTrace:
        """Drive the ram down at `speed` m/s for up to `travel` metres of commanded descent.

        With `until_seated` (the default) the run ends as soon as every socket is at full
        depth, so `trace.peak_force` is the force it took to get the brick on and nothing else.
        """

        start = float(self.data.qpos[self._ram_qadr])
        t0 = float(self.data.time)
        if travel is None:
            travel = self.cfg.start_gap + 2.0 * self.cfg.spec.stud_h
        steps = int(math.ceil(travel / speed / self.cfg.timestep))
        stop = self.seated if until_seated else None
        return self.run(lambda t: start - speed * (t - t0), steps, stop)

    def press_rocking(
        self,
        speed: float = 0.002,
        trigger_depth: float | None = None,
        travel: float | None = None,
    ) -> BenchTrace:
        """Lower the tilted brick, then roll it flat about the row of studs that went in first.

        Two phases, and they are what a robot arm actually does:

        1. come straight down with the wrist holding the commanded tilt, until the leading row
           of sockets is `trigger_depth` onto its studs;
        2. keep descending at the same speed while rotating the wrist back to level at
           `speed / lever` rad/s, which is the rate that keeps the leading row exactly where it
           is - so it becomes the hinge and every other row comes down around it.

        During phase 2 the tool is a full brick length from that hinge while the studs it is
        still pushing in average half that, so the same insertion work needs roughly half the
        vertical force. The wrist torque that buys this is reported in the trace as well, and
        it is small: a fraction of a newton-metre against the servo's 1.9 N·m.
        """

        cfg = self.cfg
        if self._tilt_qadr is None:
            raise RuntimeError("press_rocking() needs a bench built with tilt=True")
        spec = cfg.spec
        theta0 = math.radians(cfg.tilt_deg)
        overhang = spec.pitch / 2.0          # bottom face beyond the outer row of sockets
        if trigger_depth is None:
            trigger_depth = spec.stud_h - overhang * math.tan(theta0) - 0.0002
        start = float(self.data.qpos[self._ram_qadr])
        t0 = float(self.data.time)
        rate = speed / max(self.rock_lever, 1e-6)
        state = {"t_rock": None, "z_rock": start, "t_flat": None}

        def phase(t: float) -> int:
            if state["t_rock"] is None:
                if float(np.max(self.snap.stud_depths("brick"))) >= trigger_depth:
                    state["t_rock"] = t
                    state["z_rock"] = float(self.data.qpos[self._ram_qadr])
                return 1
            if state["t_flat"] is None:
                if theta0 - rate * (t - state["t_rock"]) <= 0.0:
                    state["t_flat"] = t
                else:
                    return 2
            return 3

        def ram_cmd(t: float) -> float:
            ph = phase(t)
            if ph == 1:
                return start - speed * (t - t0)
            if ph == 2:
                return state["z_rock"]
            return state["z_rock"] - speed * (t - state["t_flat"])

        def tilt_cmd(t: float) -> float:
            if state["t_rock"] is None:
                return theta0
            return max(0.0, theta0 - rate * (t - state["t_rock"]))

        if travel is None:
            travel = cfg.start_gap + 2.0 * spec.stud_h
        steps = int(math.ceil(travel / speed / cfg.timestep))
        steps += int(math.ceil(abs(theta0) / max(rate, 1e-9) / cfg.timestep))
        return self.run(ram_cmd, steps, self.seated, tilt_cmd)

    def peel(self, rate_deg: float = 1.0, angle_deg: float = 2.0) -> BenchTrace:
        """Rotate a seated brick off its studs about the grip point and log the wrist torque.

        The ram holds its height while the wrist turns, so the brick lifts progressively: the
        row nearest the grip barely moves and the far row travels the most. Every row slides at
        the same pull-off force once it is moving, so the moment the wrist has to supply is
        `sum over studs of f_clutch * (1 - beta) * lever`, and the peak of `wrist_torque` is
        the number to compare that against.
        """

        if self._tilt_qadr is None:
            raise RuntimeError("peel() needs a bench built with tilt=True")
        theta0 = float(self.data.qpos[self._tilt_qadr])
        t0 = float(self.data.time)
        target = float(self.data.ctrl[0])
        rate = math.radians(rate_deg)
        steps = int(math.ceil(math.radians(angle_deg) / rate / self.cfg.timestep))
        return self.run(
            lambda t: target,
            steps,
            None,
            lambda t: theta0 + rate * (t - t0),
        )

    def pull(self, speed: float = 0.002, travel: float = 0.006) -> BenchTrace:
        """Drive the ram back up at `speed` m/s for `travel` metres."""

        start = float(self.data.qpos[self._ram_qadr])
        t0 = float(self.data.time)
        steps = int(math.ceil(travel / speed / self.cfg.timestep))
        return self.run(lambda t: start + speed * (t - t0), steps)

    def hold(self, duration: float) -> BenchTrace:
        """Keep the ram where it is for `duration` seconds."""

        target = float(self.data.ctrl[0])
        steps = int(math.ceil(duration / self.cfg.timestep))
        return self.run(lambda t: target, steps)


def _tiny_inertial(body) -> None:
    body.mass = 1e-5
    body.inertia = [1e-9, 1e-9, 1e-9]
    body.ipos = [0.0, 0.0, 0.0]
    body.explicitinertial = True


def _perf() -> float:
    from time import perf_counter

    return perf_counter()


# ---------------------------------------------------------------------------
# Predicted curves, standard set-ups, and multi-brick scenes
# ---------------------------------------------------------------------------


def predicted_press_force(
    depth: np.ndarray, n_studs: int, params: SnapParams, stud_h: float
) -> np.ndarray:
    """`n * f_clutch * (1 + beta) * g(d)` - the curve the bench is supposed to reproduce."""

    from brick_builder.bricks.snap_api import engagement

    return n_studs * params.f_clutch * (1.0 + params.beta) * engagement(depth, params, stud_h)


def quasi_static_band(params: SnapParams, stud_h: float) -> tuple[float, float]:
    """The stretch of the stroke over which a measured press may be compared with `g(d)`.

    `predicted_press_force` is a quasi-static curve: it says what the ram has to hold once the
    studs are sliding steadily. Two of the three curve shapes start with something that is not
    quasi-static at all, and the rig reproduces it honestly rather than smoothing it away:

    - "plateau" steps from nothing to the full clutch force the instant a stud engages, and the
      ram's own stiffness takes about 1.5 mm of descent to stop ringing about the step;
    - "detent" has a bump the ram has to push through, and the moment it is through, the energy
      stored in the ram lets the brick snap forward - the click of a real brick. The force dips
      (it even goes negative for a moment) until the descent catches up, about three bump
      widths past the bump.

    Everything outside those windows is steady sliding and tracks the curve to a fraction of a
    percent. The upper end is 93 % of full depth for every shape, because a press run stops the
    instant the brick is home.
    """

    hi = 0.93 * stud_h
    if params.curve == "plateau":
        return params.engage_depth + 0.0015, hi
    if params.curve == "detent":
        return params.detent_depth + 3.0 * params.detent_width, hi
    return params.engage_depth, hi


def measure_rocking_ratio(
    params: SnapParams,
    tilt_deg: float = 2.0,
    cfg: BenchConfig | None = None,
) -> dict:
    """Press one brick flat and once rocked, and report the peak forces and their ratio.

    Both runs use the same brick, the same studs, the same gauge and the same snap parameters;
    only the tool's grip and its motion differ. The ratio is bench test 4, and what it comes
    out at is a property of `params.curve` - see the docstring of `snap_api.engagement`.
    """

    from brick_builder.bricks.snap import ColletSnap

    flat_bench = PressBench(ColletSnap(params), cfg or BenchConfig())
    flat = flat_bench.press()
    rock_bench = PressBench(ColletSnap(params), rocking_config(tilt_deg=tilt_deg))
    rock = rock_bench.press_rocking()
    return {
        "curve": params.curve,
        "tilt_deg": tilt_deg,
        "flat": flat,
        "rock": rock,
        "flat_peak": flat.peak_force,
        "rocking_peak": rock.peak_force,
        "ratio": rock.peak_force / max(flat.peak_force, 1e-9),
        "flat_seated": flat_bench.seated(),
        "rocking_seated": rock_bench.seated(),
        "flat_depth_spread": float(np.max(flat.depths.max(axis=1) - flat.depths.min(axis=1))),
        "rocking_depth_spread": float(np.max(rock.depths.max(axis=1) - rock.depths.min(axis=1))),
        "peak_wrist_torque": float(np.max(np.abs(rock.wrist_torque))),
        "warnings": int(flat.warnings + rock.warnings),
    }


def rocking_config(tilt_deg: float = 4.0, spec: BrickSpec = DUPLO, ny: int = 4) -> BenchConfig:
    """The bench set-up for a rocking press: tipped brick, tool held at the trailing edge."""

    edge = (ny - 1) / 2.0 * spec.pitch
    return BenchConfig(
        ny=ny,
        spec=spec,
        tilt=True,
        tilt_deg=tilt_deg,
        grip=(0.0, -edge - spec.pitch / 2.0, 0.0),
        start_gap=0.0005,
    )


def build_stack_scene(
    snap: SnapModel,
    offsets: list[tuple[int, int]],
    spec: BrickSpec = DUPLO,
    nx: int = 2,
    ny: int = 4,
    plate_nx: int = 6,
    plate_ny: int = 8,
    timestep: float = 0.002,
    gravity: bool = True,
    inverted: bool = False,
    cfg: BenchConfig = BenchConfig(),
) -> tuple[mujoco.MjModel, mujoco.MjData, list]:
    """A baseplate with free bricks already sitting flush on it, one layer per entry.

    `offsets` gives each layer's shift in whole studs, so `[(0, 0), (0, 1), (0, 2)]` is a
    three-layer stack stepped one stud along y - a staircase with an overhang, which is the
    arrangement PLAN section 6.3 test 5 asks for. `inverted` builds the same scene upside
    down: the plate is rolled 180 degrees about x and the bricks hang underneath it, so the
    connection carries the bricks' weight instead of resting on it.

    Returns `(model, data, handles)` with the snap model already declared and bound and the
    scene settled at `mj_forward`, ready to step.
    """

    world = mujoco.MjSpec()
    opt = world.option
    opt.timestep = timestep
    opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    opt.impratio = 10.0
    opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
    opt.iterations = 100
    opt.noslip_iterations = 0
    if not gravity:
        opt.gravity = [0.0, 0.0, 0.0]
    world.default.geom.solref = list(cfg.solref)
    world.default.geom.solimp = list(cfg.solimp) + [0.5, 2.0]
    world.default.geom.friction = [cfg.friction, 0.005, 0.0001]

    flip = (0.0, 1.0, 0.0, 0.0)          # 180 degrees about x
    sign = -1.0 if inverted else 1.0
    quat = flip if inverted else (1.0, 0.0, 0.0, 0.0)
    plate_z = 0.1 if inverted else 0.0
    plate = add_baseplate(
        world, "plate", plate_nx, plate_ny, pos=(0.0, 0.0, plate_z), spec=spec, quat=quat
    )
    top = spec.roof * _PLATE_SLAB_FACTOR
    bricks = []
    for layer, (dx, dy) in enumerate(offsets):
        z = plate_z + sign * (top + layer * spec.height)
        bricks.append(
            add_brick(
                world,
                f"brick{layer}",
                nx,
                ny,
                ["red", "blue", "yellow", "green", "orange", "white"][layer % 6],
                pos=(dx * spec.pitch, sign * dy * spec.pitch, z),
                quat=quat,
                spec=spec,
            )
        )
    snap.declare(world, bricks, [plate])
    model = world.compile()
    data = mujoco.MjData(model)
    snap.bind(model, data)
    mujoco.mj_forward(model, data)
    snap.update(model, data)
    return model, data, bricks
