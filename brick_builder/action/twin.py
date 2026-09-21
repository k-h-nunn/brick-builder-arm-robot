"""The digital twin: what the wrist camera *should* see if the world were nominal.

Feed-forward, and entirely outside the control loop. For a step such as "the held brick over
cell (5.5, 4.5) at a hover of 20 mm" the twin builds a second, separate scene from the goal and
from nominal task knowledge only -- where the baseplate nominally is, which bricks the goal says
are already on it, which brick is in the hand -- poses its arm at that step's goal pose, renders
the wrist view and runs the same `IdealSceneModel` over it. The result is the **intended**
`SceneState`: jaws, held brick, plate and the already-placed bricks all in it, so parallax,
cropping and occlusion are already right rather than approximated by projecting a point.

The twin is also where the feature Jacobian comes from. Re-posing it by +/-2 mm and +/-2 deg and
re-rendering gives d(features)/d(tool dx, dy, spin) by central differences: a measurement of the
camera's own geometry at that pose, taken in a world the controller is allowed to know about.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import numpy as np

from ..perception.ideal_scene_model import IdealSceneModel
from ..perception.scene_model import SceneState
from ..perception.views import render_view
from ..render.renderer import SceneRenderer
from ..scene.compose import (
    Scene, SceneConfig, build_scene, held_brick_pose, reset_scene, tool_down_ik,
)
from .adapter import WRIST_FULL, WRIST_RENDER


def brick_features(state, k: int) -> np.ndarray:
    """The held brick's whole published row, as a feature vector for the in-hand fit.

    Centre alone cannot separate "lower in the jaws" from "further along them"; mass, the second
    moments and `orient4` can (see `DigitalTwin._brick_jacobian`).
    """
    k = int(k)
    cx, cy = state.centre(k)
    sxx, sxy, syy = state.moments(k)
    re, im = state.schema.get(state.slots[k], "orient4")
    return np.array([cx, cy, state.mass(k), sxx, sxy, syy, float(re), float(im)], dtype=float)


class BlindStep(RuntimeError):
    """Not enough of the scene stays in view at this pose to measure an inverse from."""


@dataclass(frozen=True)
class TwinStep:
    """One step's intended state and the Jacobian measured at its goal pose."""

    name: str
    goal_tcp: np.ndarray
    goal_spin: float
    intended: SceneState
    J: np.ndarray
    """`[2n, 3]`: d(reference blob centres) / d(tool dx [m], dy [m], spin [rad])."""
    J_pinv: np.ndarray
    feature_names: tuple[str, ...]
    s_star: np.ndarray
    tol: float
    """Done threshold on the brick-to-cell error [m] -- 0.3 mm, inside the studs' capture range."""
    J_brick: np.ndarray | None = None
    """`[8, 3]`: d(held brick's published row) / d(brick's own world dx, dy, dz). Different from `J`
    because the brick's top face is much nearer the lens than the plate is: the same millimetre
    of world motion moves the two blobs by different numbers of pixels."""
    J_brick_pinv: np.ndarray | None = None
    b_star: np.ndarray | None = None
    angle_mode: str = "pair"
    """How yaw is read: from the line joining two reference blobs (`"pair"`), or from one blob's
    own second-moment ellipse (`"blob"`, for lining up on a single brick)."""
    angle_slot: str | None = None
    """Which reference's ellipse carries the angle, in `"blob"` mode."""
    angle_period: int = 1
    """The symmetry the orientation belongs to: 4 for a square footprint, 2 for an elongated
    one, 1 for the line joining two reference blobs. A 2x2 brick looks identical every quarter
    turn, and any of the four is equally good to grip."""

    spin_star: complex | None = None
    """The target orientation, carried COMPLEX so the error can be taken from the ratio rather
    than from two separately wrapped angles (see `orientation_error`)."""
    spin_gain: float | None = None
    """How much that angle turns per radian of tool spin."""
    refs: tuple = ()
    """The references as the twin calibrated them -- see `DigitalTwin.calibrate`."""
    brick_ref: object | None = None


class DigitalTwin:
    """A nominal copy of the world the controller can pose and render at will."""

    def __init__(self, cfg: SceneConfig, *, K: int = 8, seed: int = 0, prepare=None) -> None:
        self.cfg = cfg
        self.scene: Scene = build_scene(cfg)
        if prepare is not None:
            # Whatever was done to the real world's model has to be done here too, *before* the
            # renderer builds its context: the twin describes its references by the colour they
            # show, and a scene lit differently shows different colours.
            prepare(self.scene.model)
        reset_scene(self.scene, np.random.default_rng(seed))
        self.renderer = SceneRenderer(self.scene.model, width=WRIST_RENDER[0],
                                      height=WRIST_RENDER[1])
        self.renderer.instance_lut(self.scene.instance_ids)
        self._vision = IdealSceneModel(K=K, build_decode=False)
        self._q = self.scene.home_q.copy()
        self._grip: float | None = None

    def close(self) -> None:
        """Give the renderer's GL context back. A study that builds a twin per seed and never
        closes it runs the graphics driver out of contexts and the process simply stops."""
        self.renderer.close()

    # -- posing ---------------------------------------------------------------------------------

    def pose(self, tcp, spin: float = 0.0, grip: float | None = None) -> None:
        """Put the twin's tool at `tcp` pointing down at yaw `spin`, and carry the held brick
        with it. Kinematics only -- no physics, so it is fast and exactly repeatable."""
        scene = self.scene
        q = tool_down_ik(scene.kin, np.asarray(tcp, dtype=float), float(spin), q0=self._q)
        self._q = q.copy()
        scene.data.qpos[scene.arm_qpos_adr[:5]] = q
        if grip is not None:
            # The jaws are in the picture too: an expectation rendered with them closed is not
            # what a camera sees while they are open around a brick.
            scene.data.qpos[scene.arm_qpos_adr[5]] = float(grip)
        mujoco.mj_forward(scene.model, scene.data)
        if scene.held_name is not None:
            pos, quat = held_brick_pose(scene, scene.in_hand_offset)
            adr = scene.free_joint_adr(scene.held_name)
            scene.data.qpos[adr:adr + 3] = pos
            scene.data.qpos[adr + 3:adr + 7] = quat
            mujoco.mj_forward(scene.model, scene.data)

    def observe(self, tcp, spin: float = 0.0, grip: float | None = None) -> SceneState:
        """Pose, render, and turn the render into slot states."""
        self.pose(tcp, spin, grip)
        frame, _ = render_view(self.scene, WRIST_FULL, self.renderer)
        self._vision.reset()
        return self._vision.observe(
            frame.rgb, window=WRIST_FULL, instance=frame.instance, depth=frame.depth, t=0.0,
        )

    def observe_rgb(self, tcp, spin: float = 0.0) -> tuple[SceneState, np.ndarray]:
        self.pose(tcp, spin)
        frame, _ = render_view(self.scene, WRIST_FULL, self.renderer)
        self._vision.reset()
        state = self._vision.observe(
            frame.rgb, window=WRIST_FULL, instance=frame.instance, depth=frame.depth, t=0.0,
        )
        return state, np.asarray(frame.rgb)

    # -- the step ---------------------------------------------------------------------------------

    def calibrate(self, state: SceneState, bodies: dict, *, colour_tol: float = 0.30,
                  size_tol: float = 0.35, max_border: float = 0.25):
        """Turn "the yellow 2x4" into something `select_slot` can find in the real camera.

        A palette colour is not what a camera sees -- the renderer's lighting darkens a yellow
        brick until it is nearer the palette's red than its own yellow, and matching on the
        palette picks the wrong brick. The twin, though, knows which of *its* slots is which
        object (the scene model's slot id is the renderer's own instance id), so it can read off
        the colour and ellipse size that object actually shows at this pose and hand those over
        as the description. Nothing about the real world is used: this is the twin describing
        its own expectation.
        """
        from .servo import Reference

        out = []
        for label, body in bodies.items():
            iid = int(self.scene.instance_ids[body])
            match = [k for k in range(state.K)
                     if not state.is_empty(k) and int(state.slot_ids[k]) == iid]
            if not match:
                # Not fatal. The reference frame was recorded from a slightly different pose and
                # a thing that was just inside its frame can be just outside this one; the step
                # is tracked on whatever else is visible, and `build_step` refuses only when too
                # little is left.
                continue
            k = match[0]
            _, axes = state.orientation_and_axes(k)
            colour = tuple(0.5 * (c + 1.0) for c in state.colour(k))
            out.append(Reference(
                name=label, colour=colour, size=axes, colour_tol=colour_tol,
                size_tol=size_tol, min_mass=0.35 * state.mass(k), max_border=max_border,
            ))
        return out

    def build_step(self, name: str, goal_tcp, goal_spin: float, bodies: dict,
                   *, d_xy: float = 0.002, d_spin: float = math.radians(2.0),
                   tol_mm: float = 0.3, brick_body: str | None = None,
                   min_refs: int = 2, angle_from_blob: bool = False,
                   square: bool = False, grip: float | None = None,
                   brick_dof: int = 2) -> TwinStep:
        """Everything one servoing step needs, measured once in the twin before the loop runs.

        `bodies` maps a plain-words label ("the yellow 2x4") to the body the twin knows it by.
        """
        from .servo import reference_features

        goal_tcp = np.asarray(goal_tcp, dtype=float)
        self._grip = grip
        intended = self.observe(goal_tcp, goal_spin, grip)
        references = self.calibrate(intended, bodies)
        brick_ref = None
        if brick_body is not None:
            brick_ref = self.calibrate(intended, {"the brick in the hand": brick_body},
                                       max_border=0.45)[0]
        base = reference_features(intended, references)
        names = tuple(n for n in base.order if base.centres[n] is not None)
        if len(names) < min_refs:
            raise RuntimeError(
                f"step {name!r}: the twin only shows {len(names)} of the "
                f"{len(references)} reference blobs, which is not enough to line up on"
            )
        s_star = _stack(base.centres, names)

        probes = []
        for axis, delta in ((0, d_xy), (1, d_xy)):
            plus = np.array([0.0, 0.0, 0.0])
            plus[axis] = delta
            probes.append((delta,
                           self._features_at(goal_tcp, goal_spin, plus, references, names),
                           self._features_at(goal_tcp, goal_spin, -plus, references, names)))
        names = tuple(n for n in names
                      if all(p[n] is not None and m[n] is not None for _, p, m in probes))
        if len(names) < min_refs:
            raise BlindStep(
                f"step {name!r}: only {len(names)} reference blob(s) stay in view while the "
                f"Jacobian is measured"
            )
        s_star = _stack(base.centres, names)
        cols = [(_stack(p, names) - _stack(m, names)) / (2.0 * d) for d, p, m in probes]
        J = np.stack(cols, axis=1)
        J_pinv = np.linalg.pinv(J, rcond=1e-6)

        # Yaw gets its own estimator rather than a third column. Spinning the tool by a degree
        # barely moves a blob's centre compared with a millimetre of sideways travel, so a
        # least-squares solve over all three at once is badly conditioned and walks away. The
        # *angle of the line joining two far-apart reference blobs*, though, is almost pure
        # rotation: it hardly changes when the tool slides, and changes a lot when it turns.
        spin_star = spin_gain = None
        angle_mode, angle_slot, period = "pair", None, 2.0 * math.pi
        if len(names) >= 2:
            c_p = self._features_at(goal_tcp, goal_spin, (0.0, 0.0, d_spin), references, names)
            c_m = self._features_at(goal_tcp, goal_spin, (0.0, 0.0, -d_spin), references, names)
            if all(c_p[n] is not None and c_m[n] is not None for n in names):
                # Pair mode keeps plain wrapped angles. The branch cut that bites a square
                # brick sits at 45 degrees, right where a gripped brick lives; the line joining
                # two reference blobs would have to reverse end for end to reach ITS cut at 180,
                # which never happens. Switching this path to the ratio form changed the
                # placement task from 10/10 to 0/3, so it stays as it was and as it was measured.
                spin_star = _pair_angle(s_star)
                spin_gain = _wrap(_pair_angle(_stack(c_p, names))
                                  - _pair_angle(_stack(c_m, names))) / (2.0 * d_spin)
                period = 1
        elif len(names) == 1 and angle_from_blob:
            # One brick to line up on: its own ellipse is the only thing that says which way it
            # is turned. A square brick's ellipse repeats every quarter turn, so the error is
            # taken modulo that -- and it is not a fudge, because all four grips are the same
            # grip.
            angle_mode, angle_slot = "blob", names[0]
            z0, fold = blob_orientation(intended, base.slots[names[0]])
            st_p = self.observe(goal_tcp, goal_spin + d_spin, grip)
            f_p = reference_features(st_p, references)
            k_p = f_p.slots[names[0]]
            if k_p is None:
                raise BlindStep("the twin lost the brick's blob measuring its turn")
            z_p, _ = blob_orientation(st_p, k_p)
            err, _ = orientation_error(z_p, z0, fold)
            spin_star = z0
            spin_gain = err / d_spin
            period = fold

        J_brick = J_brick_pinv = b_star = None
        if brick_ref is not None:
            b_star, J_brick = self._brick_jacobian(goal_tcp, goal_spin, brick_ref, d_xy,
                                                   dof=brick_dof)
            J_brick_pinv = np.linalg.pinv(J_brick, rcond=1e-6)

        # leave the twin sitting at the goal pose, which is what a caller rendering the
        # "intended" inset wants to see
        self.pose(goal_tcp, goal_spin, grip)
        return TwinStep(
            name=name, goal_tcp=goal_tcp, goal_spin=float(goal_spin), intended=intended,
            J=J, J_pinv=J_pinv, feature_names=names, s_star=s_star, tol=tol_mm * 1e-3,
            J_brick=J_brick, J_brick_pinv=J_brick_pinv, b_star=b_star,
            refs=tuple(references), brick_ref=brick_ref,
            spin_star=spin_star, spin_gain=spin_gain,
            angle_mode=angle_mode, angle_slot=angle_slot, angle_period=period,
        )

    # -- the held brick's own Jacobian ------------------------------------------------------------

    def _brick_jacobian(self, goal_tcp, goal_spin: float, brick_ref, d_xy: float,
                        *, dof: int = 2):
        """How the held brick's blob changes when the brick sits somewhere else in the hand.

        Three columns, not two, and eight features, not two. A brick sitting lower in the jaws
        moves its blob's centre almost the same way as one sitting further along them --
        measured, `d(centre y)` is -3.99 per metre of vertical against +5.47 per metre along the
        jaws -- so a centre-only fit reads a real 12 mm of vertical as a phantom 9 mm sideways,
        and the loop then corrects for it. Mass tells them apart: -0.262 per metre of vertical
        against -0.038 along the jaws, because a brick that has slid towards the gripper body
        has part of its top face hidden. The second moments and `orient4` help as well.

        The perturbation is written in the *tool* frame (that is what `held_brick_pose` takes)
        but each column is one metre of **world** brick motion, which is the frame the correction
        is commanded in.
        """
        from .servo import reference_features
        from ..control.pick_place import _tool_down_rot

        scene = self.scene
        if scene.held_name is None:
            raise RuntimeError("this twin has no brick in the hand to measure")
        rot = _tool_down_rot(float(goal_spin))
        saved = np.asarray(scene.in_hand_offset, dtype=float).copy()

        iid = int(scene.instance_ids[scene.held_name])

        def blob(world_delta) -> np.ndarray:
            off = saved.copy()
            off[:3] = saved[:3] + rot.T @ np.asarray(world_delta, dtype=float)
            scene.in_hand_offset = off
            state = self.observe(goal_tcp, goal_spin, self._grip)
            match = [k for k in range(state.K)
                     if not state.is_empty(k) and int(state.slot_ids[k]) == iid]
            if not match:
                raise BlindStep("the twin lost the held brick's blob measuring its Jacobian")
            row = brick_features(state, match[0])
            return row if dof >= 3 else row[:2]

        try:
            b_star = blob((0.0, 0.0, 0.0))
            cols = []
            for axis in range(dof):
                plus = np.zeros(3)
                plus[axis] = d_xy
                cols.append((blob(plus) - blob(-plus)) / (2.0 * d_xy))
        finally:
            scene.in_hand_offset = saved
        return b_star, np.stack(cols, axis=1)

    def _features_at(self, goal_tcp, goal_spin, delta, references, names) -> dict:
        """The reference centres a small distance from the goal pose, for the differences.

        Returns whatever it can see rather than insisting on everything: a couple of millimetres
        of camera motion is enough to push an object that was already half out of frame all the
        way out, and a step that loses one of three references is still a step worth tracking on
        the other two.
        """
        from .servo import reference_features

        tcp = np.asarray(goal_tcp, dtype=float) + np.array([delta[0], delta[1], 0.0])
        state = self.observe(tcp, float(goal_spin) + float(delta[2]), self._grip)
        return reference_features(state, references).centres


ROUND_ENOUGH = 1.3
"""Second-moment axis ratio below which a blob is treated as square rather than elongated."""


def blob_orientation(state, k: int) -> tuple[complex, int]:
    """One blob's orientation as a COMPLEX number, with the symmetry it belongs to.

    Never as an angle. A square brick gripped square to the jaws has `c4` on the negative real
    axis, which puts `arg(c4)/4` at exactly 45 degrees -- the wrap boundary of the quarter turn
    it is defined modulo. Pixel noise then flips it between +45 and -45 from frame to frame, and
    anything that wraps two angles separately and subtracts them sees tens of degrees of error
    with random sign. Filtered into a correction, that is a slow drift: it was worth -5.6 degrees
    of tool yaw in a world where nothing was rotated.

    So the number is carried complex and the error is taken from the ratio (`orientation_error`),
    which is continuous through the boundary by construction. `fold` is 4 for a square footprint
    (the fourth moment) and 2 for an elongated one (the complex second moment).
    """
    k = int(k)
    _, axes = state.orientation_and_axes(k)
    ratio = axes[0] / max(axes[1], 1e-9)
    if ratio < ROUND_ENOUGH and state.schema.has("orient4"):
        re, im = state.schema.get(state.slots[k], "orient4")
        z = complex(float(re), float(im))
        if abs(z) > 0.05:
            return z, 4
    sxx, sxy, syy = state.moments(k)
    return complex(sxx - syy, 2.0 * sxy), 2


def orientation_error(z_obs: complex, z_target: complex, fold: int) -> tuple[float, float]:
    """How far `z_obs` is turned from `z_target` [rad], and how much to believe it.

    `arg(z_obs * conj(z_target)) / fold` lands in (-pi/fold, pi/fold] automatically and is
    continuous everywhere, including through the boundary where two wrapped angles would
    disagree by a whole period. The weight is the weaker of the two magnitudes, so a blob whose
    shape barely carries an orientation contributes barely anything.
    """
    if z_obs == 0 or z_target == 0:
        return 0.0, 0.0
    return float(np.angle(z_obs * np.conj(z_target)) / fold), float(min(abs(z_obs), abs(z_target)))


def _pair_angle(s: np.ndarray) -> float:
    """Angle of the line from the first reference blob to the second, in the image."""
    return float(math.atan2(s[3] - s[1], s[2] - s[0]))


def _wrap_to(a: float, period: float) -> float:
    """`a` folded into (-period/2, period/2]."""
    return float(a - period * round(a / period))


def _wrap(a: float) -> float:
    return float(math.remainder(a, 2.0 * math.pi))


def _stack(centres: dict, names) -> np.ndarray:
    return np.concatenate([np.asarray(centres[n], dtype=float) for n in names])
