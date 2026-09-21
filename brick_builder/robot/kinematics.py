"""Forward and inverse kinematics for the SO-101's 5-DoF arm.

The gripper joint is not part of this chain -- it does not move the TCP site along the arm's task
space in any way this module cares about, and the spec's IK task (position + tool pitch + tool
roll = 5 numbers) exactly matches the five non-gripper joints (``shoulder_pan`` .. ``wrist_roll``).
Everything here is built on MuJoCo's own kinematics and ``mujoco.mj_jacSite``: no separate
DH-parameter or screw-axis model is maintained, so a C port only ever has to call the same MuJoCo
functions.

Where the poses are evaluated
-----------------------------
A pose query is a pure function of the five joint angles, so it does not need the scene it was
asked about: bricks, the build plate and the table can only slow it down. Each
:class:`ArmKinematics` therefore keeps its own little arm-only model (``build_arm_spec``, no
scene, and with every geom stripped out of it -- see :func:`_arm_template`) and evaluates on that
with ``mj_kinematics`` (+ ``mj_comPos``, which is what ``mj_jacSite`` and the gravity term need)
instead of running ``mj_forward`` on the whole world. Measured on the press scene: 1.6 us against
108 us for ``mj_resetData`` + ``mj_forward`` with the arm at its home pose, and 397 us with the
arm driven into the plate, where the scene has contacts to solve and the arm-only model still has
nothing to do. The site pose, the site Jacobian and the gravity term all come out bit-for-bit
identical.

Two things keep that shortcut honest:

- the arm-only model's body masses, inertias and inertial frames are copied from the host model
  whenever they have changed, so the ``dynamics.link_mass_scale`` randomization -- which writes
  the scene model's ``body_mass`` between episodes -- still reaches
  :meth:`ArmKinematics.gravity_torque`. (It is the only thing that does: the SO-101 MJCF gives
  every link an explicit ``<inertial>``, so any geom a scene adds to the arm carries no mass
  of its own.) After a randomization the two models agree to about
  1e-14 relative rather than to the bit, which is ``mj_setConst`` recomputing the host's derived
  constants; before one they are identical;
- the constructor checks the two models against each other at a handful of joint configurations
  and, if the site pose does not match exactly (an arm attached somewhere other than the world
  origin, a differently built arm), quietly falls back to evaluating on the host model -- still
  with ``mj_kinematics``/``mj_comPos``, never a full ``mj_forward``.

Orientation convention
-----------------------
The tool frame's rotation is decomposed as ``R = Rz(yaw) @ Ry(pitch) @ Rx(roll)`` (world axes,
intrinsic order z-y-x).  Yaw is not an IK target -- with no independent yaw joint at the wrist, the
tool's heading is whatever the base joint (``shoulder_pan``) needs to be to reach the target x/y,
so IK only drives position (3) plus pitch and roll (2), five numbers for five joints.

Rather than solve for a full target rotation matrix (which would need a yaw value nobody supplies)
the residual pins the target frame's yaw to the arm's own *current* yaw at every evaluation
(``Rz(yaw_now) @ Ry(pitch_target) @ Rx(roll_target)``, always a well-defined rotation matrix to
build) and reads off the angular error to that frame with the usual cross-product formula.

:meth:`ArmKinematics.ik` reads "pitch error" and "roll error" off as two of that error's three
*local* components (the third, ~0 by construction, is the free yaw). That makes its residual a
function of the joint angles in a more roundabout way than position is -- differentiating it
correctly means differentiating through the yaw extraction too -- so its Newton step's Jacobian is
built by finite differences of the exact residual function instead of composed analytically, and
every step is accepted only if it actually lowers the re-evaluated residual (Levenberg-Marquardt),
which also tames the three joints (``shoulder_lift``, ``elbow_flex``, ``wrist_flex``) that share
one rotation axis and so cannot be told apart by the orientation rows alone. That is the solver
for resets and waypoints, where the seed can be far from the answer.

:meth:`ArmKinematics.ik_diff` is the per-tick solver, and it avoids the finite differences by
choosing a different pair of rows out of the same angular error. Pinning the target's yaw to the
arm's own heading means the target frame's only dependence on ``q`` is a rotation about the
*world* z axis (``d Rd/dq = [z]x Rd * d(yaw)/dq``), so the world x and y components of the angular
error have no such term at all: their exact derivative is minus the corresponding two rows of
``mj_jacSite``'s analytic angular Jacobian. Those two rows pin the same two degrees of freedom the
local pitch/roll pair does -- the discarded world-z row is the free yaw when the tool points down,
and stays the (first-order) dependent combination ``-sin(pitch) * roll_error`` when it is tilted --
so nothing is given up by the change of basis, and the chain-rule term an earlier revision
found hard to get right simply does not arise.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import mujoco
import numpy as np

from brick_builder.robot.patch_so101 import ARM_JOINTS, TCP_SITE, build_arm_spec

KIN_JOINTS: tuple[str, ...] = ARM_JOINTS[:-1]
"""The five joints this module solves for: everything but ``gripper``."""

N_DOF = len(KIN_JOINTS)


def euler_zyx(rot: np.ndarray) -> tuple[float, float, float]:
    """Decompose ``rot`` as ``Rz(yaw) @ Ry(pitch) @ Rx(roll)``; returns ``(yaw, pitch, roll)``.

    This is the convention IK targets are given in.  Like every three-angle Euler decomposition it
    is only informative away from its own gimbal lock (here, ``pitch`` near +-90 degrees); IK itself
    never inverts this decomposition, it is provided for callers that want to read off a pose's
    pitch/roll (tests, the force map's "tool-down" targets, logging).
    """
    pitch = math.atan2(-rot[2, 0], math.hypot(rot[2, 1], rot[2, 2]))
    yaw = math.atan2(rot[1, 0], rot[0, 0])
    roll = math.atan2(rot[2, 1], rot[2, 2])
    return yaw, pitch, roll


def _cross_columns(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """``sum_i a[:, i] x b[:, i]`` for two 3x3 matrices.

    Written out rather than three ``np.cross`` calls plus two array additions: the operations and
    their order are the same, so the result is bit-for-bit identical, but none of ``np.cross``'s
    axis normalization and temporary arrays are paid for.
    """

    return np.array([
        (a[1, 0] * b[2, 0] - a[2, 0] * b[1, 0])
        + (a[1, 1] * b[2, 1] - a[2, 1] * b[1, 1])
        + (a[1, 2] * b[2, 2] - a[2, 2] * b[1, 2]),
        (a[2, 0] * b[0, 0] - a[0, 0] * b[2, 0])
        + (a[2, 1] * b[0, 1] - a[0, 1] * b[2, 1])
        + (a[2, 2] * b[0, 2] - a[0, 2] * b[2, 2]),
        (a[0, 0] * b[1, 0] - a[1, 0] * b[0, 0])
        + (a[0, 1] * b[1, 1] - a[1, 1] * b[0, 1])
        + (a[0, 2] * b[1, 2] - a[1, 2] * b[0, 2]),
    ])


def _rot_zyx(yaw: float, pitch: float, roll: float) -> np.ndarray:
    cz, sz = math.cos(yaw), math.sin(yaw)
    cy, sy = math.cos(pitch), math.sin(pitch)
    cx, sx = math.cos(roll), math.sin(roll)
    rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
    return rz @ ry @ rx


@dataclass(frozen=True)
class IKResult:
    """Outcome of a damped-least-squares solve.

    ``converged`` is the honest answer to "is this usable" -- an unreachable target is reported as
    ``converged=False`` with whatever residual DLS bottomed out at, not silently clipped into a
    plausible-looking but wrong joint configuration.
    """

    q: np.ndarray
    converged: bool
    pos_error: float
    pitch_error: float
    roll_error: float
    iterations: int


@dataclass(frozen=True)
class DiffIKResult:
    """Outcome of one call to :meth:`ArmKinematics.ik_diff`.

    ``ang_error`` is the larger of the two angular residual rows (world x and y components of the
    tool's orientation error, radians); ``yaw_error`` is how far the achieved tool spin is from
    the requested one, or 0.0 when no spin was requested.
    """

    q: np.ndarray
    converged: bool
    pos_error: float
    ang_error: float
    yaw_error: float
    iterations: int


_ARM_TEMPLATE: mujoco.MjModel | None = None


def _arm_template() -> mujoco.MjModel:
    """The compiled arm-only model every :class:`ArmKinematics` copies its fast model from.

    Every geom goes, and with them the meshes, textures and materials: a pose query needs the
    body tree, the joints, the sites and the inertials, and nothing here ever collides with
    anything or is rendered. The vendored SO-101 MJCF gives every link an explicit
    ``<inertial>``, so the masses survive the geoms being removed -- and they are overwritten
    from the host model anyway. What it saves is the 160 000 mesh vertices: 0.11 MB per copy
    instead of 54, and 0.3 ms to copy instead of 30.

    Compiling it costs about 150 ms, so it is built once per process and copied per instance
    (each instance writes its own host's link masses into its own copy).
    """

    global _ARM_TEMPLATE
    if _ARM_TEMPLATE is None:
        spec = build_arm_spec()
        for group in (spec.geoms, spec.meshes, spec.textures, spec.materials):
            for element in list(group):
                spec.delete(element)
        _ARM_TEMPLATE = spec.compile()
    return _ARM_TEMPLATE


YAW_WEIGHT = 0.15
"""How much a radian of spin error weighs against a radian of tool pitch/roll error in
:meth:`ArmKinematics.ik_diff`'s least-squares step.

Five joints cannot reach every (position, tool axis, spin) triple, so something has to give when
they conflict, and the spin is what the task can most afford to lose: the press cares about where
the brick is and that it is level, and a tool anywhere near vertical can spin freely anyway, so
the conflict only shows up once the tool is tilted.

The number is measured, not taste.  Over 400 random-policy Reach ticks at the two-step budget the
environment uses (tilt and spin residuals driven to their +-20 degree clip box, which is far
harder than any real policy asks for), sweeping the weight gives median errors of: 0 -> tool
pitch/roll 0.004 deg, spin 3.5 deg (the spin command does nothing at all); 0.15 -> 0.18 deg and
0.64 deg; 0.5 -> 0.25 deg and 0.09 deg.  0.15 is the setting that
reproduces the compromise the previous solver reached by solving for pitch/roll and then nudging
``wrist_roll`` -- about a fifth of a degree of tool tilt bought for about half a degree of spin --
in one linear solve instead of several passes.  For the smooth commands a scripted primitive or a
trained policy actually produces there is no conflict to resolve and every residual comes out at
1e-5 or below whatever this is set to.
"""


_VERIFY_Q = np.array([
    [0.0, 0.0, 0.0, 0.0, 0.0],
    [0.3, -0.6, 0.9, 1.1, 0.4],
    [-0.8, 0.5, -0.7, -0.3, -1.2],
    [1.1, 1.0, -1.3, 0.6, 0.9],
])
"""Joint configurations the constructor checks the arm-only model against the host model at."""

#: Body properties the fast model mirrors from its host. Everything else about an arm link
#: (where it hangs, which axis its joint turns about) is fixed geometry both models share.
_MIRRORED = ("body_mass", "body_inertia", "body_ipos", "body_iquat")


class ArmKinematics:
    """Forward/inverse kinematics bound to one compiled :class:`mujoco.MjModel`.

    ``model`` is the host: the scene the arm lives in. Poses are evaluated on a private arm-only
    copy of it when the two agree (see the module docstring), otherwise on the host itself.
    """

    def __init__(self, model: mujoco.MjModel, tcp_site: str = TCP_SITE, *, fast: bool = True) -> None:
        self.model = model
        self.tcp_site = tcp_site
        self.site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, tcp_site)
        if self.site_id < 0:
            raise ValueError(f"model has no site named {tcp_site!r}")

        joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in KIN_JOINTS]
        missing = [name for name, jid in zip(KIN_JOINTS, joint_ids) if jid < 0]
        if missing:
            raise ValueError(f"model is missing joint(s) {missing}")
        self.joint_ids = joint_ids
        self.qpos_adr = [int(model.jnt_qposadr[j]) for j in joint_ids]
        self.dof_adr = [int(model.jnt_dofadr[j]) for j in joint_ids]
        self.jnt_range = np.array([model.jnt_range[j] for j in joint_ids])

        self._data = mujoco.MjData(model)
        mujoco.mj_resetData(model, self._data)

        # The evaluation model: the arm-only copy if it reproduces the host exactly, else the host.
        self._bind_eval_model(host_only=not fast)

    # ------------------------------------------------------------------------------- internals ---

    def _bind_eval_model(self, *, host_only: bool) -> None:
        """Pick the model poses are evaluated on and resolve its addresses."""

        self.fast_model = False
        self._e_model, self._e_data = self.model, self._data
        self._e_qposadr = np.asarray(self.qpos_adr, dtype=np.intp)
        self._e_dofadr = np.asarray(self.dof_adr, dtype=np.intp)
        self._e_site = self.site_id
        self._src_bodies = self._dst_bodies = None
        if host_only:
            self._jacp = np.zeros((3, self._e_model.nv))
            self._jacr = np.zeros((3, self._e_model.nv))
            return

        try:
            mini = copy.deepcopy(_arm_template())
        except Exception:                                   # pragma: no cover - build/copy failure
            self._jacp = np.zeros((3, self._e_model.nv))
            self._jacr = np.zeros((3, self._e_model.nv))
            return
        mini_data = mujoco.MjData(mini)
        mujoco.mj_resetData(mini, mini_data)
        mini_data.qvel[:] = 0.0      # kept at zero forever: `cvel`/`cdof_dot` then stay zero too,
                                     # which is what makes `mj_rne` the pure gravity term.

        site = mujoco.mj_name2id(mini, mujoco.mjtObj.mjOBJ_SITE, self.tcp_site)
        jids = [mujoco.mj_name2id(mini, mujoco.mjtObj.mjOBJ_JOINT, n) for n in KIN_JOINTS]
        if site < 0 or any(j < 0 for j in jids):            # pragma: no cover - patched-out site
            self._jacp = np.zeros((3, self._e_model.nv))
            self._jacr = np.zeros((3, self._e_model.nv))
            return

        src, dst = [], []
        for bid in range(mini.nbody):
            name = mujoco.mj_id2name(mini, mujoco.mjtObj.mjOBJ_BODY, bid)
            host_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name) if name else -1
            if host_bid >= 0:
                src.append(host_bid)
                dst.append(bid)
        src_bodies = np.asarray(src, dtype=np.intp)
        dst_bodies = np.asarray(dst, dtype=np.intp)

        keep = (self._e_model, self._e_data, self._e_qposadr, self._e_dofadr, self._e_site)
        self._e_model, self._e_data = mini, mini_data
        self._e_qposadr = np.asarray([int(mini.jnt_qposadr[j]) for j in jids], dtype=np.intp)
        self._e_dofadr = np.asarray([int(mini.jnt_dofadr[j]) for j in jids], dtype=np.intp)
        self._e_site = site
        self._src_bodies, self._dst_bodies = src_bodies, dst_bodies
        self._mirrored_cache: np.ndarray | None = None
        self._jacp = np.zeros((3, mini.nv))
        self._jacr = np.zeros((3, mini.nv))
        self._mirror_inertials()

        if self._agrees_with_host():
            self.fast_model = True
            return
        # Not the same arm: fall back to the host model, which is still evaluated without a
        # full `mj_forward`.
        (self._e_model, self._e_data, self._e_qposadr, self._e_dofadr, self._e_site) = keep
        self._src_bodies = self._dst_bodies = None
        self._jacp = np.zeros((3, self.model.nv))
        self._jacr = np.zeros((3, self.model.nv))

    def _agrees_with_host(self) -> bool:
        """Does the arm-only model put the TCP site exactly where the host model does?"""

        host_qpos = np.asarray(self.qpos_adr, dtype=np.intp)
        for q in _VERIFY_Q:
            q = np.clip(q, self.jnt_range[:, 0], self.jnt_range[:, 1])
            self._data.qpos[host_qpos] = q
            mujoco.mj_kinematics(self.model, self._data)
            want_pos = self._data.site_xpos[self.site_id].copy()
            want_mat = self._data.site_xmat[self.site_id].copy()
            self._e_data.qpos[self._e_qposadr] = q
            mujoco.mj_kinematics(self._e_model, self._e_data)
            if not (
                np.array_equal(want_pos, self._e_data.site_xpos[self._e_site])
                and np.array_equal(want_mat, self._e_data.site_xmat[self._e_site])
            ):
                return False
        return True

    def _mirror_inertials(self) -> None:
        """Copy the host's link masses, inertias and inertial frames into the fast model."""

        if self._src_bodies is None:
            return
        for name in _MIRRORED:
            getattr(self._e_model, name)[self._dst_bodies] = (
                getattr(self.model, name)[self._src_bodies]
            )
        self._mirrored_cache = self.model.body_mass[self._src_bodies].copy()

    def _sync_inertials(self) -> None:
        """Re-mirror the host's inertial properties if they have changed since the last check.

        The gravity term is the one pose query that depends on more than geometry, and
        ``randomize``'s ``dynamics.link_mass_scale`` changes exactly these arrays on the host.
        Only the mass vector is compared, because that is the one thing that changes at run time
        (``randomize.apply`` scales mass and inertia together; the inertial frames are fixed at
        compile time and are mirrored once at construction), and it makes the common case --
        nothing has moved since the last episode -- one eight-element array compare rather than
        rebuilding all four tables.
        """

        if self._src_bodies is None:
            return
        host = self.model
        mass = host.body_mass[self._src_bodies]
        if self._mirrored_cache is not None and np.array_equal(mass, self._mirrored_cache):
            return
        self._mirror_inertials()

    def _set_q(self, q: np.ndarray) -> None:
        """Put ``q`` on the evaluation model and run forward kinematics only.

        Everything else on the evaluation model is left where ``mj_resetData`` put it at
        construction and is never written again, so ``fk``/``jacobian``/``gravity_torque`` stay
        pure functions of ``q``: there is no leftover velocity or previous solve to bias the next
        one, and no per-call reset to pay for.
        """
        self._e_data.qpos[self._e_qposadr] = q
        mujoco.mj_kinematics(self._e_model, self._e_data)

    # ---------------------------------------------------------------------------- public API ---

    def fk(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Forward kinematics: TCP site position (3,) and rotation matrix (3, 3) at joint angles ``q``."""
        self._set_q(np.asarray(q, dtype=float))
        pos = self._e_data.site_xpos[self._e_site].copy()
        rot = self._e_data.site_xmat[self._e_site].reshape(3, 3).copy()
        return pos, rot

    def jacobian(self, q: np.ndarray) -> np.ndarray:
        """Site Jacobian (6, 5): rows 0-2 linear velocity, rows 3-5 angular velocity, world frame."""
        self._set_q(np.asarray(q, dtype=float))
        mujoco.mj_comPos(self._e_model, self._e_data)
        mujoco.mj_jacSite(self._e_model, self._e_data, self._jacp, self._jacr, self._e_site)
        cols = self._e_dofadr
        return np.vstack([self._jacp[:, cols], self._jacr[:, cols]])

    def gravity_torque(self, q: np.ndarray) -> np.ndarray:
        """Generalized torque needed to hold the arm still at ``q`` against gravity alone.

        ``mj_rne(..., flg_acc=0, ...)`` is exactly what fills ``data.qfrc_bias``: the
        Coriolis/centrifugal/gravity term.  The evaluation model's velocity is zero and stays
        zero, so the Coriolis/centrifugal part is exactly zero, leaving only gravity compensation.
        """
        self._sync_inertials()
        self._set_q(np.asarray(q, dtype=float))
        mujoco.mj_comPos(self._e_model, self._e_data)
        out = np.zeros(self._e_model.nv)
        mujoco.mj_rne(self._e_model, self._e_data, 0, out)
        return out[self._e_dofadr]

    # ------------------------------------------------------------------- per-tick differential ---

    def ik_diff(
        self,
        target_pos: np.ndarray,
        target_pitch: float,
        target_roll: float,
        q0: np.ndarray,
        *,
        target_yaw: float | None = None,
        iters: int = 3,
        damping: float = 1e-2,
        pos_tol: float = 2e-4,
        ang_tol: float = math.radians(0.2),
        yaw_tol: float = math.radians(0.1),
        yaw_weight: float = YAW_WEIGHT,
    ) -> DiffIKResult:
        """A few damped-least-squares steps with the analytic Jacobian, warm-started from ``q0``.

        This is the per-tick solver.  Between two control ticks the commanded pose moves by a
        couple of millimetres and a couple of degrees, so one or two Gauss-Newton steps from the
        previous joint goal land inside tolerance -- two is what
        :func:`~brick_builder.envs.base_env._pose_ik` allows -- and the global solver (:meth:`ik`) is
        only needed when they do not.

        The task error's first five rows are three of position and the *world* x and y components
        of the tool's orientation error, for the reason the module docstring gives: in that basis
        the derivative of the yaw-pinned target frame drops out exactly, so those rows of the
        Jacobian are ``-[jacp; jacr[0:2]]`` straight out of ``mj_jacSite``, with no finite
        differences and no chain rule through ``atan2``.

        ``target_yaw`` adds a sixth row -- the world-z angular row the first five leave out, whose
        derivative is ``-jacr[2]``.  Six goals on five joints is over-determined in general (this
        arm reaches a 5-dimensional set of poses), and the sixth is the one to give up: with the
        tool pointing straight down the spin is free (``wrist_roll`` turns about the tool axis and
        moves nothing else), so the sixth row is solved exactly, while the further the tool is
        tilted the less of the requested spin is actually available.  Rows are therefore scaled by
        their own tolerances before the solve -- position by ``pos_tol``, tool pitch/roll by
        ``ang_tol``, spin by ``yaw_weight`` times those -- so the least-squares compromise puts
        the unavoidable error in the spin rather than spreading it over the position, and the one
        linear solve does the job the fixed-point chase on ``wrist_roll`` used to do in several
        passes.

        Returns the best ``q`` found (always inside the joint ranges) and whether it met every
        tolerance.  A caller that cannot use a non-converged answer should keep its previous one.
        """

        lo, hi = self.jnt_range[:, 0], self.jnt_range[:, 1]
        q = np.clip(np.asarray(q0, dtype=float), lo, hi)
        target_pos = np.asarray(target_pos, dtype=float)
        rows = 5 if target_yaw is None else 6
        eye = np.eye(rows)
        scale = np.empty(rows)
        scale[0:3] = 1.0 / pos_tol
        scale[3:5] = 1.0 / ang_tol
        if rows == 6:
            scale[5] = yaw_weight / ang_tol

        residual = np.empty(6)
        jac = np.empty((6, N_DOF))
        it = 0
        pos_err = ang_err = yaw_err = 0.0
        for it in range(iters + 1):
            rot = self._diff_residual(q, target_pos, target_pitch, target_roll, residual)
            pos_err = float(math.sqrt(
                residual[0] * residual[0] + residual[1] * residual[1] + residual[2] * residual[2]
            ))
            ang_err = float(max(abs(residual[3]), abs(residual[4])))
            if target_yaw is None:
                yaw_err = 0.0
            else:
                yaw_err = float(math.remainder(
                    target_yaw - math.atan2(rot[1, 0], rot[0, 0]), 2.0 * math.pi
                ))
                residual[5] = yaw_err
            if pos_err < pos_tol and ang_err < ang_tol and abs(yaw_err) < yaw_tol:
                return DiffIKResult(
                    q=q, converged=True, pos_error=pos_err, ang_error=ang_err,
                    yaw_error=yaw_err, iterations=it,
                )
            if it == iters:
                break

            mujoco.mj_comPos(self._e_model, self._e_data)
            mujoco.mj_jacSite(self._e_model, self._e_data, self._jacp, self._jacr, self._e_site)
            jac[0:3] = -self._jacp[:, self._e_dofadr]
            jac[3:5] = -self._jacr[0:2, self._e_dofadr]
            if rows == 6:
                jac[5] = -self._jacr[2, self._e_dofadr]
            jw = jac[:rows] * scale[:, None]
            rw = residual[:rows] * scale
            # Damping relative to the scaled system, so the tolerance weights above do not
            # silently turn the step into an undamped (or a frozen) one.
            jjt = jw @ jw.T
            lam = damping * damping * float(np.trace(jjt)) / rows
            dq = -jw.T @ np.linalg.solve(jjt + lam * eye, rw)
            q = np.clip(q + dq, lo, hi)

        return DiffIKResult(
            q=q, converged=False, pos_error=pos_err, ang_error=ang_err,
            yaw_error=yaw_err, iterations=it,
        )

    def _diff_residual(
        self, q: np.ndarray, target_pos: np.ndarray, pitch: float, roll: float, out: np.ndarray
    ) -> np.ndarray:
        """Fill ``out`` with the five-row task error at ``q`` and return the tool rotation.

        Leaves the evaluation model's data at ``q``, so the caller can ask ``mj_jacSite`` for the
        Jacobian of the very configuration this residual was measured at.
        """

        self._set_q(q)
        pos = self._e_data.site_xpos[self._e_site]
        rot = self._e_data.site_xmat[self._e_site].reshape(3, 3)
        out[0] = target_pos[0] - pos[0]
        out[1] = target_pos[1] - pos[1]
        out[2] = target_pos[2] - pos[2]

        yaw = math.atan2(rot[1, 0], rot[0, 0])
        cz, sz = math.cos(yaw), math.sin(yaw)
        cy, sy = math.cos(pitch), math.sin(pitch)
        cx, sx = math.cos(roll), math.sin(roll)
        # Rz(yaw) @ Ry(pitch) @ Rx(roll), written out: only the first two rows are needed, since
        # the residual keeps only the world x and y components of the angular error.
        d00, d01, d02 = cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx
        d10, d11, d12 = sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx
        d20, d21, d22 = -sy, cy * sx, cy * cx
        # e = 0.5 * sum_i R[:, i] x Rd[:, i], x and y components only.
        out[3] = 0.5 * (
            (rot[1, 0] * d20 - rot[2, 0] * d10)
            + (rot[1, 1] * d21 - rot[2, 1] * d11)
            + (rot[1, 2] * d22 - rot[2, 2] * d12)
        )
        out[4] = 0.5 * (
            (rot[2, 0] * d00 - rot[0, 0] * d20)
            + (rot[2, 1] * d01 - rot[0, 1] * d21)
            + (rot[2, 2] * d02 - rot[0, 2] * d22)
        )
        return rot

    def ik(
        self,
        target_pos: np.ndarray,
        target_pitch: float,
        target_roll: float,
        q0: np.ndarray,
        *,
        damping: float = 1e-2,
        iters: int = 50,
        pos_tol: float = 1e-5,
        ang_tol: float = math.radians(0.05),
    ) -> IKResult:
        """Damped least squares for position (3) + tool pitch + tool roll (2) = 5 targets.

        Joint limits are enforced by clipping every step -- a target that needs a joint past its
        range shows up as a non-zero residual at convergence (or non-convergence), never a silent
        excursion outside the range that a real servo could not reach either.
        """
        lo, hi = self.jnt_range[:, 0], self.jnt_range[:, 1]
        q = np.clip(np.asarray(q0, dtype=float), lo, hi)
        target_pos = np.asarray(target_pos, dtype=float)

        def evaluate(qv: np.ndarray) -> tuple[np.ndarray, float, float, float, float]:
            """residual(5,), its norm, pos_error, pitch_error, roll_error at ``qv``.

            The target frame's yaw is pinned to whatever the arm's *current* heading is, since
            there is no independent wrist-yaw joint to hold it to anything else -- ``r_des``
            (hence the whole residual) is therefore a function of ``qv`` in a more roundabout way
            than the position or a fixed-target orientation error would be.  The Newton step below
            differentiates this exact function by finite differences rather than composing an
            analytic derivative through the yaw extraction: a hand-derived chain rule through
            ``atan2`` and back into a rotation matrix is easy to get subtly wrong in a way that
            still looks plausible, and finite differences are exact for whatever this function
            actually computes.
            """
            self._set_q(qv)
            pos = self._e_data.site_xpos[self._e_site].copy()
            rot = self._e_data.site_xmat[self._e_site].reshape(3, 3).copy()
            yaw_now, _, _ = euler_zyx(rot)
            r_des = _rot_zyx(yaw_now, target_pitch, target_roll)
            pos_err_vec = target_pos - pos
            # The same three cross products `np.cross` would compute, written out: this runs
            # roughly forty times a step and `np.cross`'s axis bookkeeping cost more than the
            # arithmetic did. Same operations in the same order, so same bits.
            e_r_world = 0.5 * _cross_columns(rot, r_des)
            e_r_local = r_des.T @ e_r_world  # local x -> roll, local y -> pitch, local z -> ~0 (yaw)
            roll_e = float(e_r_local[0])
            pitch_e = float(e_r_local[1])
            pos_e = float(np.linalg.norm(pos_err_vec))
            residual = np.concatenate([pos_err_vec, [pitch_e, roll_e]])
            return residual, float(np.linalg.norm(residual)), pos_e, pitch_e, roll_e

        residual, res_norm, pos_err, pitch_err, roll_err = evaluate(q)
        lam = damping**2
        fd_eps = 1e-6
        it = 0
        for it in range(1, iters + 1):
            if pos_err < pos_tol and abs(pitch_err) < ang_tol and abs(roll_err) < ang_tol:
                return IKResult(
                    q=q, converged=True, pos_error=pos_err,
                    pitch_error=pitch_err, roll_error=roll_err, iterations=it - 1,
                )

            jac = np.empty((5, N_DOF))
            for j in range(N_DOF):
                dq = np.zeros(N_DOF)
                dq[j] = fd_eps
                residual_plus, _, _, _, _ = evaluate(q + dq)
                jac[:, j] = (residual_plus - residual) / fd_eps
            self._set_q(q)  # leave the evaluation data at `q`, as every other public method does

            # Levenberg-Marquardt: try a step, accept it only if the residual really shrinks
            # (re-evaluated, not just predicted); otherwise stiffen the damping and retry. Three
            # of the five joints (shoulder_lift, elbow_flex, wrist_flex) share one rotation axis,
            # so the orientation rows of `jac` cannot tell them apart on their own, which makes a
            # plain undamped DLS step prone to overshoot.
            accepted = False
            for _ in range(30):
                jjt = jac @ jac.T + lam * np.eye(5)
                # `jac` is d(residual)/dq (residual = target-like minus current), so a Newton step
                # that reduces the residual is the *negative* damped-least-squares direction.
                dq = -jac.T @ np.linalg.solve(jjt, residual)
                q_try = np.clip(q + dq, lo, hi)
                residual_try, res_norm_try, pos_e, pitch_e, roll_e = evaluate(q_try)
                if res_norm_try < res_norm:
                    q, residual, res_norm = q_try, residual_try, res_norm_try
                    pos_err, pitch_err, roll_err = pos_e, pitch_e, roll_e
                    lam = max(lam * 0.5, 1e-10)
                    accepted = True
                    break
                lam *= 4.0
            if not accepted:
                break  # stuck: neither a smaller nor a more damped step helps any more

        return IKResult(
            q=q, converged=False, pos_error=pos_err,
            pitch_error=pitch_err, roll_error=roll_err, iterations=it,
        )
