"""Features from slot numbers, and the inverse that turns a feature error into a tool move.

A *reference* is something the controller may line itself up on, described only the way slot
states describe it: a mean colour and an ellipse size. `select_slot` finds it in both the twin's
intended state and the camera's actual state, independently -- the two scenes are different
models, so a slot index means nothing across them, but "the yellow 2x4" means the same in both.

The held brick is never a reference. It rides with the camera, so its blob barely moves in the
image no matter where the tool goes, and asking the loop to servo on it would be asking it to
chase its own hand.

The inverse is the plain analytic one: a tool change of `-lambda * J^+ * (s - s*)`, saturated,
with `lambda` shrinking as the error comes in. `J` was measured in the twin (see `twin.py`), so
nothing here needs to know the camera's intrinsics or where anything is in the world.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..perception.scene_model import SceneState
from ..perception.slot_ops import select_slot


@dataclass(frozen=True)
class Reference:
    """One thing to line up on, named only by what a slot state publishes about it."""

    name: str
    colour: tuple[float, float, float]
    size: tuple[float, float] | None = None
    colour_tol: float = 0.45
    size_tol: float = 0.9
    min_mass: float | None = None
    require_inside: bool = True
    """Skip this reference when its blob is running off the edge of the frame -- a truncated blob's
    centre is the centre of the visible part, which is not the feature the twin measured."""
    max_border: float = 0.12


def find_held(state: SceneState, expected: np.ndarray, *, exclude=(),
              max_distance: float = 0.45) -> int | None:
    """Which slot is the brick in the hand, by where it must be rather than what colour it is.

    A brick between the jaws is in almost the same place in the image every frame -- that is what
    makes it useless as a thing to line up on and perfect for identifying by position. `expected`
    is where the twin says it sits at this pose. Among the slots not already claimed, this takes
    the one nearest that spot, breaking ties towards the nearer one in depth order, which is what
    a brick held in front of the lens is.

    Colour and size alone got this wrong: the reference blobs are matched greedily in order, and
    a single mis-assignment early left the held brick pointing at another object's slot -- worth
    nine millimetres of phantom in-hand offset in a world where nothing was displaced.
    """
    expected = np.asarray(expected, dtype=float)
    best, best_score = None, np.inf
    for k in range(state.K):
        if k in exclude or state.is_empty(k) or state.mass(k) > 0.5:
            continue
        d = float(np.linalg.norm(np.asarray(state.centre(k)) - expected))
        if d > max_distance:
            continue
        score = d + 0.05 * float(state.depth[k])
        if score < best_score:
            best, best_score = k, score
    return best


@dataclass
class FeatureSet:
    """Which references were found this frame, where, and how much each is worth."""

    centres: dict[str, tuple[float, float] | None] = field(default_factory=dict)
    slots: dict[str, int | None] = field(default_factory=dict)
    weights: dict[str, float] = field(default_factory=dict)
    order: tuple[str, ...] = ()


def damped_inverse(J: np.ndarray, *, relative: float = 0.15):
    """A pseudo-inverse that refuses to invert what the features cannot see.

    `numpy.linalg.pinv` happily divides by a singular value of 1e-3 and turns a tenth of a pixel
    of noise into centimetres of command along a direction the camera has almost no opinion
    about. Measured: in a world where nothing was displaced, that alone walked a brick 113 mm.

    Singular values below `relative` times the largest are dropped, not damped towards zero --
    a direction that weak is not "hard to see", it is not seen, and the honest response is to
    leave it alone rather than to guess quietly. Returns the inverse, the singular values, and
    which of them were kept, so a caller can log the condition number and say what it ignored.
    """
    J = np.asarray(J, dtype=float)
    if J.size == 0:
        return np.zeros((J.shape[1], J.shape[0])), np.zeros(0), np.zeros(0, dtype=bool)
    U, sv, Vt = np.linalg.svd(J, full_matrices=False)
    keep = sv >= relative * sv[0] if sv.size and sv[0] > 0 else np.zeros_like(sv, dtype=bool)
    inv = np.zeros((J.shape[1], J.shape[0]))
    for i in range(sv.size):
        if keep[i]:
            inv += np.outer(Vt[i], U[:, i]) / sv[i]
    return inv, sv, keep


def _centred(v: np.ndarray) -> np.ndarray:
    """Subtract the mean x and the mean y from a stack of (x, y) feature pairs."""
    pairs = np.asarray(v, dtype=float).reshape(-1, 2)
    return (pairs - pairs.mean(axis=0)).reshape(-1)


def reference_features(state: SceneState, references) -> FeatureSet:
    """Find each reference's slot in `state` and read its centre, from slot numbers only."""
    out = FeatureSet(order=tuple(r.name for r in references))
    taken: list[int] = []
    for ref in references:
        k = select_slot(
            state, colour=ref.colour, colour_tol=ref.colour_tol,
            size=ref.size, size_tol=ref.size_tol, min_mass=ref.min_mass, exclude=taken,
        )
        if k is None:
            out.centres[ref.name] = None
            out.slots[ref.name] = None
            out.weights[ref.name] = 0.0
            continue
        border = state.border_fraction(k)
        if ref.require_inside and border > ref.max_border:
            out.centres[ref.name] = None
            out.slots[ref.name] = None
            out.weights[ref.name] = 0.0
            continue
        taken.append(k)
        out.centres[ref.name] = state.centre(k)
        out.slots[ref.name] = k
        out.weights[ref.name] = max(1.0 - float(border), 0.0)
    return out


@dataclass
class VisualServo:
    """`-lambda * J^+ * (s - s*)`, saturated, with a done test on the error itself."""

    step: "object"
    lam: float = 0.45
    max_xy: float = 0.0025
    """Most the tool target may move in one vision tick [m]."""
    max_spin: float = math.radians(1.5)
    settle_ticks: int = 3
    near: float = 3.0
    """Below `near` * tolerance the gain is halved, so the last millimetre does not ring."""

    smooth_in_hand: float = 0.35
    """Low-pass on the in-hand reading. The jaws cut into the brick's blob and the brick shifts a
    little under the pinch, so a single frame's answer jitters by a few tenths of a millimetre --
    enough to stop a 0.4 mm tolerance ever being met."""

    freeze_in_hand: int = 8
    """Vision ticks after which the in-hand reading stops being updated.

    The brick sits where it sits: its offset in the jaws is one constant, not something that
    changes as the arm moves. Measured every tick it does drift a little anyway -- the jaws
    occlude a different sliver of the blob from a different position -- and because the tool is
    moving *because of* that reading, the drift feeds back on itself and the loop settles a
    millimetre or so short. Reading it a few times, averaging, and then holding it breaks that
    loop and leaves the plate alignment free to converge on its own."""

    last_singular_values: np.ndarray = field(default_factory=lambda: np.zeros(0), init=False)
    last_kept: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool), init=False)
    last_dead: bool = field(default=False, init=False)
    last_in_hand: np.ndarray = field(default_factory=lambda: np.zeros(3), init=False)
    last_in_hand_sv: np.ndarray = field(default_factory=lambda: np.zeros(0), init=False)
    last_yaw_weight: float = field(default=0.0, init=False)

    _hits: int = field(default=0, init=False)
    _d_b: np.ndarray | None = field(default=None, init=False)
    _n_b: int = field(default=0, init=False)

    def reset(self) -> None:
        self._hits = 0
        self.reset_in_hand()

    def reset_in_hand(self) -> None:
        """Start the in-hand average again.

        Worth doing when the tool arrives where it will do the fine work: the brick's blob is
        measured from wherever the camera happens to be, and an average carried over from a pose
        200 mm away is an average of a different measurement.
        """
        self._d_b = None
        self._n_b = 0

    truncate_at: float = 0.15
    """Singular values below this fraction of the largest are dropped from the inverse."""

    dead_band_mm: float = 0.35
    """Below this much measured tool error, command nothing: it is inside the noise.

    Measured floors: about 0.3 mm over the baseplate with three reference blobs in view."""

    dead_band_single_mm: float = 1.2
    """The same, when only one reference blob is matched -- a much noisier reading. Measured on
    the supply area, where a single small blob gave a floor near 1.7 mm."""

    min_yaw_weight: float = 0.05
    """How strongly a blob's shape must carry an orientation before its yaw error is believed."""

    min_refs: int = 2
    require_brick: bool = True
    """Whether a missing in-hand reading counts as being blind. False when there is nothing in
    the hand yet -- the grasp side lines up on the brick lying on the table instead."""
    max_in_hand: float = 0.003
    """How far off-centre the brick could physically be sitting in the jaws [m].

    Three millimetres for a task that starts with the brick already held, where the only error is
    a couple of millimetres of jitter along the jaw faces. A task that picks the brick up off the
    table needs more room -- a brick lying 13 mm from the supply row was measured 8 mm off-centre
    in the jaws after the grasp -- so `place.TrackingEpisode` raises it for those runs rather
    than making every run pay for the looser bound.
    """
    brick_ref: Reference | None = None
    """When set, the loop lines up the **brick**, not the tool: the brick's own blob says how far
    off-centre it is sitting in the hand, and that is subtracted from the plate's answer."""

    def update(self, actual: SceneState, references, reference: SceneState | None = None,
               *, dry_run: bool = False, relative: bool = False) -> dict:
        """One vision tick.

        Two readings, two Jacobians, one answer:

        - `d_r = Jr^+ (r - r*)` -- where the tool sits relative to the baseplate, in metres.
          The plate's real position is unknown, so this is the only thing that says it.
        - `d_b = Jb^+ (b - b*)` -- how far off-centre the brick is sitting in the jaws, in
          metres of world motion. Closing the jaws cannot fix this, so the arm has to.

        The brick's error against the cell is their sum, and the tool is commanded by minus it.
        """
        st = self.step
        found = reference_features(actual, list(references))
        if self.brick_ref is not None and st.b_star is not None:
            # By position, not by colour: see `find_held`.
            taken = [k for k in found.slots.values() if k is not None]
            k_held = find_held(actual, st.b_star[:2], exclude=taken)
            found.order = tuple(found.order) + (self.brick_ref.name,)
            found.slots[self.brick_ref.name] = k_held
            found.centres[self.brick_ref.name] = (None if k_held is None
                                                  else actual.centre(k_held))
            found.weights[self.brick_ref.name] = (0.0 if k_held is None
                                                  else max(1.0 - actual.border_fraction(k_held),
                                                           0.0))
        # Sequence tracking: the target is the reference *frame* the controller is currently
        # asking the world for, not one fixed goal picture. `reference` is that frame's own
        # scene state, read through exactly the same feature extraction as the camera's, so the
        # difference between them is a difference of two like things.
        want = None
        if reference is not None:
            want = reference_features(reference, list(references))
            if self.brick_ref is not None and st.b_star is not None:
                # The target frame's own held brick, found the same way.
                taken_w = [k for k in want.slots.values() if k is not None]
                k_w = find_held(reference, st.b_star[:2], exclude=taken_w)
                want.order = tuple(want.order) + (self.brick_ref.name,)
                want.slots[self.brick_ref.name] = k_w
                want.centres[self.brick_ref.name] = (None if k_w is None
                                                     else reference.centre(k_w))
                want.weights[self.brick_ref.name] = 0.0 if k_w is None else 1.0
        names = [n for n in st.feature_names if found.centres[n] is not None]
        blind = len(names) < self.min_refs
        if not blind and self.brick_ref is not None and self.require_brick:
            blind = found.centres.get(self.brick_ref.name) is None
        if blind:
            return {
                "delta": np.zeros(3), "err_m": float("nan"), "err_mm": float("nan"),
                "done": False, "seen": len(names), "found": found, "blind": True,
                "d_r": np.zeros(3), "d_b": np.zeros(2),
            }

        idx = []
        for n in names:
            j = st.feature_names.index(n)
            idx.extend((2 * j, 2 * j + 1))
        idx = np.asarray(idx, dtype=int)

        if want is not None:
            names = [n for n in names if want.centres.get(n) is not None]
            if len(names) < self.min_refs:
                return {
                    "delta": np.zeros(3), "err_m": float("nan"), "err_mm": float("nan"),
                    "done": False, "seen": len(names), "found": found, "blind": True,
                    "d_r": np.zeros(3), "d_b": np.zeros(2),
                }
            idx = []
            for n in names:
                j = st.feature_names.index(n)
                idx.extend((2 * j, 2 * j + 1))
            idx = np.asarray(idx, dtype=int)
            target = np.concatenate([np.asarray(want.centres[n], dtype=float) for n in names])
        else:
            target = st.s_star[idx]
        s = np.concatenate([np.asarray(found.centres[n], dtype=float) for n in names])
        if relative:
            # Relative geometry only: each blob measured against the centre of gravity of all of
            # them, so a common shift of the whole picture cancels. Once the correction is
            # frozen and the tool is retreating, "is the camera where the recording had it" is
            # no longer the question; "is the brick where the target puts it, relative
            # to the baseplate and the bricks already on it" is, and that is what this measures.
            s = _centred(s)
            target = _centred(target)
        e = s - target
        w = np.repeat(np.array([found.weights[n] for n in names], dtype=float), 2)
        inv, sv, keep = damped_inverse(st.J[idx] * w[:, None], relative=self.truncate_at)
        self.last_singular_values = sv
        self.last_kept = keep
        d_r = inv @ (e * w)
        d_r = np.array([d_r[0], d_r[1], 0.0])
        self.last_yaw_weight = 0.0
        if st.spin_gain is not None and abs(st.spin_gain) > 0.2:
            from .twin import _pair_angle, _wrap, blob_orientation, orientation_error

            z_obs = z_star = None
            fold = int(st.angle_period)
            if st.angle_mode == "pair" and len(names) >= 2:
                star = st.spin_star if want is None else _pair_angle(target)
                d_r[2] = _wrap(_pair_angle(s) - float(star)) / st.spin_gain
                self.last_yaw_weight = 1.0
            elif st.angle_mode == "blob" and found.slots.get(st.angle_slot) is not None:
                z_obs, fold = blob_orientation(actual, found.slots[st.angle_slot])
                z_star = st.spin_star
                if want is not None and want.slots.get(st.angle_slot) is not None:
                    z_star, _ = blob_orientation(reference, want.slots[st.angle_slot])
            if z_obs is not None and z_star is not None:
                # From the ratio, never from two wrapped angles: see `twin.orientation_error`.
                err, weight = orientation_error(z_obs, z_star, fold)
                self.last_yaw_weight = weight
                if weight >= self.min_yaw_weight:
                    d_r[2] = err / st.spin_gain

        d_b = np.zeros(2)
        self.last_in_hand = np.zeros(3)
        if self.brick_ref is not None and st.J_brick is not None \
                and found.slots.get(self.brick_ref.name) is not None:
            from .twin import brick_features

            # Only as many features as the Jacobian was built from: the placement task keeps
            # the two-column centre-only form it was measured with, the pick uses the full row.
            b = brick_features(actual, found.slots[self.brick_ref.name])[:st.b_star.shape[0]]
            # Deliberately NOT taken from the reference frame. The held brick's row is read
            # against the twin posed where the tool is being *commanded*, because the reference
            # frame was recorded from a pose several millimetres away and the brick's silhouette
            # shifts a little with the camera.
            inv_b, sv_b, _ = damped_inverse(st.J_brick, relative=self.truncate_at)
            self.last_in_hand_sv = sv_b
            full = inv_b @ (b - st.b_star)
            # Three unknowns come back -- across the jaws, along them, and vertical. Only the two
            # in the plane are a placement error; the vertical is how deep the brick sits in the
            # jaws, which is not something the tool can or should correct sideways. Solving for
            # it is what stops it being read as a sideways offset.
            self.last_in_hand = full
            d_b = full[:2]
            # A brick can only sit a couple of millimetres off-centre between jaws that are
            # 31.8 mm apart. A larger reading is the jaws cutting into the blob, not the brick
            # having moved, so it is clamped rather than believed.
            n_b = float(np.linalg.norm(d_b))
            if n_b > self.max_in_hand:
                d_b = d_b * (self.max_in_hand / n_b)
            # `dry_run` is how the caller asks "what would this reference frame look like?"
            # while searching for the frame the world is really at. A search that averaged the
            # in-hand reading in nine times per tick would bury the real one under the frames it
            # only looked at.
            if dry_run:
                d_b = self._d_b if self._d_b is not None else d_b
            else:
                if self._d_b is None:
                    self._d_b = d_b
                elif self._n_b < self.freeze_in_hand:
                    self._d_b = ((1.0 - self.smooth_in_hand) * self._d_b
                                 + self.smooth_in_hand * d_b)
                self._n_b += 1
                d_b = self._d_b

        err_vec = np.array([d_r[0] + d_b[0], d_r[1] + d_b[1], d_r[2]])
        err_m = float(np.linalg.norm(err_vec[:2]))
        dead = (self.dead_band_single_mm if len(names) < 2 else self.dead_band_mm) * 1e-3
        self.last_dead = err_m < dead
        if self.last_dead:
            err_vec = np.zeros(3)
        delta = -self.lam * err_vec
        if err_m < self.near * st.tol:
            delta = delta * 0.5

        n_xy = float(np.linalg.norm(delta[:2]))
        if n_xy > self.max_xy:
            delta[:2] *= self.max_xy / n_xy
        delta[2] = float(np.clip(delta[2], -self.max_spin, self.max_spin))

        if not dry_run:
            self._hits = self._hits + 1 if err_m <= st.tol else 0
        return {
            "delta": delta, "err_m": err_m, "err_mm": 1e3 * err_m,
            "done": self._hits >= self.settle_ticks, "seen": len(names),
            "found": found, "blind": False, "d_r": d_r, "d_b": d_b,
        }
