"""A bare arm on the virtual bus: what every bring-up script (self-test, sysid, the settling-error
map) drives against before real hardware exists, and the one place that builds it.

No table, no bricks, no plate, no camera -- gravity on, nothing else, which is what a bus-only
measurement of the arm itself wants: the bring-up commands do not need a Scene, only a Bus.
"""

from __future__ import annotations

import numpy as np

from ..hardware.virtual_bus import VirtualBus
from ..robot.patch_so101 import ArmConfig
from ..robot.servo_model import ServoConfig


def bare_virtual_arm(
    arm: ArmConfig = ArmConfig(), *, seed: int = 0, sample: dict | None = None,
    servo_cfg: ServoConfig | None = None,
):
    """A ``(scene, VirtualBus)`` pair: the arm alone, gravity on, torque not yet enabled.

    ``sample`` is a domain-randomization draw from :mod:`brick_builder.scene.randomize` (typically
    the ``dynamics`` group): when given, it is written onto the compiled model with
    :func:`~brick_builder.scene.randomize.apply` before the scene resets, exactly the way the
    training randomization applies it, so a validation run exercises the same code path rather
    than a second, hand-built one. ``servo_cfg`` defaults to the command path with no delay jitter
    and no read noise -- a bring-up measurement whose own baseline is already noisy cannot say
    whether a discrepancy is the arm or the measurement.
    """
    from ..hardware.virtual_bus import simulated_bus
    from ..scene.compose import SceneConfig, build_scene, reset_scene

    scene = build_scene(SceneConfig(arm=arm, held=None, placed=[], tray=False, table=False,
                                    cameras=(), damage_monitor=False))
    if sample:
        from ..scene.randomize import apply

        apply(scene, sample)
    reset_scene(scene, np.random.default_rng(seed))
    if servo_cfg is None:
        servo_cfg = ServoConfig(action_delay_jitter_s=0.0, read_noise_ticks=0.0)
    vbus: VirtualBus = simulated_bus(scene, servo_cfg=servo_cfg, seed=seed)
    return scene, vbus
