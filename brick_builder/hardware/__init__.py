"""Hardware-in-the-loop: the STS3215 bus driver, a virtual bus, world adapters and twin sync.

Four layers, each usable without the one above it:

``sts3215``
    The Feetech serial protocol and the STS3215 memory table, as plain bytes.  No port, no
    simulator, no numpy on the hot path.
``bus``
    ``Bus``: the driver over pyserial.  One sync-write and one sync-read per 50 Hz tick, per-joint
    angle and torque limits, a soft stop, a startup self-test and error counters.
``virtual_bus``
    The same protocol answered from the other end of a pseudo-terminal, backed either by a plain
    register file or by a MuJoCo scene through the existing ``ServoBus`` emulation.  This is what
    the driver is proven against while no arm exists.
``real_world`` / ``twin_sync``
    ``RealArmWorld`` and ``HybridWorld`` implement the four ``WorldAdapter`` verbs against the
    driver and a UVC camera; ``TwinSync`` keeps a simulator scene aligned with whatever is on the
    other side of the bus and records the discrepancy metrics.

Status: verified against the virtual bus and the simulator; untested on hardware.  Guide page 08
lists what each measurement on the day the arm arrives fills in.
"""

from .sts3215 import (  # noqa: F401
    JointCalibration,
    Packet,
    ProtocolError,
    REGISTERS,
    SERVO_IDS,
    ServoState,
    so101_calibration,
)
from .bus import Bus, BusSettings, BusState, SelfTestResult  # noqa: F401
from .virtual_bus import (  # noqa: F401
    RegisterServos,
    SimulatedServos,
    VirtualBus,
    WireConfig,
    simulated_bus,
)

__all__ = [
    "Bus", "BusSettings", "BusState", "SelfTestResult",
    "JointCalibration", "Packet", "ProtocolError", "REGISTERS", "SERVO_IDS", "ServoState",
    "so101_calibration",
    "RegisterServos", "SimulatedServos", "VirtualBus", "WireConfig", "simulated_bus",
]
