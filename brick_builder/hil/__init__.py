"""Hardware in the loop: the run-time controller as one C99 library, and the host that drives it.

`firmware/control_lib` is the controller -- target-sequence tracking, the inverse kinematics, the
joint trim and the servo bus behind two function pointers. This package builds it, writes the
target bundle its digital twin work goes into, speaks the wire protocol to it and runs a
placement through it, so the same library can be exercised three ways:

    --controller c                      in this process, through ctypes
    --controller process                as a native process over a byte stream
    --controller serial:/dev/ttyACM0    over a UART to the microcontroller

The ladder and what each rung is measured on are in `docs/guide/13-hardware-in-the-loop.md`.
"""

from .library import build, load  # noqa: F401
