"""Arm bring-up self-test: ping, identity and safety registers, a torque-off range read, one
small commanded move per joint -- a pass/fail table.

Runs against the virtual bus today and the real bus the day the arm arrives, same code, same
table.  Nothing here replaces the gate :meth:`Bus.enable_torque` already runs before energizing
anything; this is the longer report a teammate reads once before trusting the bus at all.  See
``brick_builder/calib/servo_selftest.py`` for what each check does and why.

    python scripts/hardware/arm_selftest.py --virtual
    python scripts/hardware/arm_selftest.py --port /dev/ttyACM0
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from brick_builder.calib.servo_selftest import SelfTestConfig, run_self_test
from brick_builder.hardware.bus import Bus, BusSettings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", help="serial port of the servo bus")
    parser.add_argument("--virtual", action="store_true",
                        help="run against a virtual bus with a simulated arm behind it")
    parser.add_argument("--baud", type=int, default=1_000_000)
    parser.add_argument("--out", help="write the report to this JSON file")
    args = parser.parse_args(argv)
    if not args.port and not args.virtual:
        parser.error("give --port, or --virtual to run against the simulator")

    scene = vbus = None
    if args.virtual:
        from brick_builder.calib.servo_bench import bare_virtual_arm

        scene, vbus = bare_virtual_arm()
        vbus.start()
        bus = Bus(vbus.port, BusSettings(timeout_s=0.5))
    else:
        bus = Bus(args.port, BusSettings(baud=args.baud))
    try:
        report = run_self_test(bus, vbus=vbus, scene=scene, config=SelfTestConfig())
    finally:
        bus.close()
        if vbus is not None:
            vbus.stop()
            vbus.raise_if_failed()

    print(report.table())
    if args.out:
        payload = {
            "ok": report.ok,
            "joints": [asdict(j) for j in report.joints],
        }
        Path(args.out).write_text(json.dumps(payload, indent=2) + "\n")
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
