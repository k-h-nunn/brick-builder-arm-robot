"""Settling-error map: commanded vs settled tool position over a workspace grid, tool down.

Runs against the virtual bus today and the real bus the day the arm arrives -- same code, same
grid.  See ``brick_builder/calib/servo_workspace.py`` for what each point measures.

    python scripts/hardware/servo_workspace_map.py --virtual \\
        --out results/servo_bringup/workspace_map.json --figure results/servo_bringup/workspace_map.png
    python scripts/hardware/servo_workspace_map.py --port /dev/ttyACM0
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from brick_builder.calib.servo_workspace import WorkspaceGridConfig, measure_workspace_map
from brick_builder.hardware.bus import Bus, BusSettings


def _point_payload(point) -> dict:
    d = asdict(point)
    d["target_xyz_m"] = point.target_xyz_m.tolist()
    d["settled_xyz_m"] = None if point.settled_xyz_m is None else point.settled_xyz_m.tolist()
    return d


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", help="serial port of the servo bus")
    parser.add_argument("--virtual", action="store_true",
                        help="run against a virtual bus with a simulated arm behind it")
    parser.add_argument("--baud", type=int, default=1_000_000)
    parser.add_argument("--out", help="write the map to this JSON file")
    parser.add_argument("--figure", help="write a scatter figure (PNG) here")
    args = parser.parse_args(argv)
    if not args.port and not args.virtual:
        parser.error("give --port, or --virtual to run against the simulator")

    scene = vbus = None
    if args.virtual:
        from brick_builder.calib.servo_bench import bare_virtual_arm

        scene, vbus = bare_virtual_arm()
        vbus.start()
        bus = Bus(vbus.port, BusSettings(timeout_s=0.5))
        kin = scene.kin
    else:
        from brick_builder.hardware.real_world import kinematic_arm

        bus = Bus(args.port, BusSettings(baud=args.baud))
        _, _, kin = kinematic_arm()
    try:
        bus.enable_torque(check=not args.virtual)
        wmap = measure_workspace_map(bus, kin)
    finally:
        try:
            bus.disable_torque()
        except Exception:
            pass
        bus.close()
        if vbus is not None:
            vbus.stop()
            vbus.raise_if_failed()

    print(wmap.table())
    if args.out:
        payload = {"config": asdict(wmap.config), "summary": wmap.summary(),
                  "points": [_point_payload(p) for p in wmap.points]}
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2) + "\n")
    if args.figure:
        from brick_builder.calib.servo_workspace import figure_workspace_map

        Path(args.figure).parent.mkdir(parents=True, exist_ok=True)
        figure_workspace_map(wmap, args.figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
