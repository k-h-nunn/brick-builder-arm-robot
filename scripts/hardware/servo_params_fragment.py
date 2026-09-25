"""Compose the self-test and the system-identification output into a parameter-file fragment.

Reads the JSON ``--out`` files ``arm_selftest.py`` and ``servo_sysid.py`` already write and prints
(and, with ``--out``, saves) the ``arm:`` block in ``configs/params/default.yaml``'s own
``value``/``status``/``measured_by``/``measured_on``/``notes`` format -- ready to paste. Nothing
here writes to ``configs/params/`` itself; pasting the printed block in is a manual step.

    python scripts/hardware/arm_selftest.py --virtual --out results/servo_bringup/selftest.json
    python scripts/hardware/servo_sysid.py --virtual --joint shoulder_lift \\
        --out results/servo_bringup/sysid_shoulder_lift.json
    python scripts/hardware/servo_params_fragment.py \\
        --selftest results/servo_bringup/selftest.json \\
        --sysid results/servo_bringup/sysid_shoulder_lift.json \\
        --out results/servo_bringup/params_fragment.yaml
"""

from __future__ import annotations

import argparse
import json
import types
from pathlib import Path

from brick_builder.calib.servo_params import build_fields, render_param_fragment
from brick_builder.calib.servo_selftest import JointSelfTest, SelfTestConfig, SelfTestReport

_TUPLE_FIELDS = ("model_number", "firmware_version", "angle_limit_ticks", "dead_band_ticks")


def _load_selftest(path: str) -> SelfTestReport:
    payload = json.loads(Path(path).read_text())
    joints = []
    for row in payload["joints"]:
        row = dict(row)
        for name in _TUPLE_FIELDS:
            if row.get(name) is not None:
                row[name] = tuple(row[name])
        joints.append(JointSelfTest(**row))
    return SelfTestReport(joints, SelfTestConfig())


def _load_fit(path: str) -> types.SimpleNamespace:
    """A plain namespace with a fitted :class:`~scripts.hardware.servo_sysid.ServoFit`'s fields --
    :func:`~brick_builder.calib.servo_params.build_fields` only reads attributes off it."""
    return types.SimpleNamespace(**json.loads(Path(path).read_text()))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--selftest", required=True, help="arm_selftest.py --out JSON")
    parser.add_argument("--sysid", help="servo_sysid.py --out JSON (a single fit, not --validate)")
    parser.add_argument("--out", help="write the fragment to this file")
    parser.add_argument("--real", action="store_true",
                        help="mark the fragment as a real-hardware measurement, not a virtual-bus "
                             "self-consistency check")
    parser.add_argument("--measured-by", default="servo-bringup kit")
    args = parser.parse_args(argv)

    report = _load_selftest(args.selftest)
    fit = _load_fit(args.sysid) if args.sysid else None
    fields = build_fields(report, fit)
    text = render_param_fragment(fields, measured_by=args.measured_by, virtual=not args.real)
    print(text)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
