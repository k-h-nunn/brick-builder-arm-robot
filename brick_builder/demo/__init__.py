"""`python -m brick_builder.demo <name> [--record out.mp4] [options]`.

Every demo builds a small, deterministic scenario, runs it, and writes one file to
`docs/media/demos/<name>.<ext>` by default (`--record path` overrides that). Every demo also runs
completely headless -- `MUJOCO_GL=egl` and matplotlib's `Agg` backend, exactly what the test suite
already sets -- because the acceptance criterion is "each demo runs headless with `--record` and
writes a playable mp4/PNG," not "each demo opens a window."

Subcommands: ``scene``, ``teleop``, ``press``, ``rocking``, ``force-map``, ``policy``,
``tokens``. Each module's own docstring documents its
demo-specific arguments (everything after the subcommand name, e.g.
``python -m brick_builder.demo policy --checkpoint runs/reach/checkpoint.pt``).
"""

from __future__ import annotations

import argparse
import importlib

DEMOS: dict[str, str] = {
    "scene": "brick_builder.demo.scene_demo",
    "teleop": "brick_builder.demo.teleop_demo",
    "press": "brick_builder.demo.press_demo",
    "rocking": "brick_builder.demo.rocking_demo",
    "force-map": "brick_builder.demo.force_map_demo",
    "policy": "brick_builder.demo.policy_demo",
    "tokens": "brick_builder.demo.tokens_demo",
}


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(prog="python -m brick_builder.demo", description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("name", choices=sorted(DEMOS), help="which demo to run")
    parser.add_argument("--record", default=None,
                         help="output file (.mp4 or .png); default: docs/media/demos/<name>.<ext>")
    parser.add_argument("--seed", type=int, default=0)
    # `parse_known_args` rather than a `nargs=REMAINDER` positional: REMAINDER swallows every
    # remaining token unconditionally once positional matching reaches it, which would eat
    # `--record`/`--seed` themselves whenever they come after the demo name (the normal case,
    # e.g. `press --level 2 --record out.mp4`). Leftover, unrecognized tokens are each demo's own
    # arguments (e.g. `--level 2`), parsed by that demo's own `argparse.ArgumentParser`.
    args, extra = parser.parse_known_args(argv)

    module = importlib.import_module(DEMOS[args.name])
    return module.run(record=args.record, seed=args.seed, extra=extra)


if __name__ == "__main__":
    main()
