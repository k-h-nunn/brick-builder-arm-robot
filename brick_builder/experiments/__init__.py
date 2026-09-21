"""``python -m brick_builder.experiments <name> --params FILE --out DIR [experiment-specific flags]``

Every experiment reads a parameter file (``brick_builder.params.load_params`` -- defaults to
``configs/params/default.yaml`` when ``--params`` is not given), runs something with it, and
writes ``<out>/<name>/<timestamp>/`` containing:

- ``report.md`` -- what was run, the parameter table with each value's status, the results, and
  a plain-language reading of them;
- ``data.json`` -- the same results as data;
- zero or more figures (PNG).

Experiments (see each module's own docstring for its extra flags):

============================  ===================================================================
``params``                    print and check a parameter file
``press-bench``                push a 2x2 and a 2x4 on and pull them off with the virtual force
                               gauge; fits holding force and asymmetry if you supply measurements
``seating-feasibility``       can the arm's downward force seat a brick, straight or tilted?
``force-map``                 downward force available over the workspace, at your arm's settings
``grasp-hold``                can the stock jaws hold a brick by friction alone?
``pick-and-place``            full pick-and-place, N randomized trials, Wilson interval
============================  ===================================================================

Example::

    .venv/bin/python -m brick_builder.experiments press-bench --params my_params.yaml --out results/
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

from ..params import DEFAULT_PARAMS_PATH, ParamsError, load_params
from .common import ExperimentContext, make_out_dir

EXPERIMENTS: dict[str, str] = {
    "params": "params_cmd",
    "press-bench": "press_bench",
    "seating-feasibility": "seating_feasibility",
    "force-map": "force_map",
    "grasp-hold": "grasp_hold",
    "pick-and-place": "pick_and_place",
}


def _load_experiment_module(name: str):
    """Import an experiment submodule by a package-relative name (``importlib``, not a literal
    ``"brick_builder.experiments.<name>"`` string), so this keeps working under whatever this
    package is renamed to later."""
    return importlib.import_module(f".{EXPERIMENTS[name]}", package=__package__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m brick_builder.experiments",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("name", choices=sorted(EXPERIMENTS), help="which experiment to run")
    parser.add_argument(
        "--params", default=None,
        help=f"a parameter YAML file, merged over the shipped defaults (default: {DEFAULT_PARAMS_PATH})",
    )
    parser.add_argument("--out", default="results", help="base output directory (default: results)")
    parser.add_argument("--seed", type=int, default=0, help="random seed for experiments that need one")
    args, extra = parser.parse_known_args(argv)

    try:
        params = load_params(args.params)
    except ParamsError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    module = _load_experiment_module(args.name)

    exp_parser = argparse.ArgumentParser(
        prog=f"python -m brick_builder.experiments {args.name}",
        description=getattr(module, "__doc__", None),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_arguments = getattr(module, "add_arguments", None)
    if add_arguments is not None:
        add_arguments(exp_parser)
    try:
        exp_args = exp_parser.parse_args(extra)
    except SystemExit as exc:
        return int(exc.code or 2)

    out_dir = make_out_dir(args.out, args.name)
    shown_argv = sys.argv[1:] if argv is None else list(argv)
    ctx = ExperimentContext(
        name=args.name,
        params=params,
        out_dir=out_dir,
        seed=args.seed,
        argv=["python", "-m", __package__, *shown_argv],
        extra=exp_args,
    )
    module.run(ctx)
    print(f"\nWrote {out_dir / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
