"""``pick-and-place``: full pick-and-place, N randomized trials, Wilson interval.

    python -m brick_builder.experiments pick-and-place --params my_params.yaml --trials 20

Runs ``brick_builder.control.pick_place``'s scripted pick-and-place sequence ``trials`` times with
randomized tray jitter and placement error, and reports the success rate with a 95% Wilson
interval and a breakdown of why the runs that failed did.

``brick_builder/control/pick_place.py`` is still under active development elsewhere in this
codebase, so this always checks, at run time, that it can (a) be imported, (b) still has the
functions and parameters this module needs, and (c) actually completes a run without raising or
hanging, before trusting it. If any of that fails, this prints and writes a clear "not available
yet" report (with the exact error) instead of crashing the whole experiments CLI -- see
:func:`_load_pick_place`.
"""

from __future__ import annotations

import argparse
import inspect
import re
import signal
import traceback
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from .. import params as params_mod
from ..eval.wilson import wilson_interval
from .common import ExperimentContext, SERIES_COLOURS, parameter_table_md, report_header, save_figure, write_data_json, write_report

RELEVANT_PREFIXES = ("bricks.", "arm.", "workspace.")

_REQUIRED_ATTRS = ("bench", "PickPlaceConfig")
_REQUIRED_BENCH_PARAMS = ("trials", "scene", "pad_friction", "strategy", "seed0", "place_error", "dr_level")


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--place-error-mm", type=float, default=1.0,
                         help="lateral placement error the arm must survive, mm (default 1.0)")
    parser.add_argument("--timeout-s", type=int, default=600,
                         help="give up and report 'not available' if the run has not finished by "
                              "then (default 600 s: each trial runs the full scripted sequence "
                              "through real physics and measures roughly 10-15 s, so the default "
                              "20 trials need several minutes -- this is the slowest of the six "
                              "experiments)")


class _NotAvailable(Exception):
    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


class _Timeout(Exception):
    pass


def _check_pick_place_module(pp) -> None:
    """Does ``pp`` (whatever ``brick_builder.control.pick_place`` currently is) look usable?

    Pure and side-effect-free -- it never imports anything itself -- so it can be tested against
    a fake stand-in module without fighting Python's module cache for a real import failure.
    Raises :class:`_NotAvailable` with a specific reason; returns nothing on success.
    """
    missing = [name for name in _REQUIRED_ATTRS if not hasattr(pp, name)]
    if missing:
        raise _NotAvailable(f"brick_builder.control.pick_place is missing: {', '.join(missing)}")

    try:
        sig = inspect.signature(pp.bench)
    except (TypeError, ValueError) as exc:
        raise _NotAvailable(f"brick_builder.control.pick_place.bench has no inspectable signature: {exc}") from exc

    missing_params = [p for p in _REQUIRED_BENCH_PARAMS if p not in sig.parameters]
    if missing_params:
        raise _NotAvailable(
            f"brick_builder.control.pick_place.bench is missing parameter(s): {', '.join(missing_params)} "
            f"(has: {', '.join(sig.parameters)})"
        )


def _load_pick_place():
    """Import ``brick_builder.control.pick_place`` and check its shape. Raises :class:`_NotAvailable`
    with a specific, human-readable reason on any problem; never lets an exception from the
    in-progress module itself propagate up."""
    try:
        from ..control import pick_place as pp
    except BaseException as exc:  # a syntax error at import time raises SyntaxError, not Exception
        raise _NotAvailable(
            f"brick_builder.control.pick_place failed to import: {type(exc).__name__}: {exc}"
        ) from exc
    _check_pick_place_module(pp)
    return pp


def _build_scene(params, pp):
    """A scene built from this file's own arm/snap/friction parameters, for ``bench(scene=...)``.

    Deliberately not ``pp.make_scene()``, which only takes a bare ``pad_friction`` float and
    builds its arm as ``ArmConfig(pad_friction=pad_friction)`` -- every other arm field (supply
    voltage, torque limit, P gain, backlash) would stay at the bare simulator default, silently
    ignoring most of this file's ``arm.*`` section. Built the same way ``grasp_hold.py`` does:
    through ``params.filter_dataclass_kwargs`` so a field ``SceneConfig``/``HeldBrick`` drops
    does not break this call either.
    """
    from ..scene.compose import build_scene

    scene_cfg, skipped = params_mod.to_scene_config(
        params, held=None, placed=[], tray=True, tray_bricks=2,
        tray_brick=(2, 4), cameras=(),
    )
    return build_scene(scene_cfg), skipped


def _run_with_timeout(fn, seconds: int):
    def handler(signum, frame):
        raise _Timeout(f"did not finish within {seconds} s")

    has_alarm = hasattr(signal, "SIGALRM")
    if not has_alarm:  # pragma: no cover - this project only runs on Linux
        return fn()
    old = signal.signal(signal.SIGALRM, handler)
    signal.alarm(seconds)
    try:
        return fn()
    finally:
        signal.signal(signal.SIGALRM, old)
        signal.alarm(0)


def _not_available_report(ctx: ExperimentContext, reason: str) -> None:
    print("pick-and-place: NOT AVAILABLE YET")
    print(f"  reason: {reason}")

    fig, ax = plt.subplots(figsize=(9.0, 4.5))
    ax.axis("off")
    ax.text(0.5, 0.62, "pick-and-place: not available yet", ha="center", va="center",
            fontsize=20, fontweight="bold", color="#D55E00", wrap=True)
    ax.text(0.5, 0.38, reason, ha="center", va="center", fontsize=12, color="#333333", wrap=True)
    fig_path = save_figure(fig, ctx.out_dir, "pick_and_place.png")

    sections = [
        report_header(
            ctx, "pick-and-place: not available yet",
            "brick_builder.control.pick_place is still under development and is not usable "
            "right now. Nothing was run."
        ),
        f"## Why\n\n{reason}\n",
        f"![Not available]({fig_path.name})\n",
        "## Reading\n\n- Re-run this experiment once `brick_builder/control/pick_place.py` has "
        "settled down; this module checks its shape fresh on every run, so nothing here needs "
        "to change.",
    ]
    write_report(ctx.out_dir, sections)
    write_data_json(ctx.out_dir, {"available": False, "reason": reason})


def _failure_category(reason: str) -> str:
    """Group a per-trial failure reason into its general category.

    ``brick_builder.control.pick_place._diagnose`` embeds a per-trial numeric detail in one reason --
    ``"partial seat (2.20 mm short)"`` -- which is a different string almost every trial. Without
    this, a 20-trial run's failure breakdown is 14 bars of count 1 each instead of one meaningful
    "partial seat" bar; stripping the trailing parenthetical groups by category the way every
    other reason (``"never engaged"``, ``"wrong cell"``, ``"overload trip"``, ...) already is.
    """
    return re.sub(r"\s*\([^)]*\)\s*$", "", reason).strip()


def _figure(result: dict, ci, out_dir: Path) -> Path:
    counts = Counter(_failure_category(r["failure"]) for r in result["rows"] if not r["success"])
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    if counts:
        items = sorted(counts.items(), key=lambda kv: -kv[1])
        labels = [k for k, _ in items]
        values = [v for _, v in items]
        ax.barh(range(len(labels)), values, color=SERIES_COLOURS[1])
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=11)
        ax.invert_yaxis()
        ax.set_xlabel("count", fontsize=12)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        for i, v in enumerate(values):
            ax.text(v, i, f"  {v}", va="center", fontsize=11)
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "every trial succeeded", ha="center", va="center", fontsize=16,
                color="#009E73")
    fig.suptitle(
        f"pick-and-place: {ci.p_hat:.0%} success ({ci.n} trials, "
        f"95% CI [{ci.low:.0%}, {ci.high:.0%}])",
        fontsize=13,
    )
    return save_figure(fig, out_dir, "pick_and_place.png")


def run(ctx: ExperimentContext) -> None:
    params = ctx.params
    trials = getattr(ctx.extra, "trials", 20)
    place_error_mm = getattr(ctx.extra, "place_error_mm", 1.0)
    timeout_s = getattr(ctx.extra, "timeout_s", 600)

    # Everything through the report/figure/data.json is inside one try: a KeyError from an
    # unexpected result-dict shape (the module returning something the right *signature* check
    # could not have caught) is exactly as much "not stable yet" as bench() itself raising, and
    # must not crash the rest of the experiments CLI either.
    try:
        pp = _load_pick_place()
        scene, skipped = _build_scene(params, pp)
        friction = float(params.value("bricks.jaw_on_brick_friction"))

        def _do_run():
            return pp.bench(
                pad_friction=friction, trials=trials, seed0=ctx.seed,
                place_error=place_error_mm * 1e-3, dr_level=0.0, scene=scene,
            )

        result = _run_with_timeout(_do_run, timeout_s)
        # `build_scene` attached a damage monitor and `reset_scene` clears it between trials, so
        # the monitor's high-water marks are the worst any trial in this bench reached.
        if getattr(scene, "monitor", None) is not None:
            result["damage_state"] = scene.monitor.worst_state_seen
            result["peak_wall_stress_MPa"] = scene.monitor.worst_wall_stress_Pa / 1e6
        _report_result(ctx, result, trials, place_error_mm, friction, skipped)
    except _NotAvailable as exc:
        _not_available_report(ctx, exc.reason)
    except _Timeout as exc:
        _not_available_report(
            ctx, f"brick_builder.control.pick_place.bench {exc} (trials={trials}); treating it as "
                 "not stable enough to trust yet rather than waiting indefinitely."
        )
    except Exception as exc:  # the module imported and looked right, but raised (here, or above)
        _not_available_report(
            ctx,
            f"brick_builder.control.pick_place.bench raised while running or while reading its "
            f"result: {type(exc).__name__}: {exc}\n\n```\n{traceback.format_exc()}\n```",
        )


def _report_result(ctx, result, trials, place_error_mm, friction, skipped) -> None:
    """Everything from a completed ``bench()`` call to a written report. Split out from
    :func:`run` purely so that function's ``try`` covers this too -- see its own comment."""
    successes = result["successes"]
    ci = wilson_interval(successes, result["trials"])
    print(f"pick-and-place: {successes}/{result['trials']} succeeded "
          f"({ci.p_hat:.1%}, 95% CI [{ci.low:.1%}, {ci.high:.1%}])")
    counts = Counter(_failure_category(r["failure"]) for r in result["rows"] if not r["success"])
    for reason, n in counts.most_common():
        print(f"  {reason}: {n}")
    if skipped:
        print(f"Note: these SceneConfig/HeldBrick fields do not exist right now and were "
              f"skipped: {sorted(set(skipped))}")

    fig_path = _figure(result, ci, ctx.out_dir)

    damage_md = ""
    if "damage_state" in result:
        damage_md = (
            "| worst damage state | peak wall stress [MPa] |\n|---|---|\n"
            f"| {result['damage_state']} | {result['peak_wall_stress_MPa']:.2f} |\n"
        )
        print(f"damage: {result['damage_state']} | peak wall stress "
              f"{result['peak_wall_stress_MPa']:.1f} MPa")

    failure_table = "| reason | count |\n|---|---|\n"
    for reason, n in counts.most_common():
        failure_table += f"| {reason} | {n} |\n"
    if not counts:
        failure_table += "| (none -- every trial succeeded) | 0 |\n"

    # "partial seat" groups trials that missed full depth by very different amounts (the exact
    # figure is the detail _failure_category strips for the count table above); keep that detail
    # as a range instead of losing it entirely.
    depth_errors = [
        r["depth_error_mm"] for r in result["rows"]
        if not r["success"] and _failure_category(r["failure"]) == "partial seat"
        and r.get("depth_error_mm") is not None
    ]
    if depth_errors:
        failure_table += (
            f"\n\"partial seat\" shortfall ranged {min(depth_errors):.2f} to "
            f"{max(depth_errors):.2f} mm short of full depth, median "
            f"{sorted(depth_errors)[len(depth_errors) // 2]:.2f} mm "
            f"(the acceptance tolerance is 0.30 mm).\n"
        )

    reading = [
        f"{successes} of {result['trials']} trials placed the brick successfully "
        f"({ci.p_hat:.1%}, 95% Wilson interval [{ci.low:.1%}, {ci.high:.1%}]) at "
        f"{place_error_mm:g} mm of placement error and jaw_on_brick_friction = {friction:g}.",
        f"Median press force used: {result['median_held_press_n']:.1f} N; worst-case peak: "
        f"{result['max_peak_press_n']:.1f} N; worst-case joint torque: {result['max_torque_nm']:.2f} N.m.",
    ]
    if skipped:
        reading.append(
            f"Some requested scene fields do not exist on the current `SceneConfig`/`HeldBrick` "
            f"(still being extended elsewhere in the codebase): {sorted(set(skipped))}."
        )

    sections = [
        report_header(
            ctx, "pick-and-place: full sequence, randomized trials",
            f"Runs brick_builder.control.pick_place.bench for {trials} randomized trials (tray "
            f"jitter, {place_error_mm:g} mm of placement error), using this file's arm, snap-fit "
            f"and friction parameters, and reports the success rate with a Wilson interval and "
            f"why the failures failed."
        ),
        "## Parameters used\n\n" + parameter_table_md(
            ctx.params,
            sorted(p for p in ctx.params.values if any(p.startswith(pre) for pre in RELEVANT_PREFIXES)),
        ),
        f"## Result\n\n{successes}/{result['trials']} succeeded, {ci.p_hat:.1%} "
        f"(95% Wilson CI [{ci.low:.1%}, {ci.high:.1%}]).\n\n### Failure reasons\n\n" + failure_table,
        f"![Success rate and failure reasons]({fig_path.name})\n",
        *(["## Damage\n\nWorst state any trial reached, from the scene's own damage monitor.\n\n"
           + damage_md] if damage_md else []),
        "## Reading\n\n" + "\n".join(f"- {r}" for r in reading),
    ]
    write_report(ctx.out_dir, sections)

    write_data_json(ctx.out_dir, {
        "available": True,
        "trials": result["trials"],
        "successes": successes,
        "success_rate": ci.p_hat,
        "wilson_low": ci.low,
        "wilson_high": ci.high,
        "failure_counts": dict(counts),
        "place_error_mm": place_error_mm,
        "pad_friction": friction,
        "median_held_press_n": result["median_held_press_n"],
        "max_peak_press_n": result["max_peak_press_n"],
        "max_torque_nm": result["max_torque_nm"],
        "rows": result["rows"],
        "skipped_fields": sorted(set(skipped)) if skipped else [],
    })
