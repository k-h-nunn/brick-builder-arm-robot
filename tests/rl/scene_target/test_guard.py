"""Import guard: the environment neither imports nor loads the analytic controller.

Static: no module of `brick_builder.rl.scene_target` imports anything under
`brick_builder.action` (except `targets.record_placement`, the offline placement recorder, which
the environment never calls), and none names the servo, the tracking loop or the twin Jacobian.

Runtime: a fresh interpreter makes the environment, resets it and steps it on both sources, and
afterwards not one module under `brick_builder.action` (nor `rl.vision_control`) is loaded.
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

PKG = Path(__file__).resolve().parents[3] / "brick_builder" / "rl" / "scene_target"
FORBIDDEN_NAMES = {"VisualServo", "TrackingEpisode", "build_step", "jacobian_check", "baselines",
                   "reference_features"}
OFFLINE = {("targets.py", "record_placement")}


def _imports(tree):
    """`(module, enclosing function)` for every import in a module."""
    out = []

    def visit(node, func):
        for child in ast.iter_child_nodes(node):
            f = child.name if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) else func
            if isinstance(child, ast.Import):
                out.extend((a.name, f) for a in child.names)
            elif isinstance(child, ast.ImportFrom):
                mod = "." * child.level + (child.module or "")
                out.append((mod, f))
                out.extend((f"{mod}:{a.name}", f) for a in child.names)
            visit(child, f)

    visit(tree, None)
    return out


def _is_action(mod: str) -> bool:
    plain = mod.lstrip(".").split(":")[0]
    return (plain == "action" or plain.startswith("action.")
            or plain.startswith("brick_builder.action") or "vision_control" in plain)


def test_no_module_imports_the_action_package_or_names_the_analytic_controller():
    files = sorted(PKG.glob("*.py"))
    assert {f.name for f in files} >= {"codec.py", "targets.py", "env.py", "config.py",
                                       "__init__.py", "executor.py", "rules.py", "world.py"}
    offenders = []
    for path in files:
        source = path.read_text()
        tree = ast.parse(source)
        for mod, func in _imports(tree):
            if _is_action(mod) and (path.name, func) not in OFFLINE:
                offenders.append(f"{path.name}: imports {mod} in {func or 'module scope'}")
        for node in ast.walk(tree):
            name = (node.id if isinstance(node, ast.Name)
                    else node.attr if isinstance(node, ast.Attribute) else None)
            if name in FORBIDDEN_NAMES:
                offenders.append(f"{path.name}: names {name}")
    assert not offenders, offenders


@pytest.mark.slow
def test_a_running_environment_has_not_loaded_the_analytic_controller(tmp_path):
    code = textwrap.dedent(f"""
        import sys
        import numpy as np
        import gymnasium as gym
        import brick_builder.rl.scene_target as st
        for source in ("ideal", "learned"):
            kw = {{"threads": 1}} if source == "learned" else {{}}
            env = gym.make(st.ENV_ID, source=source, source_kwargs=kw, cache_dir={str(tmp_path)!r})
            env.reset(seed=0, options={{"world_seed": 0}})
            for _ in range(3):
                env.step(np.zeros(env.action_space.shape, dtype=np.float32))
            env.close()
        loaded = sorted(m for m in sys.modules
                        if m.startswith("brick_builder.action") or "vision_control" in m)
        print("LOADED:" + ",".join(loaded))
    """)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(PKG.parents[2]) + os.pathsep + env.get("PYTHONPATH", "")
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True,
                         timeout=600)
    assert out.returncode == 0, out.stderr[-3000:]
    line = [ln for ln in out.stdout.splitlines() if ln.startswith("LOADED:")][-1]
    assert line == "LOADED:", line
