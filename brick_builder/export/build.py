"""Compile the exported C actor into a shared library with the system `gcc`.

Generated headers (`policy_theta.h`, and `tanh_table.h` for the table-tanh variant) are
written into the same build directory the library lands in, by default
`brick_builder/export/_build/` -- git-ignored, since both are derived from a checkpoint or are a
fixed, regeneratable table, not source. `c/policy.c` and `c/policy.h` are the hand-written,
checkpoint-independent sources and are never written to.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from brick_builder.export.actor_to_c import write_header, write_tanh_table

C_DIR = Path(__file__).resolve().parent / "c"
DEFAULT_BUILD_DIR = Path(__file__).resolve().parent / "_build"


def build_library(
    checkpoint: Path,
    build_dir: Path = DEFAULT_BUILD_DIR,
    tanh_table: bool = False,
    cc: str = "gcc",
) -> tuple[Path, Path]:
    """Generate `policy_theta.h` (and `tanh_table.h` if `tanh_table`) from `checkpoint`,
    then compile `c/policy.c` against them with:

        gcc -O2 -std=c99 -ffp-contract=off -shared -fPIC ...

    `-ffp-contract=off` disables fused multiply-add contraction, so the C build's rounding
    matches the naive multiply-then-add torch's CPU kernels also do, which is part of why
    parity holds to 1e-5 rather than a much looser bound. Only the float-`tanhf` build links
    `-lm`; the table build needs no libm symbol at all.

    Returns `(shared_library_path, policy_theta_header_path)`.
    """
    build_dir = Path(build_dir)
    build_dir.mkdir(parents=True, exist_ok=True)

    header = write_header(checkpoint, build_dir)
    if tanh_table:
        write_tanh_table(build_dir)

    lib_name = "policy_table.so" if tanh_table else "policy_float.so"
    lib_path = build_dir / lib_name

    cmd = [
        cc, "-O2", "-std=c99", "-ffp-contract=off", "-shared", "-fPIC",
        "-Wall", "-Wextra",
        "-I", str(C_DIR), "-I", str(build_dir),
    ]
    if tanh_table:
        cmd.append("-DPOLICY_TANH_TABLE")
    cmd += [str(C_DIR / "policy.c"), "-o", str(lib_path)]
    if not tanh_table:
        cmd.append("-lm")

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"gcc failed (exit {result.returncode}) on: {' '.join(cmd)}\n"
            f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
        )
    return lib_path, header
