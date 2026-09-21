"""`python -m brick_builder.demo force-map`: the static press-force heatmap over the workspace.

Thin wrapper around `brick_builder.robot.force_map.make_figures`, the package's own
figure-producing function; this demo exists so the heatmap is reachable by name from
`python -m brick_builder.demo` alongside every other one.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from brick_builder.robot.force_map import make_figures


def run(record: str | None = None, seed: int = 0, extra=()) -> Path:
    del seed, extra
    out_dir = Path("docs/media/demos")
    result = make_figures(out_dir=out_dir)
    path = Path(result["force_map"])

    if record is not None and Path(record).resolve() != path.resolve():
        dest = Path(record)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(path, dest)
        path = dest

    print(f"wrote {path}")
    return path
