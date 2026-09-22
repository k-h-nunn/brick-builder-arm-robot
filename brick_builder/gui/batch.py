"""N seeds run headless in the background, and the table they produce.

The sweep is the one the nominal-tracking tests make: record the reference sequence once, then
for every seed build the world that seed displaces, track the sequence in it, and score the brick
with `verify_placement` once the arm has left. Nothing renders and nothing is composed, so the
page keeps its own frame rate while a sweep is running.
"""

from __future__ import annotations

import csv
import io
import threading
import time
from pathlib import Path

import numpy as np

from ..action.place import PlaceConfig, TrackingEpisode, record_ideal
from ..action.worlds import SUPPLY_BODY
from ..control.pick_place import verify_placement
from ..eval.wilson import wilson_interval
from . import world as worldmod

COLUMNS = ("seed", "seated", "on cell", "lateral [mm]", "depth error [mm]", "ticks", "failure")


class Batch:
    """One sweep. Start it once; read `status()` while it runs."""

    def __init__(self, setup: worldmod.Setup, params, *, seeds: int, target: tuple[int, int],
                 out_dir: Path) -> None:
        self.setup = setup
        self.params = params
        self.seeds = int(seeds)
        self.target = target
        self.out_dir = Path(out_dir)
        self._lock = threading.Lock()
        self._rows: list[dict] = []
        self._done = 0
        self._state = "idle"
        self._message = ""
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # -- control -------------------------------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        with self._lock:
            self._rows = []
            self._done = 0
            self._state = "running"
            self._message = "recording the reference sequence"
        self._thread = threading.Thread(target=self._run, name="gui-batch", daemon=True)
        self._thread.start()

    def cancel(self) -> None:
        self._stop.set()

    def status(self) -> dict:
        with self._lock:
            rows = list(self._rows)
            done, state, message = self._done, self._state, self._message
        seated = sum(1 for r in rows if r["seated"])
        summary = None
        if rows:
            interval = wilson_interval(seated, len(rows))
            lateral = np.array([r["lateral [mm]"] for r in rows], dtype=float)
            summary = {
                "n": len(rows),
                "seated": seated,
                "on_cell": sum(1 for r in rows if r["on cell"]),
                "rate": round(interval.p_hat, 3),
                "wilson_low": round(interval.low, 3),
                "wilson_high": round(interval.high, 3),
                "lateral_median_mm": round(float(np.median(lateral)), 3),
                "lateral_max_mm": round(float(np.max(lateral)), 3),
            }
        return {"state": state, "message": message, "done": done, "total": self.seeds,
                "columns": list(COLUMNS), "rows": rows, "summary": summary}

    def csv_text(self) -> str:
        with self._lock:
            rows = list(self._rows)
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=list(COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
        return buf.getvalue()

    def write_csv(self) -> Path:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        path = self.out_dir / f"batch_{time.strftime('%Y%m%d_%H%M%S')}.csv"
        path.write_text(self.csv_text())
        return path

    # -- the sweep -----------------------------------------------------------------------------

    def _run(self) -> None:
        from .session import make_source

        try:
            cfg = PlaceConfig(pick=self.setup.pick, closed_loop=True,
                              inverse=self.setup.controller)
            ideal, nominal = record_ideal(cfg, seed=0, target=self.target,
                                          source=make_source(self.setup.source))
            nominal.close()
            for i in range(self.seeds):
                if self._stop.is_set():
                    break
                with self._lock:
                    self._message = f"seed {i}"
                self._one(cfg, ideal, i, make_source)
                with self._lock:
                    self._done = i + 1
            with self._lock:
                self._state = "cancelled" if self._stop.is_set() else "done"
                self._message = ""
        except Exception as exc:                              # noqa: BLE001
            with self._lock:
                self._state = "failed"
                self._message = f"{type(exc).__name__}: {exc}"

    def _one(self, cfg: PlaceConfig, ideal, seed: int, make_source) -> None:
        setup = worldmod.Setup.from_json(self.setup.to_json())
        setup.seed = seed
        setup.plate_offset = "seed" if setup.plate_offset != "nominal" else "nominal"
        setup.supply_offset = "seed" if setup.supply_offset != "nominal" else "nominal"
        built = worldmod.build(setup, self.params)
        scene = built["scene"]
        brick = SUPPLY_BODY if setup.pick else scene.held_name
        episode = TrackingEpisode(scene, ideal, cfg=cfg, seed=seed, build_decode=False,
                                  source=make_source(setup.source))
        ticks = 0
        try:
            for rec in episode.run():
                if rec.vision:
                    ticks += 1
                if self._stop.is_set():
                    break
            check = verify_placement(scene, brick,
                                     (self.target[0] + 0.5, self.target[1] + 0.5, 0))
        finally:
            episode.close()
        row = {
            "seed": seed,
            "seated": bool(check.seated),
            "on cell": bool(check.at_target_cell),
            "lateral [mm]": round(float(check.lateral_mm), 3),
            "depth error [mm]": round(float(check.depth_error_mm), 3),
            "ticks": ticks,
            "failure": episode.failure or "",
        }
        with self._lock:
            self._rows.append(row)
