"""One simulation behind the page: physics in one thread, rendering in another.

The physics thread owns the world. It builds the scene, records the reference sequence, runs the
tracking loop one vision tick at a time, applies whatever disturbance the page has posted, and
publishes a copy of `MjData` when it is done with a tick. The render thread takes that copy,
renders the free camera, draws the panels and the readout strip over it, and hands the finished
frame to whoever is watching the stream.

They are separate because they run at different rates. One vision tick of this world costs about
a quarter of a second of physics on this machine and a composed frame costs about eighty
milliseconds; rendering inside the physics thread would tie the stream to the slower of the two
and the page would show three frames a second while paused as well as while running.
"""

from __future__ import annotations

import math
import pickle
import queue
import threading
import time
from pathlib import Path

import mujoco
import numpy as np

from ..action.adapter import VISION_EVERY, WRIST_FULL
from ..action.place import PlaceConfig, TrackingEpisode, record_ideal
from ..action.worlds import SUPPLY_BODY
from ..control.pick_place import CONTROL_DT, gripper_press_force, verify_placement
from ..demo.video import panel_size_for
from ..language import compile_command
from ..perception import sources
from ..perception.slot_ops import slot_distance
from . import world as worldmod
from .panels import (
    PANEL_TITLES, Camera, Composer, Labeller, Panels, PlainView, quality, recording_labeller,
)

SPEEDS = ("real time", "as fast as possible")

SOURCE_LABELS = {
    "ideal": "Ground truth",
    "colour": "Colour threshold",
    "learned": "Convolutional (registry)",
    "cnn": "Convolutional",
}
"""Titles for the page's source list. Anything else in the registry is offered under its own
name, so a source another package registers needs no change here."""

TWIN_VIEW = (640, 360)


def source_names() -> list[str]:
    names = list(sources.available())
    if "cnn" not in names:
        names.append("cnn")
    return names


def make_source(name: str):
    """One scene model, built fresh. The recording and the run each get their own, so neither
    carries the other's slot assignment into its first frame."""
    if name == "cnn":
        from ..perception.cnn.source import CnnSceneModel

        return CnnSceneModel(build_decode=False)
    return sources.get(name)(K=8, build_decode=False)


class Slot:
    """The latest frame of one stream, and a counter the readers wait on."""

    def __init__(self) -> None:
        self._cv = threading.Condition()
        self._buf: bytes | None = None
        self._n = 0

    def put(self, buf: bytes) -> None:
        with self._cv:
            self._buf = buf
            self._n += 1
            self._cv.notify_all()

    def get(self, seen: int, timeout: float = 5.0):
        with self._cv:
            if self._n == seen:
                self._cv.wait(timeout)
            return self._buf, self._n


def encode(rgb: np.ndarray, jpeg_quality: int = 80) -> bytes:
    import cv2

    ok, buf = cv2.imencode(".jpg", np.ascontiguousarray(rgb[:, :, ::-1]),
                           [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
    if not ok:
        raise RuntimeError("frame would not encode")
    return buf.tobytes()


class Session:
    """Everything one browser page drives."""

    def __init__(self, *, params_file: str | None = None, quality_name: str = "draft",
                 out_dir: Path = Path("media_out/gui")) -> None:
        self.quality = quality(quality_name)
        self.out_dir = Path(out_dir)
        self.setup = worldmod.Setup()
        if params_file:
            self.setup.params_file = str(params_file)
        self.camera = Camera()
        self.streams = {"view": Slot(), "twin": Slot()}

        self._cmds: queue.Queue = queue.Queue()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._pub = threading.Lock()
        self._pub_data: mujoco.MjData | None = None
        self._pub_twin: mujoco.MjData | None = None
        self._pub_scene = None
        self._pub_twin_scene = None
        self._pub_panels: list = []
        self._pub_phase = ""
        self._pub_values: dict = {}
        self._pub_new = False
        self._mark_step = False
        self._gen = 0

        self._scene = None
        self._twin_scene = None
        self._twin_renderer = None
        self._twin_source = None
        self._episode = None
        self._gen_iter = None
        self._source = None
        self._panels: Panels | None = None
        self._truth: Labeller | None = None
        self._hold: worldmod.ArmHold | None = None
        self._ideal_cache: dict = {}
        self._log: list[dict] = []
        self._brick = None
        self._target = (4, 3)
        self._steps_left = 0
        self._speed = SPEEDS[0]
        self._record: dict = {}
        self._still = False
        self._fps = {"view": 0.0, "twin": 0.0}
        self._tick_rate = 0.0

        self.state = {
            "status": "empty",
            "message": "no scene",
            "mode": "sim",
            "step": "",
            "tick": 0,
            "k": 0,
            "frames": 0,
            "readouts": {},
            "steps": [],
            "result": None,
            "failure": None,
            "problems": [],
            "placement": "",
            "grid": [],
            "displacement": {},
            "twin": {},
            "recording": False,
            "files": [],
        }

        self._sim = threading.Thread(target=self._sim_loop, name="gui-sim", daemon=True)
        self._render = threading.Thread(target=self._render_loop, name="gui-render", daemon=True)

    # -- lifecycle ------------------------------------------------------------------------------

    def start(self) -> None:
        self._sim.start()
        self._render.start()

    def close(self) -> None:
        self._stop.set()
        self._sim.join(timeout=30.0)
        self._render.join(timeout=10.0)

    def post(self, name: str, payload: dict | None = None) -> None:
        self._cmds.put((name, payload or {}))

    def snapshot(self) -> dict:
        with self._lock:
            out = dict(self.state)
        out["setup"] = self.setup.to_json()
        out["camera"] = self.camera.as_json()
        out["speed"] = self._speed
        out["fps"] = {k: round(v, 1) for k, v in self._fps.items()}
        out["tick_rate"] = round(self._tick_rate, 2)
        return out

    def _set(self, **kw) -> None:
        with self._lock:
            self.state.update(kw)

    def _say(self, message: str) -> None:
        self._set(message=message)

    # -- the physics thread ---------------------------------------------------------------------

    def _sim_loop(self) -> None:
        while not self._stop.is_set():
            try:
                name, payload = self._cmds.get(timeout=0.05)
            except queue.Empty:
                name, payload = None, {}
            if name is not None:
                try:
                    self._handle(name, payload)
                except Exception as exc:                      # noqa: BLE001
                    problem = f"{type(exc).__name__}: {exc}"
                    # Only a scene that would not build leaves the session unusable. A control
                    # or a disturbance that was refused is reported and the run carries on.
                    if name in ("setup", "reset"):
                        self._set(status="failed", message=problem)
                    else:
                        self._say(problem)
                continue

            running = self.state["status"] == "running"
            if running or self._steps_left > 0:
                t0 = time.perf_counter()
                try:
                    self._advance()
                except Exception as exc:                      # noqa: BLE001
                    self._set(status="failed", message=f"{type(exc).__name__}: {exc}")
                    self._steps_left = 0
                    continue
                dt = time.perf_counter() - t0
                self._tick_rate = (VISION_EVERY * CONTROL_DT) / max(dt, 1e-6)
                if self._steps_left > 0 and self.state["status"] == "running":
                    self._steps_left -= 1
                    if self._steps_left == 0:
                        self._set(status="paused", message="paused")
                if self._speed == SPEEDS[0]:
                    left = VISION_EVERY * CONTROL_DT - dt
                    if left > 0:
                        time.sleep(left)
            else:
                time.sleep(0.02)
        self._teardown()

    def _handle(self, name: str, payload: dict) -> None:
        if name == "setup":
            self.setup = worldmod.Setup.from_json(payload)
            self._build()
        elif name == "reset":
            self._build()
        elif name == "run":
            if self.state["status"] in ("ready", "paused"):
                self._set(status="running", message="running")
        elif name == "pause":
            if self.state["status"] == "running":
                self._set(status="paused", message="paused")
        elif name == "step":
            if self.state["status"] in ("ready", "paused"):
                self._steps_left = int(payload.get("ticks", 1))
                self._set(status="running", message="stepping")
        elif name == "speed":
            self._speed = payload.get("speed", SPEEDS[0])
        elif name == "disturb":
            self._disturb(payload)
        elif name == "record":
            self._recording(payload)
        elif name == "log":
            self._write_log()

    # -- building -------------------------------------------------------------------------------

    def _teardown(self) -> None:
        for obj in (self._episode, self._truth, self._twin_renderer):
            try:
                if obj is not None:
                    obj.close()
            except Exception:                                 # noqa: BLE001
                pass
        self._episode = None
        self._truth = None
        self._twin_renderer = None
        self._twin_source = None
        self._gen_iter = None

    def _build(self) -> None:
        self._set(status="building", message="building the scene", result=None, failure=None,
                  steps=[], readouts={}, tick=0, k=0)
        self._teardown()
        self._log = []
        setup = self.setup

        compiled = compile_command(setup.instruction)
        problems = list(compiled.problems)
        if compiled.goal is not None and compiled.goal.steps:
            self._target = worldmod.target_cell(compiled.goal)
            placement = compiled.goal.to_text()
            grid = _grid_rows(compiled.goal)
            p = compiled.goal.steps[0].placement
            if (p.h, p.w, p.colour) != (2, 2, "blue"):
                problems.append(
                    "the supply row holds one blue 2x2; the brick in the command is ignored")
        else:
            placement, grid = "", []
            problems.append("no goal compiled; the target cell stays at column 5 row 4")

        scratch = self.out_dir / "params_edited.yaml"
        params, param_problems = worldmod.load_edited(
            setup.params_file, setup.param_edits, scratch)
        problems += param_problems

        prepare = lambda model: None if model is None else _look(model, self.quality)  # noqa: E731
        built = worldmod.build(setup, params, prepare=prepare)
        problems += built["problems"]
        scene = built["scene"]
        self._scene = scene
        self._brick = SUPPLY_BODY if setup.pick else scene.held_name
        self._hold = worldmod.ArmHold(scene)

        cfg = PlaceConfig(pick=setup.pick, closed_loop=True, inverse=setup.controller)
        ideal = self._reference(cfg, setup, prepare)

        self._say("starting the loop")
        self._source = make_source(setup.source)
        self._episode = TrackingEpisode(
            scene, ideal, cfg=cfg, seed=int(setup.seed), twin_prepare=prepare,
            build_decode=False, source=self._source)
        self._gen_iter = self._episode.run()
        self._truth = Labeller(scene)

        self._twin_scene = None
        self._twin_renderer = None
        self._twin_source = None
        twin_info = {}
        if setup.mode == "twin (sim)":
            twin_params, _ = worldmod.load_edited(setup.twin_params_file, {}, scratch)
            twin_built = worldmod.build(setup, twin_params, prepare=prepare)
            self._twin_scene = twin_built["scene"]
            twin_info = {"params": setup.twin_params_file}

        with self._pub:
            self._pub_scene = scene
            self._pub_data = mujoco.MjData(scene.model)
            self._pub_twin_scene = self._twin_scene
            self._pub_twin = (None if self._twin_scene is None
                              else mujoco.MjData(self._twin_scene.model))
            self._gen += 1

        self._panels = Panels(panel_size_for(self.quality, 4), source=self._source)
        self._first_frame()

        self._set(status="ready", message="ready", problems=problems, placement=placement,
                  grid=grid, mode=setup.mode, twin=twin_info,
                  displacement={
                      "plate_mm": round(1e3 * float(np.linalg.norm(built["plate_d_xy"])), 2),
                      "plate_deg": round(math.degrees(built["plate_d_yaw"]), 2),
                      "supply_mm": round(1e3 * float(np.linalg.norm(built["supply_d_xy"])), 2),
                      "supply_deg": round(math.degrees(built["supply_d_yaw"]), 2),
                  })

    def _reference(self, cfg: PlaceConfig, setup, prepare):
        """The reference sequence, recorded in the nominal world -- or read back from disk.

        Recording it costs about a minute of physics and nothing in it depends on the world the
        run happens in, so it is kept: keyed by the job, the target cell, the scene model and the
        seed, which is everything `record_ideal` is handed.
        """
        key = (setup.pick, self._target, setup.source, int(setup.seed))
        if key in self._ideal_cache:
            return self._ideal_cache[key]
        cache = self.out_dir / "reference" / (
            "_".join(str(k) for k in ("pick" if setup.pick else "place",
                                      *self._target, setup.source, setup.seed)) + ".pkl")
        if cache.is_file():
            self._say("reading the reference sequence")
            try:
                with cache.open("rb") as fh:
                    ideal = pickle.load(fh)
                self._ideal_cache.clear()
                self._ideal_cache[key] = ideal
                return ideal
            except Exception:                                 # noqa: BLE001
                cache.unlink(missing_ok=True)
        self._say("recording the reference sequence")
        ideal, nominal = record_ideal(
            cfg, seed=int(setup.seed), target=self._target, prepare=prepare,
            build_decode=False, source=make_source(setup.source),
            label_hook=recording_labeller())
        nominal.close()
        self._ideal_cache.clear()
        self._ideal_cache[key] = ideal
        cache.parent.mkdir(parents=True, exist_ok=True)
        try:
            with cache.open("wb") as fh:
                pickle.dump(ideal, fh, protocol=4)
        except Exception:                                     # noqa: BLE001
            cache.unlink(missing_ok=True)
        return ideal

    def _first_frame(self) -> None:
        """Fill the panels before anything runs, so the page opens on a picture."""
        _, rgb = self._episode.world.observe_rgb()
        label = self._episode.ideal.frames[0].label
        self._panels.update(rgb, label, self._truth(self._scene), window=WRIST_FULL)
        self._publish(phase=self._episode.ideal.frames[0].step, values={})

    # -- one vision tick ------------------------------------------------------------------------

    def _advance(self) -> None:
        if self._gen_iter is None:
            self._steps_left = 0
            return
        # One vision tick is the frame's control writes and then the camera. `run` always ends a
        # frame with the vision record, so the bound is a guard against a changed contract, not a
        # timeout: reaching it leaves the readouts as they were rather than filing a half tick.
        for _ in range(4 * VISION_EVERY):
            try:
                rec = next(self._gen_iter)
            except StopIteration:
                self._finish()
                return
            self._twin_step()
            if rec.vision:
                self._on_vision(rec)
                return
        self._say("the loop issued a frame with no camera tick in it")

    def _twin_step(self) -> None:
        if self._twin_scene is None:
            return
        reading = self._episode.world.joint_position
        n = min(len(reading), self._twin_scene.model.nu)
        self._twin_scene.data.ctrl[:n] = reading[:n]
        substeps = max(int(round(CONTROL_DT / self._twin_scene.model.opt.timestep)), 1)
        self._twin_scene.step(substeps)

    def _on_vision(self, rec) -> None:
        scene = self._scene
        episode = self._episode
        if self._hold is not None:
            self._hold.tick()
        self._panels.update(rec.rgb, rec.target_label, self._truth(scene), window=WRIST_FULL)

        press = float(gripper_press_force(scene, self._brick)) if self._brick else float("nan")
        depths = np.asarray(scene.snap.stud_depths(self._brick), dtype=float) \
            if self._brick else np.zeros(0)
        values = {
            "err_mm": _num(rec.err_mm),
            "predict_mm": _num(rec.predict_mm),
            "miou": _num(self._panels.miou),
            "press_n": round(press, 2) if press == press else None,
            "depth_mm": round(1e3 * float(np.min(depths)), 3) if depths.size else None,
        }
        readouts = dict(values)
        readouts["tol_mm"] = _num(rec.tol_mm)
        readouts["damage"] = scene.monitor.state if scene.monitor is not None else "off"
        readouts["correction_mm"] = round(
            1e3 * float(np.linalg.norm(np.asarray(episode.correction)[:2])), 2)
        readouts["step"] = rec.step
        readouts["refs_seen"] = int(rec.refs_seen)

        steps = self._step_table(rec)
        twin = self._twin_readouts(rec)
        self._log.append({
            "tick": int(rec.tick), "t": float(rec.t), "k": int(rec.k),
            "step_index": int(rec.step_index),
            "err_mm": float(rec.err_mm), "tol_mm": float(rec.tol_mm),
            "predict_mm": float(rec.predict_mm),
            "press_n": press,
            "depth_mm": float(1e3 * np.min(depths)) if depths.size else float("nan"),
            "miou": float(self._panels.miou),
            "cmd_pos": np.asarray(rec.cmd_pos, dtype=float).copy(),
            "correction": np.asarray(rec.correction, dtype=float).copy(),
        })
        self._set(step=rec.step, tick=int(rec.tick), k=int(rec.k), readouts=readouts, steps=steps,
                  twin=twin)
        self._publish(phase=rec.step, values=values)

    def _step_table(self, rec) -> list[dict]:
        """One row per step of the sequence, filled as the run goes.

        Keyed on the step's own index, not its title: the script visits "Press the brick down"
        twice on a regrasp and the two visits are two rows, not one row counted twice.
        """
        with self._lock:
            rows = list(self.state["steps"])
        index = int(rec.step_index)
        if rows and rows[-1].get("index") == index:
            rows[-1]["ticks"] += 1
            rows[-1]["err_mm"] = _num(rec.err_mm)
        else:
            if rows:
                self._mark_step = True
            rows.append({"index": index, "step": rec.step, "ticks": 1,
                         "err_mm": _num(rec.err_mm)})
        return rows

    def _twin_readouts(self, rec) -> dict:
        if self._twin_scene is None:
            return {}
        a, b = self._scene, self._twin_scene
        qa = a.data.qpos[a.arm_qpos_adr]
        qb = b.data.qpos[b.arm_qpos_adr]
        rmse = math.degrees(float(np.sqrt(np.mean((qa - qb) ** 2))))
        pa = a.tcp_pose()[0]
        pb = b.tcp_pose()[0]
        return {"joint_rmse_deg": round(rmse, 3),
                "tool_mm": round(1e3 * float(np.linalg.norm(pa - pb)), 2),
                "scene_distance": self._twin_scene_distance(rec)}

    def _twin_scene_distance(self, rec) -> float | None:
        """Scene-state distance between the two worlds, from the same wrist window in each."""
        if rec.state is None:
            return None
        try:
            from ..perception.views import render_view
            from ..render.renderer import SceneRenderer

            if self._twin_renderer is None:
                self._twin_renderer = SceneRenderer(self._twin_scene.model, width=512, height=512)
                self._twin_renderer.instance_lut(self._twin_scene.instance_ids)
                self._twin_source = make_source(self.setup.source)
            frame, _ = render_view(self._twin_scene, WRIST_FULL, self._twin_renderer)
            state = self._twin_source.observe(frame.rgb, window=WRIST_FULL,
                                              instance=frame.instance, depth=frame.depth,
                                              t=self._episode.world.t)
            return round(float(slot_distance(rec.state, state)), 4)
        except Exception:                                     # noqa: BLE001
            return None

    def _finish(self) -> None:
        scene = self._scene
        episode = self._episode
        self._gen_iter = None
        self._steps_left = 0
        result = None
        if self._brick:
            check = verify_placement(scene, self._brick,
                                     (self._target[0] + 0.5, self._target[1] + 0.5, 0))
            result = {
                "seated": bool(check.seated),
                "on_cell": bool(check.at_target_cell),
                "lateral_mm": round(float(check.lateral_mm), 3),
                "depth_error_mm": round(float(check.depth_error_mm), 3),
                "damage": scene.monitor.state if scene.monitor is not None else "off",
                "success": bool(check.success),
            }
        rows = [{"index": i, "step": r.name, "ticks": int(r.vision_ticks),
                 "err_mm": _num(r.final_err_mm)} for i, r in enumerate(episode.reports)]
        self._set(status="done", message="finished", result=result, failure=episode.failure,
                  steps=rows or self.state["steps"])
        self._write_log()

    # -- disturbances ---------------------------------------------------------------------------

    def _disturb(self, payload: dict) -> None:
        if self._scene is None:
            return
        kind = payload.get("kind", "")
        if kind == "brick":
            worldmod.nudge(self._scene, str(payload.get("body", SUPPLY_BODY)),
                           float(payload.get("dx_mm", 0.0)), float(payload.get("dy_mm", 0.0)),
                           float(payload.get("dyaw_deg", 0.0)))
            self._say(f"nudged {payload.get('body')}")
        elif kind == "remove":
            worldmod.remove(self._scene, str(payload.get("body", "")))
            self._say(f"removed {payload.get('body')}")
        elif kind == "plate":
            worldmod.displace_plate(self._scene, float(payload.get("dx_mm", 0.0)),
                                    float(payload.get("dy_mm", 0.0)),
                                    float(payload.get("dyaw_deg", 0.0)))
            self._say("displaced the baseplate")
        elif kind == "hold":
            self._hold.start(int(payload.get("ticks", 10)))
            self._say(f"torque off for {int(payload.get('ticks', 10))} ticks")

    # -- recording ------------------------------------------------------------------------------

    def _recording(self, payload: dict) -> None:
        action = payload.get("action", "")
        if action == "still":
            self._still = True
        elif action == "start" and not self._record:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            path = self.out_dir / f"run_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
            self._record = {"path": path, "writer": None}
            self._set(recording=True)
            self._say(f"recording {path}")
        elif action == "stop" and self._record:
            self._record["stop"] = True

    def _write_log(self) -> None:
        if not self._log:
            return
        self.out_dir.mkdir(parents=True, exist_ok=True)
        path = self.out_dir / f"run_{time.strftime('%Y%m%d_%H%M%S')}.npz"
        keys = [k for k in self._log[0] if k not in ("cmd_pos", "correction")]
        arrays = {k: np.array([r[k] for r in self._log]) for k in keys}
        arrays["cmd_pos"] = np.stack([r["cmd_pos"] for r in self._log])
        arrays["correction"] = np.stack([r["correction"] for r in self._log])
        np.savez_compressed(path, **arrays)
        self._add_file(path)

    def _add_file(self, path: Path) -> None:
        with self._lock:
            files = list(self.state["files"])
            files.append(str(path))
            self.state["files"] = files[-12:]

    # -- publishing to the render thread ---------------------------------------------------------

    def _publish(self, *, phase: str, values: dict) -> None:
        with self._pub:
            if self._pub_data is not None:
                mujoco.mj_copyData(self._pub_data, self._scene.model, self._scene.data)
            if self._pub_twin is not None and self._twin_scene is not None:
                mujoco.mj_copyData(self._pub_twin, self._twin_scene.model, self._twin_scene.data)
            self._pub_panels = [(t, img) for t, img in self._panels.entries()]
            self._pub_phase = phase
            self._pub_values = dict(values)
            self._pub_new = True

    # -- the render thread ------------------------------------------------------------------------

    def _render_loop(self) -> None:
        composer: Composer | None = None
        twin_view: PlainView | None = None
        data = twin_data = None
        gen = -1
        frames = 0
        t_fps = time.perf_counter()
        n_fps = 0
        while not self._stop.is_set():
            with self._pub:
                my_gen = self._gen
                scene, twin_scene = self._pub_scene, self._pub_twin_scene
                if my_gen != gen:
                    data = None if scene is None else mujoco.MjData(scene.model)
                    twin_data = (None if twin_scene is None
                                 else mujoco.MjData(twin_scene.model))
                if data is not None and self._pub_data is not None:
                    mujoco.mj_copyData(data, scene.model, self._pub_data)
                if twin_data is not None and self._pub_twin is not None:
                    mujoco.mj_copyData(twin_data, twin_scene.model, self._pub_twin)
                panels = list(self._pub_panels)
                phase = self._pub_phase
                values = dict(self._pub_values)
                new, self._pub_new = self._pub_new, False
                mark, self._mark_step = self._mark_step, False

            if data is None:
                time.sleep(0.05)
                continue
            if gen != my_gen:
                if composer is not None:
                    composer.close()
                if twin_view is not None:
                    twin_view.close()
                    twin_view = None
                composer = Composer(scene, self.quality)
                if twin_scene is not None:
                    twin_view = PlainView(twin_scene, *TWIN_VIEW)
                gen = my_gen

            if new:
                composer.push(values)
                if mark:
                    composer.mark_step()

            if len(panels) != 4:
                panels = [(t, np.zeros((composer.panel_size[1], composer.panel_size[0], 3),
                                       np.uint8)) for t in PANEL_TITLES]
            panels = [(t, _fit(img, composer.panel_size)) for t, img in panels]
            composed = composer.compose(data, self.camera, panels=panels, phase=phase,
                                        values=values)
            self.streams["view"].put(encode(composed))
            self._roll_recording(composed)
            if twin_view is not None:
                self.streams["twin"].put(encode(twin_view.render(twin_data, self.camera)))

            frames += 1
            n_fps += 1
            now = time.perf_counter()
            if now - t_fps >= 2.0:
                self._fps["view"] = n_fps / (now - t_fps)
                self._fps["twin"] = self._fps["view"] if twin_view is not None else 0.0
                t_fps, n_fps = now, 0
                self._set(frames=frames)

        if composer is not None:
            composer.close()
        if twin_view is not None:
            twin_view.close()

    def _roll_recording(self, composed: np.ndarray) -> None:
        if self._still:
            self._still = False
            self.out_dir.mkdir(parents=True, exist_ok=True)
            path = self.out_dir / f"still_{time.strftime('%Y%m%d_%H%M%S')}.png"
            import imageio.v2 as imageio

            imageio.imwrite(path, composed)
            self._add_file(path)
            self._say(f"still {path}")
        if not self._record:
            return
        if self._record.get("stop"):
            writer = self._record.get("writer")
            if writer is not None:
                writer.close()
                self._add_file(self._record["path"])
            self._record = {}
            self._set(recording=False)
            self._say("recording stopped")
            return
        if self._record["writer"] is None:
            from ..demo.video import VideoWriter

            self._record["writer"] = VideoWriter(
                self._record["path"], width=composed.shape[1], height=composed.shape[0],
                fps=15, crf=self.quality.crf, preset=self.quality.preset)
        self._record["writer"].append(composed)


def _look(model, q) -> None:
    from ..demo.video import studio_look

    studio_look(model, q)


def _fit(img: np.ndarray, size) -> np.ndarray:
    import cv2

    w, h = int(size[0]), int(size[1])
    img = np.asarray(img)
    if img.shape[:2] == (h, w):
        return img
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST)


def _num(v) -> float | None:
    v = float(v)
    return None if v != v else round(v, 3)


def _grid_rows(goal) -> list[list[str]]:
    """The expected grid, bottom layer, as eight rows of eight cells."""
    grid = goal.final_grid
    nx, ny = goal.grid_shape
    rows = []
    for y in range(ny - 1, -1, -1):
        row = []
        for x in range(nx):
            cell = grid.get((x, y, 0))
            row.append("" if cell is None else cell.colour)
        rows.append(row)
    return rows
