"""The HTTP routes.

One `ThreadingHTTPServer` from the standard library. Three kinds of reply: the page and its two
static files, JSON for everything the page reads and writes, and `multipart/x-mixed-replace` for
the two live views. No dependency beyond what the simulator already needs, and no script is
fetched from a network, so the page loads on a machine with no route off the site.
"""

from __future__ import annotations

import json
import mimetypes
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .. import params as params_mod
from ..language import compile_command
from . import world as worldmod
from .batch import Batch
from .session import SOURCE_LABELS, SPEEDS, Session, source_names

STATIC = Path(__file__).resolve().parent / "static"
BOUNDARY = "frameboundary"

MAX_BODY = 4 * 1024 * 1024
STREAM_SECONDS = 24 * 3600.0


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "brick_builder.gui"
    sys_version = ""

    # -- plumbing ------------------------------------------------------------------------------

    def log_message(self, fmt: str, *args) -> None:
        if getattr(self.server, "verbose", False):
            super().log_message(fmt, *args)

    @property
    def session(self) -> Session:
        return self.server.session

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code: int = 200) -> None:
        self._send(code, json.dumps(obj, default=_plain).encode(), "application/json")

    def _body(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0 or length > MAX_BODY:
            return {}
        raw = self.rfile.read(length)
        try:
            return json.loads(raw.decode())
        except ValueError:
            return {"_raw": raw.decode("utf-8", "replace")}

    # -- GET -----------------------------------------------------------------------------------

    def do_GET(self) -> None:                                 # noqa: N802
        url = urlparse(self.path)
        route = url.path
        query = parse_qs(url.query)
        try:
            if route in ("/", "/index.html"):
                self._static("index.html")
            elif route.startswith("/static/"):
                self._static(route[len("/static/"):])
            elif route == "/api/state":
                self._json(self.session.snapshot())
            elif route == "/api/options":
                self._json(self._options())
            elif route == "/api/params":
                name = (query.get("file") or [self.session.setup.params_file])[0]
                self._json(self._params(name))
            elif route == "/api/batch":
                batch = getattr(self.server, "batch", None)
                self._json({"state": "idle", "done": 0, "total": 0, "rows": [],
                            "columns": [], "summary": None}
                           if batch is None else batch.status())
            elif route == "/api/batch.csv":
                batch = getattr(self.server, "batch", None)
                text = "" if batch is None else batch.csv_text()
                self._send(200, text.encode(), "text/csv")
            elif route.startswith("/stream/"):
                self._stream(route[len("/stream/"):].removesuffix(".mjpg"))
            else:
                self._send(404, b"not found", "text/plain")
        except BrokenPipeError:
            pass
        except ConnectionResetError:
            pass

    def _static(self, name: str) -> None:
        path = (STATIC / name).resolve()
        if not str(path).startswith(str(STATIC)) or not path.is_file():
            self._send(404, b"not found", "text/plain")
            return
        kind = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self._send(200, path.read_bytes(), kind)

    def _stream(self, name: str) -> None:
        slot = self.session.streams.get(name)
        if slot is None:
            self._send(404, b"no such view", "text/plain")
            return
        # The body has no length and never ends on its own, so this connection is not reused.
        self.close_connection = True
        self.send_response(200)
        self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={BOUNDARY}")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()
        seen = -1
        # The thread ends when the reader goes away, which raises on the next write. The deadline
        # is only there so a reader that neither reads nor closes cannot hold a thread for ever.
        deadline = time.time() + STREAM_SECONDS
        while time.time() < deadline:
            buf, seen = slot.get(seen, timeout=5.0)
            if buf is None:
                continue
            self.wfile.write(
                f"--{BOUNDARY}\r\nContent-Type: image/jpeg\r\n"
                f"Content-Length: {len(buf)}\r\n\r\n".encode())
            self.wfile.write(buf)
            self.wfile.write(b"\r\n")

    # -- POST ----------------------------------------------------------------------------------

    def do_POST(self) -> None:                                # noqa: N802
        route = urlparse(self.path).path
        body = self._body()
        try:
            if route == "/api/setup":
                self.session.post("setup", body)
                self._json({"ok": True})
            elif route == "/api/command":
                name = str(body.get("command", ""))
                self.session.post(name, body)
                self._json({"ok": True})
            elif route == "/api/camera":
                cam = self.session.camera
                if body.get("reset"):
                    cam.reset()
                cam.orbit(float(body.get("d_azimuth", 0.0)),
                          float(body.get("d_elevation", 0.0)))
                if body.get("zoom"):
                    cam.zoom(float(body["zoom"]))
                self._json(cam.as_json())
            elif route == "/api/disturb":
                self.session.post("disturb", body)
                self._json({"ok": True})
            elif route == "/api/record":
                self.session.post("record", body)
                self._json({"ok": True})
            elif route == "/api/compile":
                self._json(_compile(str(body.get("text", ""))))
            elif route == "/api/params":
                self._json(self._check_params(body))
            elif route == "/api/upload_params":
                self._json(self._upload(body))
            elif route == "/api/batch":
                self._json(self._batch(body))
            else:
                self._send(404, b"not found", "text/plain")
        except BrokenPipeError:
            pass

    # -- the JSON the page reads ----------------------------------------------------------------

    def _options(self) -> dict:
        scene = self.session._scene                           # noqa: SLF001
        files = worldmod.param_files()
        # A file named on the command line, or uploaded, is not on the shelf; it still has to be
        # in the list or the page's own selection would not be one of its options.
        if self.session.setup.params_file not in files:
            files.append(self.session.setup.params_file)
        return {
            "param_files": files,
            "sources": [{"name": n, "label": SOURCE_LABELS.get(n, n)} for n in source_names()],
            "controllers": list(worldmod.CONTROLLERS),
            "jobs": list(worldmod.JOBS),
            "offsets": list(worldmod.OFFSETS),
            "modes": [{"name": m, "connected": m in worldmod.CONNECTED_MODES}
                      for m in worldmod.MODES],
            "speeds": list(SPEEDS),
            "bricks": [] if scene is None else worldmod.brick_names(scene),
            "quality": self.session.quality.name,
            "size": [self.session.quality.width, self.session.quality.height],
            "out_dir": str(self.session.out_dir),
        }

    def _params(self, name: str) -> dict:
        try:
            pf = params_mod.load_params(worldmod.params_path(name))
        except Exception as exc:                              # noqa: BLE001
            return {"file": name, "rows": [], "problems": [str(exc)]}
        rows = []
        for path in sorted(pf.values):
            value = pf.values[path]
            spec = params_mod.SCHEMA.get(path)
            rows.append({
                "path": path,
                "section": path.split(".")[0],
                "value": value.value,
                "unit": value.unit,
                "status": value.status,
                "lo": None if spec is None else spec.lo,
                "hi": None if spec is None else spec.hi,
                "choices": None if spec is None else (
                    list(spec.choices) if spec.choices else None),
            })
        return {"file": name, "rows": rows, "problems": []}

    def _check_params(self, body: dict) -> dict:
        name = str(body.get("file", self.session.setup.params_file))
        edits = dict(body.get("edits") or {})
        scratch = self.session.out_dir / "params_check.yaml"
        _, problems = worldmod.load_edited(name, edits, scratch)
        return {"problems": problems}

    def _upload(self, body: dict) -> dict:
        text = body.get("text") or body.get("_raw") or ""
        name = str(body.get("name") or "uploaded.yaml")
        if not name.endswith(".yaml"):
            name += ".yaml"
        out = self.session.out_dir / Path(name).name
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text)
        try:
            params_mod.load_params(out)
        except params_mod.ParamsError as exc:
            return {"file": str(out), "problems": list(exc.problems)}
        except Exception as exc:                              # noqa: BLE001
            return {"file": str(out), "problems": [str(exc)]}
        return {"file": str(out), "problems": []}

    def _batch(self, body: dict) -> dict:
        action = str(body.get("action", "start"))
        batch = getattr(self.server, "batch", None)
        if action == "cancel":
            if batch is not None:
                batch.cancel()
            return {"ok": True}
        if action == "csv":
            if batch is None:
                return {"ok": False}
            return {"ok": True, "path": str(batch.write_csv())}
        session = self.session
        setup = worldmod.Setup.from_json(body.get("setup") or session.setup.to_json())
        scratch = session.out_dir / "params_batch.yaml"
        params, problems = worldmod.load_edited(setup.params_file, setup.param_edits, scratch)
        compiled = compile_command(setup.instruction)
        target = (worldmod.target_cell(compiled.goal)
                  if compiled.goal is not None and compiled.goal.steps else (4, 3))
        batch = Batch(setup, params, seeds=int(body.get("seeds", 10)), target=target,
                      out_dir=session.out_dir)
        self.server.batch = batch
        batch.start()
        return {"ok": True, "problems": problems}


def _compile(text: str) -> dict:
    from .session import _grid_rows

    result = compile_command(text)
    return {
        "understood": bool(result.understood),
        "problems": list(result.problems),
        "paraphrase": result.paraphrase,
        "placement": "" if result.goal is None else result.goal.to_text(),
        "grid": [] if result.goal is None else _grid_rows(result.goal),
        "target": list(worldmod.target_cell(result.goal))
        if result.goal is not None and result.goal.steps else None,
    }


def _plain(obj):
    import numpy as np

    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if v != v else v
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def serve(session: Session, *, host: str = "0.0.0.0", port: int = 8765,
          verbose: bool = False) -> ThreadingHTTPServer:
    """Start the server on `port` and hand it back, already serving in its own thread."""
    httpd = ThreadingHTTPServer((host, port), Handler)
    httpd.daemon_threads = True
    httpd.session = session
    httpd.batch = None
    httpd.verbose = verbose
    thread = threading.Thread(target=httpd.serve_forever, name="gui-http", daemon=True)
    thread.start()
    return httpd
