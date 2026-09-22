"""A contact sheet of what the page shows, composed without a browser.

    python -m brick_builder.gui.contact_sheet --out docs/media/gui/simulator-gui.png

It starts the server, drives it through the same JSON routes the page uses, pulls JPEG frames off
the same MJPEG streams the page's two `<img>` elements read, and lays them out beside the tables
drawn from the same `/api/state`. The tiles are therefore the page's own pictures and the page's
own numbers; the surrounding HTML is not rendered here and this is not a browser screenshot.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import socket
import sys
import time
import urllib.request
from pathlib import Path
from typing import Sequence

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

TILE_W = 760
GAP = 14
BG = (18, 20, 26)
PANEL = (27, 30, 38)
TEXT = (240, 242, 247)
DIM = (167, 175, 192)
LINE = (44, 49, 61)


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _get(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=120) as r:
        return r.read()


def _post(url: str, body: dict) -> dict:
    request = urllib.request.Request(
        url, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=120) as r:
        return json.loads(r.read())


def _frame(url: str):
    from PIL import Image

    with urllib.request.urlopen(url, timeout=120) as r:
        r.readline()
        length = 0
        while True:
            line = r.readline().strip()
            if line.lower().startswith(b"content-length"):
                length = int(line.split(b":")[1])
            if not line:
                break
        return Image.open(io.BytesIO(r.read(length))).convert("RGB")


def _wait(base: str, until: set[str], limit: float = 900.0) -> dict:
    t0 = time.perf_counter()
    state = json.loads(_get(f"{base}/api/state"))
    while time.perf_counter() - t0 < limit:
        state = json.loads(_get(f"{base}/api/state"))
        if state["status"] in until:
            return state
        time.sleep(1.0)
    raise RuntimeError(f"still {state['status']}: {state['message']}")


# ------------------------------------------------------------------------------ drawing ---


def _fonts():
    from PIL import ImageFont

    from ..demo.video_panels import _font_dir

    d = _font_dir()
    return (ImageFont.truetype(str(d / "DejaVuSans-Bold.ttf"), 19),
            ImageFont.truetype(str(d / "DejaVuSans-Bold.ttf"), 14),
            ImageFont.truetype(str(d / "DejaVuSans.ttf"), 13))


def _table_tile(title: str, pairs, width: int):
    from PIL import Image, ImageDraw

    _, bold, plain = _fonts()
    rows = [(str(k), "—" if v is None else str(v)) for k, v in pairs]
    height = 30 + 20 * len(rows) + 10
    img = Image.new("RGB", (width, height), PANEL)
    draw = ImageDraw.Draw(img)
    draw.text((12, 8), title, font=bold, fill=TEXT)
    y = 32
    for key, value in rows:
        draw.text((12, y), key, font=plain, fill=DIM)
        draw.text((width // 2, y), value, font=plain, fill=TEXT)
        draw.line((12, y + 18, width - 12, y + 18), fill=LINE)
        y += 20
    return img


def _picture_tile(title: str, picture, width: int):
    from PIL import Image, ImageDraw

    _, bold, _ = _fonts()
    scaled = picture.resize((width, round(width * picture.height / picture.width)),
                            Image.LANCZOS)
    img = Image.new("RGB", (width, scaled.height + 30), PANEL)
    ImageDraw.Draw(img).text((12, 7), title, font=bold, fill=TEXT)
    img.paste(scaled, (0, 30))
    return img


def _sheet(title: str, subtitle: str, tiles: Sequence, columns: int = 2):
    from PIL import Image, ImageDraw

    big, _, plain = _fonts()
    width = columns * TILE_W + (columns + 1) * GAP
    column_h = [0] * columns
    places = []
    for tile in tiles:
        c = column_h.index(min(column_h))
        places.append((c, column_h[c]))
        column_h[c] += tile.height + GAP
    head = 62
    sheet = Image.new("RGB", (width, head + max(column_h) + GAP), BG)
    draw = ImageDraw.Draw(sheet)
    draw.text((GAP, 12), title, font=big, fill=TEXT)
    draw.text((GAP, 38), subtitle, font=plain, fill=DIM)
    for tile, (c, y) in zip(tiles, places):
        sheet.paste(tile, (GAP + c * (TILE_W + GAP), head + y))
    return sheet


def _step_value(row: dict) -> str:
    """One step's cell: its vision ticks, and its last error where one was measured."""
    err = row.get("err_mm")
    return f"{row['ticks']} ticks" + ("" if err is None else f", {err} mm")


# --------------------------------------------------------------------------------- run ---


def build(out: Path, *, seed: int = 1, quality_name: str = "draft",
          run_seconds: float = 55.0) -> Path:
    from .server import serve
    from .session import Session

    import tempfile

    session = Session(quality_name=quality_name,
                      out_dir=Path(tempfile.mkdtemp(prefix="gui-sheet-")))
    session.start()
    port = _free_port()
    httpd = serve(session, host="127.0.0.1", port=port)
    base = f"http://127.0.0.1:{port}"
    try:
        setup = json.loads(_get(f"{base}/api/state"))["setup"]
        setup["seed"] = seed
        setup["mode"] = "twin (sim)"
        _post(f"{base}/api/setup", setup)
        state = _wait(base, {"ready", "failed"})
        if state["status"] != "ready":
            raise RuntimeError(state["message"])

        start = _frame(f"{base}/stream/view.mjpg")
        options = json.loads(_get(f"{base}/api/options"))
        _post(f"{base}/api/command", {"command": "speed", "speed": "as fast as possible"})
        _post(f"{base}/api/command", {"command": "run"})
        time.sleep(run_seconds / 2.0)
        middle = _frame(f"{base}/stream/view.mjpg")
        twin = _frame(f"{base}/stream/twin.mjpg")
        mid_state = json.loads(_get(f"{base}/api/state"))
        _post(f"{base}/api/disturb", {"kind": "brick", "body": "held", "dx_mm": 1.5})
        time.sleep(run_seconds / 2.0)
        late = _frame(f"{base}/stream/view.mjpg")
        end_state = json.loads(_get(f"{base}/api/state"))
        _post(f"{base}/api/command", {"command": "pause"})

        readouts = end_state["readouts"]
        result = end_state["result"] or {}
        tiles = [
            _picture_tile("Main view, composed frame — step start", start, TILE_W),
            _picture_tile(f"Main view — step {mid_state['readouts'].get('step', '')}",
                          middle, TILE_W),
            _picture_tile(f"Main view — step {readouts.get('step', '')}", late, TILE_W),
            _picture_tile("Twin (sim)", twin, TILE_W),
            _table_tile("Readouts", [
                ("Step", readouts.get("step")),
                ("Scene-state error [mm]", readouts.get("err_mm")),
                ("Tracking tolerance [mm]", readouts.get("tol_mm")),
                ("Segmentation mIoU [%]", readouts.get("miou")),
                ("Prediction error [mm]", readouts.get("predict_mm")),
                ("Press force [N]", readouts.get("press_n")),
                ("Insertion depth [mm]", readouts.get("depth_mm")),
                ("Damage state", readouts.get("damage")),
                ("Correction [mm]", readouts.get("correction_mm")),
            ], TILE_W),
            _table_tile("Steps", [(r["step"], _step_value(r)) for r in end_state["steps"]],
                        TILE_W),
            _table_tile("Discrepancy", [
                ("Joint RMSE [deg]", end_state["twin"].get("joint_rmse_deg")),
                ("Tool pose error [mm]", end_state["twin"].get("tool_mm")),
                ("Scene-state distance [-]", end_state["twin"].get("scene_distance")),
            ], TILE_W),
            _table_tile("Scene", [
                ("Instruction", setup["instruction"]),
                ("Placement", end_state["placement"]),
                ("Job", setup["job"]),
                ("Seed [-]", setup["seed"]),
                ("Scene-model source", setup["source"]),
                ("Controller", setup["controller"]),
                ("Mode", setup["mode"]),
                ("Baseplate [mm]", end_state["displacement"].get("plate_mm")),
                ("Baseplate [deg]", end_state["displacement"].get("plate_deg")),
                ("Seated", result.get("seated")),
                ("On cell", result.get("on_cell")),
                ("Lateral [mm]", result.get("lateral_mm")),
                ("Depth error [mm]", result.get("depth_error_mm")),
                ("Main view [px]", "x".join(str(v) for v in options["size"])),
                ("Main view [fps]", end_state["fps"]["view"]),
            ], TILE_W),
        ]
        sheet = _sheet(
            "Simulator",
            "frames pulled from the page's own MJPEG streams and tables drawn from /api/state; "
            "the page's HTML is not rendered here",
            tiles)
        out.parent.mkdir(parents=True, exist_ok=True)
        sheet.save(out)
        return out
    finally:
        httpd.shutdown()
        session.close()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m brick_builder.gui.contact_sheet")
    parser.add_argument("--out", default="docs/media/gui/simulator-gui.png")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--quality", default="draft", choices=("live", "draft", "hd1080"))
    parser.add_argument("--seconds", type=float, default=55.0)
    args = parser.parse_args(argv)
    path = build(Path(args.out), seed=args.seed, quality_name=args.quality,
                 run_seconds=args.seconds)
    print(f"contact sheet {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
