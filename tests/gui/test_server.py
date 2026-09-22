"""The page over HTTP: the routes a browser uses, and one run through the whole of it.

Everything here goes through a real socket on a free port, so what is tested is what the browser
gets: the page itself, the JSON routes, and one JPEG pulled off the `multipart/x-mixed-replace`
stream. The end-to-end test is marked slow because it records a reference sequence and runs the
tracking loop; the rest need no scene and take about a second.
"""

from __future__ import annotations

import json
import socket
import time
import urllib.error
import urllib.request

import pytest


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _get(url: str, timeout: float = 30.0) -> bytes:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read()


def _get_json(url: str, timeout: float = 30.0) -> dict:
    return json.loads(_get(url, timeout))


def _post(url: str, body: dict, timeout: float = 30.0) -> dict:
    request = urllib.request.Request(
        url, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as r:
        return json.loads(r.read())


def _frame(url: str, timeout: float = 60.0) -> bytes:
    """One JPEG off an MJPEG stream, headers and all."""
    with urllib.request.urlopen(url, timeout=timeout) as r:
        assert r.headers["Content-Type"].startswith("multipart/x-mixed-replace")
        boundary = r.readline().strip()
        assert boundary.startswith(b"--")
        length = 0
        while True:
            line = r.readline().strip()
            if line.lower().startswith(b"content-length"):
                length = int(line.split(b":")[1])
            if not line:
                break
        return r.read(length)


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    """A session and its server, with no scene in it: nothing builds until a setup is posted."""
    from brick_builder.gui.server import serve
    from brick_builder.gui.session import Session

    out = tmp_path_factory.mktemp("gui")
    session = Session(quality_name="live", out_dir=out)
    session.start()
    port = _free_port()
    httpd = serve(session, host="127.0.0.1", port=port)
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        httpd.shutdown()
        session.close()


# ------------------------------------------------------------------------------- the page ---


def test_the_page_and_its_two_files_are_served(server) -> None:
    page = _get(f"{server}/").decode()
    assert "<title>Simulator</title>" in page
    assert '/static/app.js' in page and '/static/style.css' in page
    assert "/stream/view.mjpg" in page
    assert b"function" in _get(f"{server}/static/app.js")
    assert b"--bg" in _get(f"{server}/static/style.css")


def test_a_path_outside_the_static_directory_is_refused(server) -> None:
    with pytest.raises(urllib.error.HTTPError) as caught:
        _get(f"{server}/static/../session.py")
    assert caught.value.code == 404


def test_the_options_name_every_choice_the_page_offers(server) -> None:
    options = _get_json(f"{server}/api/options")
    assert "default.yaml" in options["param_files"]
    names = [s["name"] for s in options["sources"]]
    assert {"ideal", "colour", "cnn"} <= set(names)
    assert options["controllers"] == ["analytic", "learned"]
    modes = {m["name"]: m["connected"] for m in options["modes"]}
    assert modes == {"sim": True, "twin (sim)": True,
                     "real arm": False, "microcontroller": False}


# -------------------------------------------------------------------------- scene setup ---


def test_an_instruction_compiles_to_a_placement_and_a_target_cell(server) -> None:
    out = _post(f"{server}/api/compile", {"text": "put the blue 2x2 at column 5 row 4"})
    assert out["understood"] is True
    assert out["placement"] == "2x2 (4,3,0) blue"
    assert out["target"] == [4, 3]
    assert out["grid"][3][4] == "blue"


def test_an_instruction_that_does_not_compile_reports_its_problems(server) -> None:
    out = _post(f"{server}/api/compile", {"text": "put the blue 2x2 at column 99 row 4"})
    assert out["understood"] is False
    assert out["problems"]
    assert out["placement"] == ""


def test_the_parameter_table_carries_a_value_a_unit_and_a_range(server) -> None:
    table = _get_json(f"{server}/api/params?file=default.yaml")
    rows = {r["path"]: r for r in table["rows"]}
    row = rows["bricks.brick_on_brick_friction"]
    assert row["unit"] == "-"
    assert row["lo"] is not None and row["hi"] is not None


def test_a_typed_parameter_out_of_range_is_refused_by_the_schema(server) -> None:
    bad = _post(f"{server}/api/params",
                {"file": "default.yaml", "edits": {"bricks.brick_on_brick_friction": 99.0}})
    assert bad["problems"] and "brick_on_brick_friction" in bad["problems"][0]
    good = _post(f"{server}/api/params",
                 {"file": "default.yaml", "edits": {"bricks.brick_on_brick_friction": 0.4}})
    assert good["problems"] == []


def test_the_camera_orbits_and_comes_home(server) -> None:
    moved = _post(f"{server}/api/camera",
                  {"d_azimuth": 20.0, "d_elevation": -5.0, "zoom": 0.8})
    assert moved["azimuth"] != 268.0 and moved["distance"] < 1.0
    home = _post(f"{server}/api/camera", {"reset": True})
    assert home == {"distance": 1.0, "azimuth": 268.0, "elevation": -24.0}


def test_an_unknown_route_is_a_404(server) -> None:
    with pytest.raises(urllib.error.HTTPError) as caught:
        _get(f"{server}/api/nothing")
    assert caught.value.code == 404


def test_the_sweep_table_is_empty_until_one_is_started(server) -> None:
    batch = _get_json(f"{server}/api/batch")
    assert batch["done"] == 0 and batch["rows"] == []
    assert _get(f"{server}/api/batch.csv") == b""


# ---------------------------------------------------------------------------- one run ---


@pytest.mark.slow
def test_a_scene_sets_up_steps_five_vision_ticks_and_streams_a_frame(server) -> None:
    """The whole path a teammate takes: set a scene up, step it, read the numbers, watch it."""
    setup = _get_json(f"{server}/api/state")["setup"]
    setup["seed"] = 1
    setup["source"] = "ideal"
    _post(f"{server}/api/setup", setup)

    state = _wait(server, {"ready", "failed"}, limit=900.0)
    assert state["status"] == "ready", state["message"]
    assert state["placement"] == "2x2 (4,3,0) blue"
    assert state["displacement"]["plate_mm"] > 0.0

    _post(f"{server}/api/command", {"command": "step", "ticks": 5})
    state = _wait(server, {"paused", "failed", "done"}, limit=300.0)
    assert state["status"] == "paused", state["message"]
    # Not the progress index, which holds while the correction catches up, and not a fixed
    # number of control ticks either: a held index issues one write per vision tick instead of
    # five. What five steps buy is five rows of measurement, which is what the table counts.
    assert state["tick"] >= 5

    readouts = state["readouts"]
    assert readouts["step"] in (
        "Approach", "Grasp", "Lift", "Transport", "Align",
        "Press the brick down", "Release", "Retract", "Verify")
    assert readouts["err_mm"] is not None
    assert readouts["tol_mm"] == pytest.approx(0.4)
    assert readouts["damage"] == "intact"
    assert sum(row["ticks"] for row in state["steps"]) >= 5

    jpeg = _frame(f"{server}/stream/view.mjpg")
    assert jpeg[:3] == b"\xff\xd8\xff"
    assert len(jpeg) > 2000

    _post(f"{server}/api/record", {"action": "still"})
    time.sleep(2.0)
    files = _get_json(f"{server}/api/state")["files"]
    assert any(f.endswith(".png") for f in files), files


def _wait(server: str, until: set[str], *, limit: float) -> dict:
    t0 = time.perf_counter()
    state = _get_json(f"{server}/api/state")
    while time.perf_counter() - t0 < limit:
        state = _get_json(f"{server}/api/state")
        if state["status"] in until:
            return state
        time.sleep(0.5)
    raise AssertionError(f"still {state['status']}: {state['message']}")
