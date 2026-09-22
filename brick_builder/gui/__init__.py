"""Browser front end for the closed-loop simulator.

    python -m brick_builder.gui [--params my_params.yaml] [--port 8765]

serves one page from the process that runs the simulation. The live views arrive as MJPEG
streams (`multipart/x-mixed-replace`), the readouts as JSON, and every control posts JSON back.
Nothing is installed in the browser and no script is fetched from a network, so the page works
on a machine with no display and from any other machine on the same network.

Layout of the package:

- `world.py`  -- the scene setup: one `Setup` record, and the scene it builds through the
  helpers in `brick_builder.action.worlds`. Disturbances live here too, on the scene-setup side.
- `panels.py` -- the four square panels and the composed frame, drawn with the video toolkit.
- `session.py` -- one simulation, two threads: physics and rendering.
- `batch.py`  -- N seeds run headless in the background.
- `server.py` -- the HTTP routes.
- `static/`   -- the page itself.
"""

from .world import Setup

__all__ = ["Setup"]
