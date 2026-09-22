"""Start the page:

    python -m brick_builder.gui [--params my_params.yaml] [--port 8765]

Prints the address to open and then stays up until interrupted.
"""

from __future__ import annotations

import argparse
import os
import socket
import sys
import time
from pathlib import Path
from typing import Sequence

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m brick_builder.gui")
    parser.add_argument("--params", default=None,
                        help="parameter file to open with (configs/params/*.yaml, or a path)")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--quality", default="draft", choices=("live", "draft", "hd1080"),
                        help="main view size: live 960x540, draft 1280x720, hd1080 1920x1080")
    parser.add_argument("--out", default="media_out/gui",
                        help="where recordings, stills, logs and sweep tables are written")
    parser.add_argument("--verbose", action="store_true", help="log every request")
    args = parser.parse_args(argv)

    from .server import serve
    from .session import Session

    session = Session(params_file=args.params, quality_name=args.quality,
                      out_dir=Path(args.out))
    session.start()
    httpd = serve(session, host=args.host, port=args.port, verbose=args.verbose)

    shown = "127.0.0.1" if args.host in ("0.0.0.0", "") else args.host
    print(f"simulator page: http://{shown}:{args.port}")
    if args.host in ("0.0.0.0", ""):
        print(f"from another machine: http://{_host_address()}:{args.port}")
    print(f"output directory: {Path(args.out).resolve()}")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("stopping")
    finally:
        httpd.shutdown()
        session.close()
    return 0


def _host_address() -> str:
    """This machine's address on the network it routes out of, without asking for a name."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))
        return s.getsockname()[0]
    except OSError:
        return socket.gethostname()
    finally:
        s.close()


if __name__ == "__main__":
    sys.exit(main())
