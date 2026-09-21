"""Lets `python -m brick_builder.experiments <name> ...` work; see `brick_builder/experiments/__init__.py`."""

from . import main

if __name__ == "__main__":
    raise SystemExit(main())
