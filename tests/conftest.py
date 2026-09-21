"""Shared pytest fixtures and setup for the BrickBuilder test suite."""

import os

# MuJoCo picks its rendering backend at import time, so this must run before
# mujoco is imported anywhere (including by other test modules or fixtures).
os.environ.setdefault("MUJOCO_GL", "egl")
