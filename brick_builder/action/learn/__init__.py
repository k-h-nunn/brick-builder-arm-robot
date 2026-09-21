"""Collecting tuples from the arm's own motion, training the action models, and measuring
them -- offline and in the closed loop.

None of this is on the control path: `collect.py` drives whole episodes and `evaluate.py` reads
the simulator to score them, both of which the controller itself may not do.
"""
