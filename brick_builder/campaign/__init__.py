"""The closed-loop campaign: perform the tasks, train on what happened, redeploy, measure.

One round is four phases (`campaign.py`):

    perform -> train -> redeploy -> evaluate

**Perform** runs `N` episodes of each task on fresh seeds with whatever models the campaign has
deployed, logging one tuple file per closed-loop cycle. **Train** fits the next-scene-state
predictor `F` and the one-step inverse `G` on everything logged so far, through
`brick_builder.action.learn.rounds`. **Redeploy** copies that round's checkpoints onto the pair
the campaign's own runs load and gives them a version id. **Evaluate** scores every task on the
held-out seeds, with the learned inverse and with the analytic one.

The tasks are in `tasks.py`: `placement`, `pick_and_place`, and `build_2`, a two-brick build
compiled from one English command by `brick_builder.language`. Seeds 0-9 are the held-out
evaluation set and are never performed on.
"""

from __future__ import annotations

from .campaign import Campaign, CampaignConfig, RoundRecord
from .tasks import (
    BUILD_COMMAND, EVAL_SEEDS, TASKS, BrickRecord, EpisodeRecord, TaskSpec, build_goal,
    perform_seeds, run_episode,
)

__all__ = [
    "BUILD_COMMAND", "EVAL_SEEDS", "TASKS", "BrickRecord", "Campaign", "CampaignConfig",
    "EpisodeRecord", "RoundRecord", "TaskSpec", "build_goal", "perform_seeds", "run_episode",
]
