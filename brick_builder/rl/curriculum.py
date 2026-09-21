"""The success-gated level scheduler the Deliverable 1 press campaign trains against.

Curriculum scheduling asks for two things that `PressEnv`'s original
``curriculum=(promote_at, window)`` hook does not do:

1. **Advance on a success rate over a long window** -- at least 0.8 over the last 200
   episodes, rather than a short streak that a lucky run of easy draws can clear.
2. **Keep about 20 % of episodes on earlier levels.** Without this the policy stops seeing
   level 0 the moment it clears it, and a network that has to keep one set of weights for
   every rung quietly forgets the easy one while it learns the hard one. Twenty per cent is
   enough to keep the old behaviour alive and cheap enough not to slow the climb.

The scheduler is deliberately a plain object with no MuJoCo, no torch and no environment
import, so it can be unit-tested on its own and so each worker process in a vectorized run
owns one privately. Nothing here is shared between processes: a `SyncVectorEnv` or
`AsyncVectorEnv` ends up with one scheduler per sub-environment, each climbing at its own
pace, which is what gives a training batch a mix of difficulties instead of a single
synchronized jump.

Only episodes run at the currently unlocked level count toward promotion. Replay episodes
are drawn from levels the policy has already cleared, so folding them into the same window
would inflate the rate and promote on the strength of work already done.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["CurriculumConfig", "LevelScheduler"]


@dataclass(frozen=True)
class CurriculumConfig:
    """How the press curriculum advances.

    ``promote_at`` and ``window`` are the spec's own numbers (0.8 over 200 episodes);
    ``replay_frac`` is the share of episodes deliberately spent below the unlocked level.
    ``min_level`` exists so a campaign that has already decided level 0 is solved can start
    the replay floor higher.
    """

    promote_at: float = 0.8
    window: int = 200
    replay_frac: float = 0.2
    min_level: int = 0

    def __post_init__(self) -> None:
        if not 0.0 < self.promote_at <= 1.0:
            raise ValueError(f"promote_at must be in (0, 1], got {self.promote_at}")
        if self.window < 1:
            raise ValueError(f"window must be at least 1, got {self.window}")
        if not 0.0 <= self.replay_frac < 1.0:
            raise ValueError(f"replay_frac must be in [0, 1), got {self.replay_frac}")


@dataclass
class LevelScheduler:
    """One environment copy's own view of the curriculum.

    Usage, once per episode, in this order::

        scheduler.record(level_that_just_ran, episode_succeeded)
        level = scheduler.next_level()      # what to run now
        env.set_level(level)

    ``record`` is what promotes; ``next_level`` only reads the result. Keeping them apart
    means an environment whose episode ended in a way it does not want to score (a crash, a
    reset before the first step) can simply skip the ``record`` call.
    """

    n_levels: int
    cfg: CurriculumConfig = field(default_factory=CurriculumConfig)
    seed: int = 0

    unlocked: int = field(init=False, default=0)
    episodes: int = field(init=False, default=0)
    promotions: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        if self.n_levels < 1:
            raise ValueError(f"n_levels must be at least 1, got {self.n_levels}")
        self.unlocked = int(np.clip(self.cfg.min_level, 0, self.n_levels - 1))
        self._rng = np.random.default_rng(self.seed)
        self._window: list[float] = []
        # Per-level tallies for the report: how many episodes ran at each rung and how many
        # of them succeeded, counting replay episodes too (they are what says whether the
        # earlier levels are still solved).
        self._ran = np.zeros(self.n_levels, dtype=np.int64)
        self._won = np.zeros(self.n_levels, dtype=np.int64)

    # -- the two calls an environment makes ---------------------------------------------------

    def record(self, level: int, success: bool) -> bool:
        """Score the episode that just finished.  Returns True if this promoted the copy."""
        level = int(np.clip(level, 0, self.n_levels - 1))
        self.episodes += 1
        self._ran[level] += 1
        self._won[level] += int(bool(success))
        if level != self.unlocked:
            return False                      # a replay episode: kept in the tallies, not the gate
        self._window.append(1.0 if success else 0.0)
        del self._window[: -self.cfg.window]
        if (
            len(self._window) >= self.cfg.window
            and float(np.mean(self._window)) >= self.cfg.promote_at
            and self.unlocked < self.n_levels - 1
        ):
            self.unlocked += 1
            self.promotions += 1
            self._window.clear()
            return True
        return False

    def next_level(self) -> int:
        """The level the next episode should run at: usually the unlocked one, sometimes a
        replay of an earlier rung."""
        floor = min(self.cfg.min_level, self.unlocked)
        if self.unlocked > floor and self._rng.random() < self.cfg.replay_frac:
            return int(self._rng.integers(floor, self.unlocked))
        return int(self.unlocked)

    # -- diagnostics ---------------------------------------------------------------------------

    @property
    def window_rate(self) -> float:
        """Success rate over the current promotion window; NaN before any episode."""
        return float(np.mean(self._window)) if self._window else float("nan")

    @property
    def window_len(self) -> int:
        return len(self._window)

    def summary(self) -> dict:
        """A small, JSON-safe dictionary for the training log and the report."""
        return {
            "unlocked": int(self.unlocked),
            "episodes": int(self.episodes),
            "promotions": int(self.promotions),
            "window_len": int(len(self._window)),
            "window_rate": self.window_rate,
            "episodes_per_level": self._ran.tolist(),
            "successes_per_level": self._won.tolist(),
        }
