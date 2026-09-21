"""Generic gymnasium wrappers shared by every BrickBuilder task.

These sit on top of :class:`~brick_builder.envs.base_env.ArmEnv`'s ``Dict{"actor", "critic"}``
observation and single ``Box`` action; nothing here is task-specific. The base env already gives
every proprioceptive actor entry a 4-tick history and lets ``randomize.sample`` draw
``command.obs_delay_ticks`` for a task to consume (design report 01 section 6's command-path
group). :class:`ObservationDelay` is that consumer: a task wires the sampled delay into the
wrapper itself, since the delay is a property of one episode's command path, not of the raw env.
"""

from __future__ import annotations

from collections import deque

import gymnasium as gym
import numpy as np


class HistoryStack(gym.ObservationWrapper):
    """Stack the last ``k`` raw ``Dict`` observations, oldest first, per key.

    This is on top of (not instead of) the 4-tick history every proprioceptive actor entry
    already carries: it exists for a caller that wants a longer window than the base env bakes
    in, or that wraps a non-``ArmEnv`` environment with the same ``Dict{"actor","critic"}``
    contract and no history of its own. ``k=1`` is a no-op pass-through.
    """

    def __init__(self, env: gym.Env, k: int = 2) -> None:
        super().__init__(env)
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        self.k = int(k)
        self._buffers: dict[str, deque] = {}
        self.observation_space = gym.spaces.Dict({
            key: self._stacked_space(space) for key, space in env.observation_space.spaces.items()
        })

    def _stacked_space(self, space: gym.spaces.Box) -> gym.spaces.Box:
        low = np.tile(space.low, self.k)
        high = np.tile(space.high, self.k)
        return gym.spaces.Box(low=low, high=high, dtype=space.dtype)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._buffers = {
            key: deque([np.asarray(v, dtype=np.float32).copy() for _ in range(self.k)], maxlen=self.k)
            for key, v in obs.items()
        }
        return self.observation(obs), info

    def observation(self, observation: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        for key, value in observation.items():
            self._buffers[key].append(np.asarray(value, dtype=np.float32))
        return {key: np.concatenate(list(buf)) for key, buf in self._buffers.items()}


class ObservationDelay(gym.Wrapper):
    """Return the observation from ``delay_ticks`` steps ago instead of the current one.

    Models the latency between when the world was actually in some state and when a stale
    perception/telemetry pipeline hands the policy that state -- the deployable use of
    ``randomize.sample(...)["command.obs_delay_ticks"]``. ``delay_ticks=0`` is a no-op. The
    reward, ``terminated``/``truncated`` and ``info`` returned by ``step`` are always the
    *current* tick's (a real control loop always knows whether its own last command finished the
    episode; only the sensory picture lags), only the returned observation is old.
    """

    def __init__(self, env: gym.Env, delay_ticks: int = 0) -> None:
        super().__init__(env)
        if delay_ticks < 0:
            raise ValueError(f"delay_ticks must be >= 0, got {delay_ticks}")
        self.delay_ticks = int(delay_ticks)
        self._buffer: deque = deque(maxlen=self.delay_ticks + 1)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._buffer = deque([obs] * (self.delay_ticks + 1), maxlen=self.delay_ticks + 1)
        return self._buffer[0], info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._buffer.append(obs)
        delayed = self._buffer[0]
        return delayed, reward, terminated, truncated, info


class ActionRepeatLatency(gym.Wrapper):
    """Repeat each action for ``repeat`` ticks, and/or delay when an action takes effect.

    ``repeat`` runs the same action for that many inner ``env.step`` calls, summing reward and
    returning the final tick's observation/info -- the usual "act every k ticks" wrapper.
    ``latency_ticks`` additionally queues the action so it only starts being applied that many
    ticks after it was chosen (the previous action, or a zero action before the queue has filled,
    is applied in the meantime): a coarse, wrapper-level stand-in for a slower outer control loop
    sitting in front of the 50 Hz tick, distinct from ``ServoBus``'s own per-joint command delay.
    """

    def __init__(self, env: gym.Env, repeat: int = 1, latency_ticks: int = 0) -> None:
        super().__init__(env)
        if repeat < 1:
            raise ValueError(f"repeat must be >= 1, got {repeat}")
        if latency_ticks < 0:
            raise ValueError(f"latency_ticks must be >= 0, got {latency_ticks}")
        self.repeat = int(repeat)
        self.latency_ticks = int(latency_ticks)
        self._queue: deque = deque()

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        zero = np.zeros(self.env.action_space.shape, dtype=self.env.action_space.dtype)
        self._queue = deque([zero.copy() for _ in range(self.latency_ticks)], maxlen=None)
        return obs, info

    def step(self, action):
        action = np.asarray(action)
        self._queue.append(action)
        applied = self._queue.popleft() if len(self._queue) > self.latency_ticks else np.zeros_like(action)

        total_reward = 0.0
        obs = info = None
        terminated = truncated = False
        for _ in range(self.repeat):
            obs, reward, terminated, truncated, info = self.env.step(applied)
            total_reward += float(reward)
            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, info


class EpisodeStats(gym.Wrapper):
    """Track per-episode return/length/success and attach them to ``info`` when an episode ends.

    Deliberately small and BrickBuilder-specific rather than a re-export of
    ``gymnasium.wrappers.RecordEpisodeStatistics``: it also reads the ``success``/``failure``
    keys :class:`~brick_builder.envs.base_env.ArmEnv` already puts in ``info`` at episode end and
    folds them into a rolling window (``success_rate``), which the plain gymnasium wrapper knows
    nothing about.
    """

    def __init__(self, env: gym.Env, window: int = 100) -> None:
        super().__init__(env)
        self.window = int(window)
        self._successes: deque = deque(maxlen=self.window)
        self._return = 0.0
        self._length = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._return = 0.0
        self._length = 0
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._return += float(reward)
        self._length += 1
        if terminated or truncated:
            success = bool(info.get("success", False))
            self._successes.append(success)
            info = dict(info)
            info["episode"] = {
                "r": self._return,
                "l": self._length,
                "success": success,
                "success_rate": float(np.mean(self._successes)) if self._successes else float("nan"),
            }
        return obs, reward, terminated, truncated, info
