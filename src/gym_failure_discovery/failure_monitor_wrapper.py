"""Gymnasium wrapper that turns failure detection into rewards."""

from typing import Any

import gymnasium as gym
from gymnasium.core import ActType, ObsType

from gym_failure_discovery.failure_monitors.failure_monitor import FailureMonitor


class FailureMonitorWrapper(gym.Wrapper[ObsType, ActType, ObsType, ActType]):
    """Wraps an environment with a failure monitor.

    Rewards are 0 on every step except when the monitor detects a
    failure, in which case the reward is 1 and the episode terminates.
    The underlying environment's reward is discarded.
    """

    def __init__(self, env: gym.Env[ObsType, ActType], monitor: FailureMonitor) -> None:
        super().__init__(env)
        self._monitor = monitor
        self._last_obs: ObsType | None = None

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        obs, info = self.env.reset(seed=seed, options=options)
        self._monitor.reset(obs)
        self._last_obs = obs
        return obs, info

    def step(
        self, action: ActType
    ) -> tuple[ObsType, float, bool, bool, dict[str, Any]]:
        assert self._last_obs is not None
        obs, _, terminated, truncated, info = self.env.step(action)
        failure = self._monitor.step(self._last_obs, action, obs)
        self._last_obs = obs
        if failure:
            return obs, 1.0, True, truncated, info
        return obs, 0.0, terminated, truncated, info
