"""Shared utilities for failure discovery."""

import abc
from typing import Any, SupportsFloat

import gymnasium as gym
from gymnasium.core import ActType, ObsType
from gymnasium.wrappers import RecordVideo

from gym_failure_discovery.failure_monitor_wrapper import FailureMonitorWrapper
from gym_failure_discovery.failure_monitors.failure_monitor import FailureMonitor


class Policy(abc.ABC):
    """A stateful policy that maps observations to actions."""

    def reset(self) -> None:
        """Reset any internal state at the start of an episode."""

    @abc.abstractmethod
    def act(self, obs: Any) -> Any:
        """Return an action given the current observation."""


def rollout(
    env: gym.Env[ObsType, ActType],
    monitor: FailureMonitor,
    policy: Policy,
    seed: int,
    max_steps: int,
) -> list[tuple[ObsType, ActType]] | None:
    """Roll out a policy and return the trajectory if a failure occurs.

    Returns a list of (observation, action) pairs leading to the
    failure, or None if no failure was detected within *max_steps*.
    """
    wrapped = FailureMonitorWrapper(env, monitor)
    obs, _ = wrapped.reset(seed=seed)
    policy.reset()
    trajectory: list[tuple[ObsType, ActType]] = []
    for _ in range(max_steps):
        action: ActType = policy.act(obs)
        trajectory.append((obs, action))
        obs, reward, terminated, truncated, _ = wrapped.step(action)
        if reward == 1.0:
            return trajectory
        if terminated or truncated:
            break
    return None


class RecordBufferedVideo(RecordVideo):  # type: ignore[type-arg]
    """Like RecordVideo, but also captures buffered intermediate frames.

    Some environments (e.g. BlocksEnv) execute many internal simulation
    steps per high-level ``step()`` call and buffer the intermediate
    renders.  This wrapper drains that buffer so every intermediate frame
    appears in the recorded video.

    Falls back to standard single-frame capture for environments that
    don't implement ``pop_frame_buffer()``.
    """

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, rew, terminated, truncated, info = self.env.step(action)
        self.step_id += 1

        if self.step_trigger and self.step_trigger(self.step_id):
            self.start_recording(f"{self.name_prefix}-step-{self.step_id}")

        if self.recording:
            unwrapped = self.env.unwrapped
            if hasattr(unwrapped, "pop_frame_buffer"):
                frames = unwrapped.pop_frame_buffer()
                self.recorded_frames.extend(frames)
            else:
                self._capture_frame()  # type: ignore[no-untyped-call]

            if len(self.recorded_frames) > self.video_length:
                self.stop_recording()  # type: ignore[no-untyped-call]

        return obs, rew, terminated, truncated, info
