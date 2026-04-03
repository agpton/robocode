"""Failure monitor for the hovercraft environment."""

from typing import Any

from tomsgeoms2d.structs import Circle, geom2ds_intersect

from robocode.environments.hovercraft_env import HoverCraftSceneSpec
from gym_failure_discovery.failure_monitors.failure_monitor import FailureMonitor


class HoverCraftFailureMonitor(FailureMonitor):
    """Detects collisions between the hovercraft and obstacles."""

    def __init__(self, scene_spec: HoverCraftSceneSpec) -> None:
        self._scene_spec = scene_spec

    def reset(self, obs: Any) -> None:
        """No internal state to reset."""

    def step(self, obs: Any, action: Any, next_obs: Any) -> bool:
        """Return True if the hovercraft collides with an obstacle."""
        x, _, y, _ = next_obs[:4]
        circ = Circle(float(x), float(y), self._scene_spec.hovercraft_radius)
        return any(geom2ds_intersect(circ, o) for o in self._scene_spec.obstacles)
