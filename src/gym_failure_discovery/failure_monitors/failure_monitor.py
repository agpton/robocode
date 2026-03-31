"""Base class for failure monitors."""

import abc
from typing import Any


class FailureMonitor(abc.ABC):
    """Monitors environment transitions for failures."""

    @abc.abstractmethod
    def reset(self, obs: Any) -> None:
        """Reset the monitor at the start of an episode."""

    @abc.abstractmethod
    def step(self, obs: Any, action: Any, next_obs: Any) -> bool:
        """Return True if a failure occurred on this transition."""
