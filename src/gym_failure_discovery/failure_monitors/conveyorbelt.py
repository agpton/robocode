"""Failure monitor for the conveyor belt environment.

The conveyor belt explodes when the agent issues a specific consecutive
sequence of drop-rate mode actions in the correct order.  The four available
modes are: off (0), slow (1), mid (2), fast (3).  The secret sequence has a
fixed length; once the most-recent actions match it exactly, the explosion
is triggered and ``next_obs["exploded"]`` becomes True.  The agent must
search over orderings and repetitions of the four modes to discover the
sequence.
"""

from typing import Any

from gym_failure_discovery.failure_monitors.failure_monitor import FailureMonitor


class ConveyorBeltFailureMonitor(FailureMonitor):
    """Detects when the conveyor belt explodes.

    The explosion is triggered by issuing the correct consecutive sequence of
    drop-rate modes (off=0, slow=1, mid=2, fast=3).  The sequence length and
    exact ordering are unknown to the agent; the failure finder must discover
    them by systematically varying the stream of mode actions issued to the
    environment.  Once triggered, ``next_obs["exploded"]`` is True.
    """

    def reset(self, obs: Any) -> None:
        pass

    def step(self, obs: Any, action: Any, next_obs: Any) -> bool:
        return next_obs["exploded"]
