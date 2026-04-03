"""Failure monitor for the blocks environment.

Detects when a non-held block is displaced during a high-level action,
which indicates the carried block (or the robot arm) knocked it.
With teleportation-style control, non-held blocks should not move
unless something collides with them.
"""

from typing import Any

import numpy as np

from robocode.environments.blocks_env import BlocksEnv, BlocksEnvState
from gym_failure_discovery.failure_monitors.failure_monitor import FailureMonitor


class BlocksFailureMonitor(FailureMonitor):
    """Collision monitor for the blocks environment.

    Checks whether any non-held block was displaced between the
    start and end of a high-level step.  With position-control
    teleportation, non-held blocks only move when something hits them.
    """

    def __init__(
        self,
        env: BlocksEnv,
        move_tol: float = 0.01,
    ) -> None:
        self._env = env
        self._move_tol = move_tol
        self._prev_state: BlocksEnvState | None = None

    def reset(self, obs: Any) -> None:
        self._prev_state = self._env.get_state()

    def step(self, obs: Any, action: Any, next_obs: Any) -> bool:
        assert self._prev_state is not None
        state = self._env.get_state()

        held_names: set[str] = set()
        if self._prev_state.held_block_idx >= 0:
            held_names.add(self._env.block_names[self._prev_state.held_block_idx])
        if state.held_block_idx >= 0:
            held_names.add(self._env.block_names[state.held_block_idx])

        for prev_block in self._prev_state.blocks:
            if prev_block.name in held_names:
                continue
            curr_block = state.get_block_state(prev_block.name)
            prev_pos = np.array(prev_block.pose.position)
            curr_pos = np.array(curr_block.pose.position)
            if not np.allclose(curr_pos, prev_pos, atol=self._move_tol):
                self._prev_state = state
                return True

        self._prev_state = state
        return False
