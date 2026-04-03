"""Episode-running, video-saving, and approach-loading utilities."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import numpy as np
from numpy.typing import NDArray
from pybullet_helpers.inverse_kinematics import InverseKinematicsError

from robocode.approaches.base_approach import BaseApproach

logger = logging.getLogger(__name__)


def load_generated_approach(
    path: Path,
    action_space: Any,
    observation_space: Any,
    primitives: dict[str, Any],
) -> Any:
    """Load a ``GeneratedApproach`` class from the given file.

    Temporarily adds the parent directory of *path* to ``sys.path`` so that
    ``approach.py`` can import sibling modules written by the agent, then
    removes it to avoid polluting the global import path.
    """
    sandbox_dir = str(path.parent.resolve())
    if sandbox_dir not in sys.path:
        sys.path.insert(0, sandbox_dir)
    try:
        source = path.read_text()
        # Set __file__ so the exec'd code can use it (e.g. to locate
        # sibling modules via os.path.dirname(__file__)).  exec() does
        # not set this automatically unlike a normal module import.
        namespace: dict[str, Any] = {"__file__": str(path)}
        exec(compile(source, str(path), "exec"), namespace)  # pylint: disable=exec-used
    finally:
        sys.path.remove(sandbox_dir)
    cls = namespace["GeneratedApproach"]
    instance = cls(action_space, observation_space, primitives=primitives)
    logger.info("Loaded generated approach from %s", path)
    return instance


def run_episode(
    env: Any,
    approach: BaseApproach,
    seed: int,
    max_steps: int,
    render: bool = False,
) -> tuple[dict[str, Any], list[NDArray[np.uint8]]]:
    """Run a single evaluation episode and return metrics + frames."""
    state, info = env.reset(seed=seed)
    approach.reset(state, info)

    frames: list[NDArray[np.uint8]] = []

    def _capture() -> None:
        rendered: Any = env.render()
        if isinstance(rendered, np.ndarray):
            frames.append(rendered)

    if render:
        _capture()

    total_reward = 0.0
    num_steps = 0
    terminated = False
    consecutive_ik_failures = 0
    max_consecutive_ik_failures = 10
    for _ in range(max_steps):
        action = approach.step()
        # Save robot joints before stepping so we can restore them if
        # IK fails partway through an action (e.g. pick moves the arm
        # down then the lift-up raises InverseKinematicsError, leaving
        # the joints in a corrupted intermediate configuration).
        saved_joints = None
        if hasattr(env, "robot"):
            saved_joints = list(env.robot.get_joint_positions())
        try:
            state, reward, terminated, truncated, info = env.step(action)
            consecutive_ik_failures = 0
        except InverseKinematicsError:
            consecutive_ik_failures += 1
            logger.warning(
                "IK failed for action %s (%d consecutive); treating as no-op",
                action,
                consecutive_ik_failures,
            )
            # Restore robot joints to pre-action state so the env is
            # not left in a corrupted intermediate configuration.
            if saved_joints is not None:
                env.robot.set_joints(saved_joints)
            # Fetch the actual observation (block positions may have
            # shifted due to partial execution + physics).
            if hasattr(env, "_get_obs"):
                state = env._get_obs()
            reward = 0.0
            terminated = False
            truncated = consecutive_ik_failures >= max_consecutive_ik_failures
            info = {"ik_failed": True}
            if truncated:
                logger.warning(
                    "Truncating episode after %d consecutive IK failures",
                    max_consecutive_ik_failures,
                )
        total_reward += float(reward)
        num_steps += 1
        approach.update(state, float(reward), terminated or truncated, info)
        if render:
            _capture()
        if terminated or truncated:
            break

    metrics = {
        "total_reward": total_reward,
        "num_steps": num_steps,
        "solved": bool(terminated),
    }
    return metrics, frames


def save_video(frames: list[NDArray[np.uint8]], path: Path, fps: int = 10) -> None:
    """Save a list of RGB frames as a gif."""
    duration = 1000.0 / fps  # ms per frame
    iio.imwrite(str(path), frames, duration=duration, loop=0)
    logger.info("Saved video to %s", path)


def save_frames(
    frames: list[NDArray[np.uint8]],
    output_dir: Path,
    max_frames: int | None = None,
) -> list[str]:
    """Save frames as individual PNGs, returning the list of filenames."""
    output_dir.mkdir(parents=True, exist_ok=True)
    to_save = frames[:max_frames] if max_frames is not None else frames
    filenames: list[str] = []
    for i, frame in enumerate(to_save):
        filename = f"frame_{i:04d}.png"
        iio.imwrite(str(output_dir / filename), frame)
        filenames.append(filename)
    logger.info("Saved %d frames to %s", len(filenames), output_dir)
    return filenames
