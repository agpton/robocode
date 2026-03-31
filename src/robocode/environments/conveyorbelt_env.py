"""1D conveyor belt environment with packages.

The agent selects a drop-rate mode each step (off / slow / mid / fast).
The environment internally handles drop timing and physics.  A hidden
secret sequence of modes triggers an explosion when issued in order
— the agent's job is to discover this sequence through exploration.
"""

import hashlib
from dataclasses import dataclass, field
from typing import Any

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from gymnasium.core import RenderFrame

# Box, Dict added for llm
from gymnasium.spaces import Box, Dict, Discrete
from numpy.typing import NDArray
from prpl_utils.utils import fig2data

OFF = 0
SLOW = 1
MID = 2
FAST = 3
NUM_MODES = 4
MODE_NAMES = ["off", "slow", "mid", "fast"]
MAX_BOXES = 32  # fixed observation array length; slots beyond active boxes are 0


def _secret_hash(history: list[int], salt: str = "conveyorbelt_v1", length: int = 8) -> bool:
    """Check whether the tail of *history* matches the secret sequence.

    The sequence is derived from a hash and compared in constant time so that
    it never materialises as a stored attribute the agent could inspect.
    """
    if len(history) < length:
        return False
    digest = hashlib.sha256(salt.encode()).digest()
    tail = history[-length:]
    return all(tail[i] == digest[i] % NUM_MODES for i in range(length))


@dataclass(frozen=True)
class ConveyorBeltSceneSpec:
    """Static hyperparameters for the conveyor belt environment."""

    conveyor_belt_velocity: float = 1.0
    dt: float = 0.01
    belt_length: float = 5.0
    box_width: float = 0.4
    box_height: float = 0.5
    gravity: float = 9.81
    drop_start_height: float = 1.0
    initial_drop_position: float = 0.0


class ConveyorBeltEnv(gym.Env[dict[str, Any], int]):
    """1D conveyor belt with mode-controlled package drops.

    Action space: Discrete(4)
        0 = off, 1 = slow, 2 = mid, 3 = fast

    Observation space: dict with
        - "positions": (n,) float array of package positions
        - "falling_heights": (n,) float array of heights above belt
        - "exploded": bool
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}
    env_description = "1D conveyor belt environment with packages. Agent selects drop-rate mode each step."

    def __init__(
        self,
        scene_spec: ConveyorBeltSceneSpec | None = None,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        if scene_spec is None:
            scene_spec = ConveyorBeltSceneSpec()
        self.scene_spec = scene_spec
        self.render_mode = render_mode
        self.action_space = Discrete(NUM_MODES)  # type: ignore[assignment]
        self.observation_space = Dict(  # type: ignore[assignment]
            {
                "positions": Box(
                    low=0.0,
                    high=scene_spec.belt_length,
                    shape=(MAX_BOXES,),
                    dtype=np.float32,
                ),
                "falling_heights": Box(
                    low=0.0,
                    high=scene_spec.drop_start_height,
                    shape=(MAX_BOXES,),
                    dtype=np.float32,
                ),
                "exploded": Discrete(2),
            }
        )
        # added -- for llm -- END
        # Mode timing: steps between drops.
        dt = scene_spec.dt
        self._mode_to_steps: dict[int, int | None] = {
            OFF: None,
            SLOW: int(2.0 / dt),
            MID: int(1.2 / dt),
            FAST: int(0.04 / dt),
        }

        # State.
        self._positions: NDArray[np.float32] = np.array([], dtype=np.float32)
        self._falling_heights: NDArray[np.float32] = np.array([], dtype=np.float32)
        self._step_count: int = 0
        self._exploded: bool = False
        self._steps_since_last_drop: int = 10**9
        self._mode_history: list[int] = []

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        super().reset(seed=seed, options=options)
        self._positions = np.array([], dtype=np.float32)
        self._falling_heights = np.array([], dtype=np.float32)
        self._step_count = 0
        self._exploded = False
        self._steps_since_last_drop = 10**9
        self._mode_history = []
        return self._get_obs(), {}

    def step(
        self, action: int
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        mode = int(action)
        self._mode_history.append(mode)

        # Check secret sequence (computed on the fly, never stored).
        if _secret_hash(self._mode_history):
            self._exploded = True

        if not self._exploded:
            self._physics_step(mode)

        self._step_count += 1
        obs = self._get_obs()
        return obs, 0.0, self._exploded, False, {}

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        return self._render_frame()  # type: ignore[return-value]

    def _get_obs(self) -> dict[str, Any]:
        positions = np.zeros(MAX_BOXES, dtype=np.float32)
        falling_heights = np.zeros(MAX_BOXES, dtype=np.float32)
        n = min(len(self._positions), MAX_BOXES)
        positions[:n] = self._positions[:n]
        falling_heights[:n] = self._falling_heights[:n]
        return {
            "positions": positions,
            "falling_heights": falling_heights,
            "exploded": self._exploded,
        }

    @property
    def exploded(self) -> bool:
        """Whether the secret sequence has been triggered."""
        return self._exploded

    def _physics_step(self, mode: int) -> None:
        dt = self.scene_spec.dt
        gravity = self.scene_spec.gravity
        box_width = self.scene_spec.box_width
        belt_length = self.scene_spec.belt_length

        # Update falling heights.
        falling = self._falling_heights.copy()
        mask = falling > 0
        falling[mask] -= gravity * dt
        falling = np.maximum(falling, 0.0)
        self._falling_heights = falling

        # Move packages on belt.
        on_belt = self._falling_heights == 0
        self._positions[on_belt] += self.scene_spec.conveyor_belt_velocity * dt

        # Decide whether to drop.
        self._steps_since_last_drop += 1
        steps_required = self._mode_to_steps[mode]
        drop = False
        if steps_required is not None and self._steps_since_last_drop >= steps_required:
            if self._safe_to_drop():
                drop = True

        if drop:
            self._steps_since_last_drop = 0
            self._positions = np.append(
                self._positions, self.scene_spec.initial_drop_position
            ).astype(np.float32)
            self._falling_heights = np.append(
                self._falling_heights, self.scene_spec.drop_start_height
            ).astype(np.float32)

        # Remove packages off the belt.
        keep = (self._positions < belt_length - box_width) & (self._positions >= 0)
        self._positions = self._positions[keep]
        self._falling_heights = self._falling_heights[keep]

    def _safe_to_drop(self) -> bool:
        drop_pos = self.scene_spec.initial_drop_position
        box_w = self.scene_spec.box_width
        for pos, h in zip(self._positions, self._falling_heights):
            if h > 0:
                return False
            if abs(pos - drop_pos) < box_w:
                return False
        return True

    def _render_frame(self) -> np.ndarray:
        fig, ax = plt.subplots(figsize=(10, 3))
        spec = self.scene_spec
        ax.set_xlim(0, spec.belt_length)
        ax.set_ylim(0, 2)
        ax.axis("off")

        ax.add_patch(
            plt.Rectangle(
                (0, 0),
                spec.belt_length,
                2,
                fill=False,
                edgecolor="red",
                linewidth=2,
                zorder=10,
            )
        )
        ax.add_patch(plt.Rectangle((0, 0.4), spec.belt_length, 0.3, color="#777777"))
        ax.set_title(f"Conveyor Belt: {len(self._positions)} Boxes")

        belt_y = 0.3
        belt_height = 0.4
        y_pos = belt_y + belt_height

        ax.add_patch(
            plt.Rectangle(
                (0, belt_y),
                spec.belt_length,
                belt_height,
                color="#555555",
                zorder=0,
            )
        )
        top_coords = [
            (0, belt_y + belt_height),
            (0.3, belt_y + belt_height + 0.15),
            (spec.belt_length + 0.3, belt_y + belt_height + 0.15),
            (spec.belt_length, belt_y + belt_height),
        ]
        ax.add_patch(plt.Polygon(top_coords, color="#777777", zorder=2))
        side_coords = [
            (spec.belt_length, belt_y),
            (spec.belt_length + 0.3, belt_y + 0.15),
            (spec.belt_length + 0.3, belt_y + belt_height + 0.15),
            (spec.belt_length, belt_y + belt_height),
        ]
        ax.add_patch(plt.Polygon(side_coords, color="#444444", zorder=1))

        # Rollers.
        roller_radius = 0.05
        roller_spacing = 1.0
        roller_count = int(np.ceil(spec.belt_length / roller_spacing)) + 1
        time_elapsed = self._step_count * spec.dt
        angle = -(time_elapsed * 2 * np.pi) % (2 * np.pi)

        for i in range(roller_count):
            rx = i * roller_spacing - 0.1
            ry = belt_y - roller_radius
            ax.add_patch(plt.Circle((rx, ry), roller_radius, color="#222222", zorder=0))
            spoke = roller_radius * 0.7
            for a in [angle, angle + np.pi / 2]:
                ax.plot(
                    [rx - spoke * np.cos(a), rx + spoke * np.cos(a)],
                    [ry - spoke * np.sin(a), ry + spoke * np.sin(a)],
                    color="#AAAAAA",
                    zorder=1,
                    lw=0.5,
                )

        # Packages.
        for pos, height in zip(self._positions, self._falling_heights, strict=True):
            if pos < 0 or pos > spec.belt_length:
                continue
            y = y_pos + height
            base_color = plt.get_cmap("viridis")(0.5)
            ax.add_patch(
                plt.Rectangle(
                    (pos, y),
                    spec.box_width,
                    spec.box_height,
                    color=base_color,
                    zorder=3,
                )
            )
            top_rgb = np.minimum(np.array(base_color[:3]) * 1.3, 1.0)
            side_rgb = np.array(base_color[:3]) * 0.7
            ax.add_patch(
                plt.Polygon(
                    [
                        (pos, y + spec.box_height),
                        (pos + 0.15, y + spec.box_height + 0.15),
                        (pos + spec.box_width + 0.15, y + spec.box_height + 0.15),
                        (pos + spec.box_width, y + spec.box_height),
                    ],
                    color=(*top_rgb, 1.0),
                    zorder=4,
                )
            )
            ax.add_patch(
                plt.Polygon(
                    [
                        (pos + spec.box_width, y),
                        (pos + spec.box_width + 0.15, y + 0.15),
                        (pos + spec.box_width + 0.15, y + spec.box_height + 0.15),
                        (pos + spec.box_width, y + spec.box_height),
                    ],
                    color=(*side_rgb, 1.0),
                    zorder=2,
                )
            )

        plt.tight_layout()
        image = fig2data(fig)
        plt.close(fig)
        return image
