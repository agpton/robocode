"""Hovercraft environment with an internal LQR controller.

The agent issues high-level commands (switch goal pair or not) and an
LQR controller translates them into continuous forces applied to the
hovercraft.  The observation is the full physical state.
"""

from dataclasses import dataclass, field
from typing import Any

import control as ct
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from gymnasium.core import RenderFrame
from gymnasium.spaces import Discrete
from numpy.typing import NDArray
from tomsgeoms2d.structs import Circle, Geom2D, Rectangle
from prpl_utils.utils import fig2data


@dataclass(frozen=True)
class HoverCraftSceneSpec:
    """Static hyperparameters for the hovercraft environment."""

    dt: float = 0.1

    # Initial state ranges.
    init_x_range: tuple[float, float] = (-0.2, 0.2)
    init_vx: float = 0
    init_y_range: tuple[float, float] = (-0.2, 0.2)
    init_vy: float = 0

    # Scene geometry.
    scene_width: float = 1
    scene_height: float = 1
    hovercraft_radius: float = 0.04

    obstacles: list[Geom2D] = field(
        default_factory=lambda: [
            Rectangle(-0.5, -0.5, 0.3, 0.3, 0.0),
            Rectangle(0.2, -0.5, 0.3, 0.3, 0.0),
            Rectangle(0.2, 0.2, 0.3, 0.3, 0.0),
            Rectangle(-0.5, 0.2, 0.3, 0.3, 0.0),
        ]
    )

    # Goal pairs: the hovercraft shuttles between the two goals in a pair.
    goal_pairs: list[tuple[tuple[float, float], tuple[float, float]]] = field(
        default_factory=lambda: [
            ((-0.42, 0.0), (0.42, 0.0)),
            ((0.0, -0.42), (0.0, 0.42)),
        ]
    )
    goal_atol: float = 1e-3

    # Rendering.
    render_figscale: float = 5
    render_padding: float = 0.05
    hovercraft_color: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0)
    obstacle_color: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 1.0)
    goal_circle_color: tuple[float, float, float, float] = (0.0, 1.0, 0.0, 0.25)

    @property
    def A(self) -> NDArray:
        """Discrete-time state transition matrix."""
        return np.array(
            [
                [1, self.dt, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, self.dt],
                [0, 0, 0, 1],
            ]
        )

    @property
    def B(self) -> NDArray:
        """Discrete-time input matrix."""
        return np.array(
            [
                [self.dt**2 / 2, 0],
                [self.dt, 0],
                [0, self.dt**2 / 2],
                [0, self.dt],
            ]
        )

    @property
    def Q(self) -> NDArray:
        """LQR state cost matrix."""
        return np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.1, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.1],
            ]
        )

    @property
    def R(self) -> NDArray:
        """LQR input cost matrix."""
        return 1e-2 * np.eye(2)


class HoverCraftEnv(gym.Env[np.ndarray, int]):
    """A 2D hovercraft with an internal LQR controller.

    Action space: Discrete(2)
        0 = do not switch goal pair
        1 = switch goal pair

    Observation space: Box of shape (5,)
        [x, vx, y, vy, t]
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 10}
    env_description = "Hovercraft environment with LQR controller. Agent issues high-level commands."

    def __init__(
        self,
        scene_spec: HoverCraftSceneSpec | None = None,
        render_mode: str = "rgb_array",
    ) -> None:
        super().__init__()
        self.scene_spec = scene_spec or HoverCraftSceneSpec()
        self.render_mode = render_mode

        self.action_space = Discrete(2)  # type: ignore[assignment]
        low = np.array([-np.inf, -np.inf, -np.inf, -np.inf, 0.0])
        high = np.array([np.inf, np.inf, np.inf, np.inf, np.inf])
        self.observation_space = gym.spaces.Box(low, high, dtype=np.float64)

        # LQR gain.
        K, _, _ = ct.dlqr(
            self.scene_spec.A,
            self.scene_spec.B,
            self.scene_spec.Q,
            self.scene_spec.R,
        )
        self._K: NDArray = K

        # Internal state, set in reset().
        self._state_vec: NDArray | None = None
        self._time: float = 0.0
        self._goal_pair_index: tuple[int, int] = (0, 0)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed, options=options)
        assert self.np_random is not None

        x = self.np_random.uniform(*self.scene_spec.init_x_range)
        y = self.np_random.uniform(*self.scene_spec.init_y_range)
        self._state_vec = np.array(
            [x, self.scene_spec.init_vx, y, self.scene_spec.init_vy]
        )
        self._time = 0.0
        self._goal_pair_index = (0, 0)
        return self._get_obs(), {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        assert self._state_vec is not None
        switch = bool(action)

        gi, gj = self._goal_pair_index
        gx, gy = self.scene_spec.goal_pairs[gi][gj]
        goal_vec = np.array([gx, 0, gy, 0])

        # Toggle within a pair when the current goal is reached.
        if np.allclose(goal_vec, self._state_vec, atol=self.scene_spec.goal_atol):
            gj = int(not gj)

        # Switch pair if commanded.
        if switch:
            gi = int(not gi)
            gj = 0

        self._goal_pair_index = (gi, gj)
        gx, gy = self.scene_spec.goal_pairs[gi][gj]

        # LQR action.
        goal_vec = np.array([gx, 0, gy, 0])
        error_vec = self._state_vec - goal_vec
        action_vec = -self._K @ error_vec

        # Integrate dynamics.
        self._state_vec = (
            self.scene_spec.A @ self._state_vec + self.scene_spec.B @ action_vec
        )
        self._time += self.scene_spec.dt

        return self._get_obs(), 0.0, False, False, {}

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        assert self._state_vec is not None
        spec = self.scene_spec
        pad = spec.render_padding
        half_w = spec.scene_width / 2
        half_h = spec.scene_height / 2

        scale = spec.render_figscale
        fig, ax = plt.subplots(
            1,
            1,
            figsize=(
                scale * (2 * half_w + 2 * pad),
                scale * (2 * half_h + 2 * pad),
            ),
        )

        x, _, y, _ = self._state_vec
        circ = Circle(float(x), float(y), spec.hovercraft_radius)
        circ.plot(ax, facecolor=spec.hovercraft_color, edgecolor="black")

        for obstacle in spec.obstacles:
            obstacle.plot(ax, facecolor=spec.obstacle_color, edgecolor="black")

        for goal_pair in spec.goal_pairs:
            for gx, gy in goal_pair:
                Circle(gx, gy, spec.hovercraft_radius).plot(
                    ax, facecolor=spec.goal_circle_color
                )

        ax.set_xlim(-half_w, half_w)
        ax.set_ylim(-half_h, half_h)
        plt.tight_layout()
        img = fig2data(fig)
        plt.close()
        return img  # type: ignore

    def get_hovercraft_circle(self) -> Circle:
        """Helper for external collision checking."""
        assert self._state_vec is not None
        x, _, y, _ = self._state_vec
        return Circle(float(x), float(y), self.scene_spec.hovercraft_radius)

    def _get_obs(self) -> np.ndarray:
        assert self._state_vec is not None
        return np.concatenate([self._state_vec, [self._time]])
