"""PyBullet block stacking environment with high-level actions.

The agent issues high-level commands (Pick, Unstack, Stack) and the
environment internally executes the corresponding motion plan using
an IK-based controller.  The ``safe_height`` parameter controls how
high the robot lifts blocks during transport — lower values make
collisions more likely.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import gymnasium as gym
import numpy as np
import pybullet as p
from gymnasium.core import RenderFrame
from gymnasium.spaces import Dict, Discrete
from pybullet_helpers.camera import capture_image
from pybullet_helpers.geometry import (
    Pose,
    get_pose,
    iter_between_poses,
    multiply_poses,
    set_pose,
)
from pybullet_helpers.gui import create_gui_connection
from pybullet_helpers.joint import JointPositions
from pybullet_helpers.motion_planning import (
    create_joint_distance_fn,
    smoothly_follow_end_effector_path,
)
from pybullet_helpers.robots import create_pybullet_robot
from pybullet_helpers.robots.single_arm import FingeredSingleArmPyBulletRobot
from pybullet_helpers.utils import create_pybullet_block

# Action types.
PICK = 0
UNSTACK = 1
STACK = 2
NUM_ACTION_TYPES = 3


@dataclass(frozen=True)
class BlocksSceneSpec:
    """Static hyperparameters for the blocks environment."""

    gravity: float = 9.80665
    num_sim_steps_per_action: int = 30
    safe_height: float = 0.25
    max_smoothing_iters_per_step: int = 1

    robot_name: str = "panda"
    robot_base_pose: Pose = Pose.identity()
    initial_joints: JointPositions = field(
        default_factory=lambda: [
            -1.6760817784086874,
            -0.8633617886115512,
            1.0820023618960484,
            -1.7862427129376002,
            0.7563762599673787,
            1.3595324116603988,
            1.7604148617061273,
            0.04,
            0.04,
        ]
    )

    robot_stand_pose: Pose = Pose((0.0, 0.0, -0.2))
    robot_stand_rgba: tuple[float, float, float, float] = (0.5, 0.5, 0.5, 1.0)
    robot_stand_half_extents: tuple[float, float, float] = (0.2, 0.2, 0.225)

    table_pose: Pose = Pose((0.5, 0.0, -0.175))
    table_rgba: tuple[float, float, float, float] = (0.5, 0.5, 0.5, 1.0)
    table_half_extents: tuple[float, float, float] = (0.25, 0.4, 0.25)

    block_rgba: tuple[float, float, float, float] = (0.5, 0.0, 0.5, 1.0)
    block_half_extents: tuple[float, float, float] = (0.025, 0.025, 0.025)
    block_mass: float = 0.5
    block_friction: float = 0.9
    num_blocks: int = 6

    @property
    def block_init_position_lower(self) -> tuple[float, float, float]:
        """Lower bounds for block position."""
        return (
            self.table_pose.position[0]
            - self.table_half_extents[0]
            + self.block_half_extents[0],
            self.table_pose.position[1]
            - self.table_half_extents[1]
            + self.block_half_extents[1],
            self.table_pose.position[2]
            + self.table_half_extents[2]
            + self.block_half_extents[2],
        )

    @property
    def block_init_position_upper(self) -> tuple[float, float, float]:
        """Upper bounds for block position."""
        return (
            self.table_pose.position[0]
            + self.table_half_extents[0]
            - self.block_half_extents[0],
            self.table_pose.position[1]
            + self.table_half_extents[1]
            - self.block_half_extents[1],
            self.table_pose.position[2]
            + self.table_half_extents[2]
            + self.block_half_extents[2],
        )


@dataclass(frozen=True)
class BlockState:
    """A state of a single block."""

    name: str
    pose: Pose


@dataclass(frozen=True)
class RobotState:
    """A state of the robot."""

    joint_positions: JointPositions


@dataclass(frozen=True)
class BlocksEnvState:
    """Full internal state of the blocks environment."""

    robot: RobotState
    blocks: list[BlockState]
    held_block_idx: int  # -1 if nothing held
    held_block_grasp: Pose | None = None

    def get_block_state(self, name: str) -> BlockState:
        """Get the state of a single block by name."""
        for block in self.blocks:
            if block.name == name:
                return block
        raise ValueError(f"Block with name {name} not in state")


class BlocksEnv(gym.Env[dict[str, Any], dict[str, Any]]):
    """A pybullet blocks environment with high-level actions.

    Action space (Dict):
        - "type": Discrete(3) — 0=Pick, 1=Unstack, 2=Stack
        - "block": Discrete(num_blocks) — which block to pick/unstack,
          or which block to stack onto

    Observation space (Dict):
        - "block_positions": (num_blocks, 3) float array
        - "held_block": int in [0, num_blocks], where num_blocks means
          nothing held
        - "robot_joints": (9,) float array
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 10}
    env_description = "PyBullet block stacking environment. Agent issues high-level commands (Pick, Unstack, Stack)."

    def __init__(
        self,
        scene_spec: BlocksSceneSpec | None = None,
        use_gui: bool = False,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        if scene_spec is None:
            scene_spec = BlocksSceneSpec()
        self.scene_spec = scene_spec
        self.render_mode = render_mode

        # PyBullet setup.
        if use_gui:
            self.physics_client_id = create_gui_connection(camera_yaw=90)
        else:
            self.physics_client_id = p.connect(p.DIRECT)

        p.setGravity(0, 0, -scene_spec.gravity, physicsClientId=self.physics_client_id)

        # Create robot.
        robot = create_pybullet_robot(
            scene_spec.robot_name,
            self.physics_client_id,
            base_pose=scene_spec.robot_base_pose,
            control_mode="reset",
            home_joint_positions=scene_spec.initial_joints,
        )
        assert isinstance(robot, FingeredSingleArmPyBulletRobot)
        robot.close_fingers()
        self.robot = robot

        # Create robot stand.
        self.robot_stand_id = create_pybullet_block(
            scene_spec.robot_stand_rgba,
            half_extents=scene_spec.robot_stand_half_extents,
            physics_client_id=self.physics_client_id,
        )
        set_pose(
            self.robot_stand_id, scene_spec.robot_stand_pose, self.physics_client_id
        )

        # Create table.
        self.table_id = create_pybullet_block(
            scene_spec.table_rgba,
            half_extents=scene_spec.table_half_extents,
            physics_client_id=self.physics_client_id,
        )
        set_pose(self.table_id, scene_spec.table_pose, self.physics_client_id)

        # Create blocks (poses set on reset).
        self.block_names = [f"block{i}" for i in range(scene_spec.num_blocks)]
        self.block_ids: dict[str, int] = {
            name: create_pybullet_block(
                scene_spec.block_rgba,
                scene_spec.block_half_extents,
                self.physics_client_id,
                scene_spec.block_mass,
                scene_spec.block_friction,
            )
            for name in self.block_names
        }

        # Grasp state.
        self._held_block_idx: int = -1
        self._held_grasp_transform: Pose | None = None

        # Frame buffer for intermediate renders during high-level steps.
        self._frame_buffer: list[np.ndarray] = []

        # Motion planning helper.
        self._joint_distance_fn = create_joint_distance_fn(self.robot)

        # Spaces.
        nb = scene_spec.num_blocks
        self.action_space = Dict(
            {
                "type": Discrete(NUM_ACTION_TYPES),
                "block": Discrete(nb),
            }
        )
        self.observation_space = Dict(
            {
                "block_positions": gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(nb, 3), dtype=np.float64
                ),
                "held_block": Discrete(nb + 1),
                "robot_joints": gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(9,), dtype=np.float64
                ),
            }
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        super().reset(seed=seed, options=options)

        # Reset robot.
        self.robot.set_joints(self.scene_spec.initial_joints)

        # Reset grasp.
        self._held_block_idx = -1
        self._held_grasp_transform = None

        # Place blocks in a line on the table.
        lx, ly, lz = self.scene_spec.block_init_position_lower
        ux, uy, uz = self.scene_spec.block_init_position_upper
        x = (lx + ux) / 2
        z = (lz + uz) / 2
        ys = np.linspace(ly, uy, num=self.scene_spec.num_blocks, endpoint=True)
        for i, name in enumerate(self.block_names):
            block_id = self.block_ids[name]
            set_pose(block_id, Pose((x, float(ys[i]), z)), self.physics_client_id)

        return self._get_obs(), {}

    def step(
        self, action: dict[str, Any]
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        action_type = int(action["type"])
        block_idx = int(action["block"])
        block_name = self.block_names[block_idx]

        if action_type in (PICK, UNSTACK):
            self._execute_pick(block_name)
        elif action_type == STACK:
            self._execute_stack(block_name)

        obs = self._get_obs()
        return obs, 0.0, False, False, {}

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        return self._capture_frame()  # type: ignore[return-value]

    def pop_frame_buffer(self) -> list[np.ndarray]:
        """Return and clear buffered intermediate frames."""
        frames = list(self._frame_buffer)
        self._frame_buffer.clear()
        return frames

    def _capture_frame(self) -> np.ndarray:
        """Take a snapshot of the current scene."""
        img = capture_image(
            self.physics_client_id,
            camera_target=self.scene_spec.robot_base_pose.position,
            camera_yaw=90,
            camera_distance=1.5,
            camera_pitch=-20,
        )
        background_mask = (img == [255, 255, 255]).all(axis=2)
        img[background_mask] = 0
        return img

    def close(self) -> None:
        p.disconnect(self.physics_client_id)

    def get_state(self) -> BlocksEnvState:
        """Get the full internal state (for failure monitoring)."""
        robot_state = RobotState(self.robot.get_joint_positions())
        block_states = [
            BlockState(name, get_pose(self.block_ids[name], self.physics_client_id))
            for name in self.block_names
        ]
        return BlocksEnvState(
            robot_state,
            block_states,
            self._held_block_idx,
            self._held_grasp_transform,
        )

    def _get_obs(self) -> dict[str, Any]:
        nb = self.scene_spec.num_blocks
        positions = np.zeros((nb, 3))
        for i, name in enumerate(self.block_names):
            pose = get_pose(self.block_ids[name], self.physics_client_id)
            positions[i] = pose.position
        held = self._held_block_idx if self._held_block_idx >= 0 else nb
        joints = np.array(self.robot.get_joint_positions())
        return {
            "block_positions": positions,
            "held_block": held,
            "robot_joints": joints,
        }

    def _execute_pick(self, block_name: str) -> None:
        """Execute a pick/unstack: move to block, close gripper, lift."""
        state = self.get_state()
        block_pose = state.get_block_state(block_name).pose
        self.robot.set_joints(state.robot.joint_positions)

        start_pose = self.robot.get_end_effector_pose()
        safe_h = self.scene_spec.safe_height

        waypoints = [
            start_pose,
            Pose(
                (start_pose.position[0], start_pose.position[1], safe_h),
                start_pose.orientation,
            ),
            Pose(
                (block_pose.position[0], block_pose.position[1], safe_h),
                start_pose.orientation,
            ),
            Pose(block_pose.position, start_pose.orientation),
        ]

        ee_path: list[Pose] = []
        for p1, p2 in zip(waypoints[:-1], waypoints[1:], strict=True):
            ee_path.extend(iter_between_poses(p1, p2))

        motion_plan = smoothly_follow_end_effector_path(
            self.robot,
            ee_path,
            state.robot.joint_positions,
            collision_ids=set(),
            joint_distance_fn=self._joint_distance_fn,
            max_smoothing_iters_per_step=self.scene_spec.max_smoothing_iters_per_step,
        )

        # Execute motion to the block.
        for joints in motion_plan:
            self._sim_step_with_joints(joints)

        # Close gripper — attempt grasp.
        self._attempt_grasp(block_name)

        # Lift up to safe height.
        up_waypoints = [waypoints[-1], waypoints[-2]]
        ee_path_up: list[Pose] = []
        for p1, p2 in zip(up_waypoints[:-1], up_waypoints[1:], strict=True):
            ee_path_up.extend(iter_between_poses(p1, p2))

        current_joints = self.robot.get_joint_positions()
        motion_plan_up = smoothly_follow_end_effector_path(
            self.robot,
            ee_path_up,
            current_joints,
            collision_ids=set(),
            joint_distance_fn=self._joint_distance_fn,
            max_smoothing_iters_per_step=self.scene_spec.max_smoothing_iters_per_step,
        )
        for joints in motion_plan_up:
            self._sim_step_with_joints(joints)

    def _execute_stack(self, target_block_name: str) -> None:
        """Execute a stack: move held block above target, open gripper."""
        if self._held_block_idx < 0:
            return
        state = self.get_state()
        target_pose = state.get_block_state(target_block_name).pose
        self.robot.set_joints(state.robot.joint_positions)

        start_pose = self.robot.get_end_effector_pose()
        safe_h = self.scene_spec.safe_height
        block_h = 2 * self.scene_spec.block_half_extents[2]

        waypoints = [
            start_pose,
            Pose(
                (start_pose.position[0], start_pose.position[1], safe_h),
                start_pose.orientation,
            ),
            Pose(
                (target_pose.position[0], target_pose.position[1], safe_h),
                start_pose.orientation,
            ),
            Pose(
                (
                    target_pose.position[0],
                    target_pose.position[1],
                    target_pose.position[2] + block_h,
                ),
                start_pose.orientation,
            ),
        ]

        ee_path: list[Pose] = []
        for p1, p2 in zip(waypoints[:-1], waypoints[1:], strict=True):
            ee_path.extend(iter_between_poses(p1, p2))

        motion_plan = smoothly_follow_end_effector_path(
            self.robot,
            ee_path,
            state.robot.joint_positions,
            collision_ids=set(),
            joint_distance_fn=self._joint_distance_fn,
            max_smoothing_iters_per_step=self.scene_spec.max_smoothing_iters_per_step,
        )

        for joints in motion_plan:
            self._sim_step_with_joints(joints)

        # Release.
        self._held_block_idx = -1
        self._held_grasp_transform = None

    def _attempt_grasp(self, block_name: str) -> None:
        """Try to grasp a block near the end effector."""
        block_id = self.block_ids[block_name]
        world_to_robot = self.robot.get_end_effector_pose()
        world_to_block = get_pose(block_id, self.physics_client_id)
        dist = float(
            np.sum(
                np.square(np.subtract(world_to_robot.position, world_to_block.position))
            )
        )
        if dist < 1e-3:
            self._held_grasp_transform = multiply_poses(
                world_to_robot.invert(), world_to_block
            )
            self._held_block_idx = self.block_names.index(block_name)

    def _sim_step_with_joints(self, target_joints: JointPositions) -> None:
        """Set robot joints, apply grasp transform, run physics."""
        clipped = np.clip(
            target_joints,
            self.robot.joint_lower_limits,
            self.robot.joint_upper_limits,
        ).tolist()

        for i in range(2):
            self.robot.set_joints(clipped)
            if self._held_grasp_transform is not None:
                world_to_robot = self.robot.get_end_effector_pose()
                world_to_block = multiply_poses(
                    world_to_robot, self._held_grasp_transform
                )
                held_name = self.block_names[self._held_block_idx]
                set_pose(
                    self.block_ids[held_name],
                    world_to_block,
                    self.physics_client_id,
                )
            if i == 0:
                for _ in range(self.scene_spec.num_sim_steps_per_action):
                    p.stepSimulation(physicsClientId=self.physics_client_id)

        if self.render_mode == "rgb_array":
            self._frame_buffer.append(self._capture_frame())
