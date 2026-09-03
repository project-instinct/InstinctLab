"""Base event term for reset-to-fall-or-default dispatch.

Subclasses must implement ``_reset_by_default`` and ``_reset_by_fall``.
"""

from __future__ import annotations

import torch
from abc import abstractmethod
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.managers import ManagerTermBase, SceneEntityCfg

from .motion_reference import reset_robot_state_by_reference

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.managers import EventTermCfg

    from instinctlab.motion_reference.motion_reference_manager import MotionReferenceManager


class ResetToFallOrDefault(ManagerTermBase):
    """Base reset event that dispatches between a default reset and a fall reset.

    At each episode reset, each environment is assigned to either "default" mode
    or "fall" mode based on ``reset_to_fall_prob``. The ``reset_difficulty``
    attribute (mutable, shape ``(num_envs,)``) can be ramped by curriculum terms
    to control how extreme the fall resets are.

    Subclasses provide the actual reset logic by implementing
    :meth:`_reset_by_default` and :meth:`_reset_by_fall`.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        self.reset_as_fall_mode = torch.zeros((env.num_envs,), dtype=torch.bool, device=env.device)
        self.reset_difficulty = torch.ones((env.num_envs,), device=env.device) * cfg.params.get(
            "init_reset_difficulty", 0.0
        )

        self._reset_to_fall_prob = cfg.params.get("reset_to_fall_prob", 0.15)

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        reset_to_fall_prob: float = 0.15,
        init_reset_difficulty: float = 0.0,
    ):
        if len(env_ids) == 0:
            return

        self.reset_as_fall_mode[env_ids] = torch.rand(len(env_ids), device=env_ids.device) < self._reset_to_fall_prob

        fall_mask = self.reset_as_fall_mode[env_ids]
        default_ids = env_ids[~fall_mask]
        fall_ids = env_ids[fall_mask]

        if len(default_ids) > 0:
            self._reset_by_default(default_ids)
        if len(fall_ids) > 0:
            self._reset_by_fall(fall_ids)

    @abstractmethod
    def _reset_by_default(self, env_ids: torch.Tensor):
        """Reset environments to the default state."""
        ...

    @abstractmethod
    def _reset_by_fall(self, env_ids: torch.Tensor):
        """Reset environments to a fall / recovery state."""
        ...


class ResetToFallOrReference(ResetToFallOrDefault):
    """A reset event that chooses between resetting to the motion reference state or a random
    recovery state, controlled by a difficulty level.

    At difficulty 0.0 all environments reset to the motion reference state (with small
    randomization). At difficulty 1.0 all environments reset to a random recovery state sampled
    around the default joint configuration. Values in between produce a interpolate mix between
    the reference and the samples from _fall_..._range

    The ``difficulty`` attribute is mutable so that curriculum terms can ramp it during training.
    The ``reset_mode`` attribute is mutable so that this event can resample based on probability.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        self.asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.motion_ref_cfg: SceneEntityCfg = cfg.params["motion_ref_cfg"]
        self.asset: Articulation = env.scene[self.asset_cfg.name]
        self.motion_ref: MotionReferenceManager = env.scene[self.motion_ref_cfg.name]

        self._position_offset = cfg.params.get("position_offset", [0.0, 0.0, 0.0])

        self._randomize_joint_pos_range = cfg.params.get("randomize_joint_pos_range", [-0.1, 0.1])
        self._randomize_base_pose_range = cfg.params.get(
            "randomize_base_pose_range",
            {
                "x": [-0.2, 0.2],
                "y": [-0.2, 0.2],
                "z": [-0.1, 0.1],
                "roll": [-0.2, 0.2],
                "pitch": [-0.2, 0.2],
                "yaw": [0.5, 0.5],
            },
        )
        self._randomize_base_vel_range = cfg.params.get(
            "randomize_base_vel_range",
            {
                "x": [-0.2, 0.2],
                "y": [-0.2, 0.2],
                "z": [-0.1, 0.1],
                "roll": [-0.2, 0.2],
                "pitch": [-0.2, 0.2],
                "yaw": [0.5, 0.5],
            },
        )

        self._fall_joint_pos_range = cfg.params.get("fall_joint_pos_range", [-1.57, 1.57])
        self._fall_joint_vel_range = cfg.params.get("fall_joint_vel_range", [-3.14, 3.14])
        self._fall_base_pose_range = cfg.params.get(
            "fall_base_pose_range",
            {
                "x": [0.0, 0.0],
                "y": [0.0, 0.0],
                "z": [0.1, 1.5],
                "roll": [-1.57, 1.57],
                "pitch": [-1.57, 1.57],
                "yaw": [0.0, 0.0],
            },
        )
        self._fall_base_vel_range = cfg.params.get(
            "fall_base_vel_range",
            {
                "x": [-1.0, 1.0],
                "y": [-1.0, 1.0],
                "z": [-2.0, 1.0],
                "roll": [-3.14, 3.14],
                "pitch": [-3.14, 3.14],
                "yaw": [-3.14, 3.14],
            },
        )

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        motion_ref_cfg: SceneEntityCfg = SceneEntityCfg("motion_reference"),
        reset_to_fall_prob: float = 0.15,
        position_offset: list | None = None,
        randomize_joint_pos_range: tuple | None = None,
        randomize_base_pose_range: dict | None = None,
        randomize_base_vel_range: dict | None = None,
        fall_joint_pos_range: tuple | None = None,
        fall_joint_vel_range: tuple | None = None,
        fall_base_pose_range: dict | None = None,
        fall_base_vel_range: dict | None = None,
        init_reset_difficulty: float = 0.0,
    ):
        super().__call__(env, env_ids, reset_to_fall_prob=reset_to_fall_prob)

    def _sample_pose_range(self, range_dict, n):
        keys = ["x", "y", "z", "roll", "pitch", "yaw"]
        ranges = torch.tensor(
            [range_dict.get(k, [0.0, 0.0]) for k in keys],
            device=self.device,
        )
        return math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (n, 6), device=self.device)

    def _reset_by_default(self, env_ids: torch.Tensor):
        """Reset to motion reference with small randomization."""
        reset_robot_state_by_reference(
            self._env,
            env_ids,
            motion_ref_cfg=self.motion_ref_cfg,
            asset_cfg=self.asset_cfg,
            position_offset=self._position_offset,
            randomize_pose_range=self._randomize_base_pose_range,
            randomize_velocity_range=self._randomize_base_vel_range,
            randomize_joint_pos_range=tuple(self._randomize_joint_pos_range),
        )

    def _reset_by_fall(self, env_ids: torch.Tensor):
        """Reset via interpolation between reference and random fall state.

        alpha = reset_difficulty: 0 → reference, 1 → full fall.
        """
        n = len(env_ids)
        alpha = self.reset_difficulty[env_ids].unsqueeze(-1)

        # Reference state
        ref_state = self.motion_ref.get_init_reference_state(env_ids)

        # Fall state: sample around default
        root_default = self.asset.data.default_root_state.torch[env_ids].clone()
        pose_noise = self._sample_pose_range(self._fall_base_pose_range, n)
        fall_pos = root_default[:, 0:3] + self._env.scene.env_origins[env_ids] + pose_noise[:, 0:3]
        fall_quat_delta = math_utils.quat_from_euler_xyz(pose_noise[:, 3], pose_noise[:, 4], pose_noise[:, 5])
        fall_quat = math_utils.quat_mul(root_default[:, 3:7], fall_quat_delta)

        vel_noise = self._sample_pose_range(self._fall_base_vel_range, n)
        fall_lin_vel = root_default[:, 7:10] + vel_noise[:, 0:3]
        fall_ang_vel = root_default[:, 10:13] + vel_noise[:, 3:6]

        fall_joint_pos = self.asset.data.default_joint_pos.torch[env_ids].clone() + math_utils.sample_uniform(
            *self._fall_joint_pos_range,
            self.asset.data.default_joint_pos.torch[env_ids].shape,
            device=self.device,
        )
        fall_joint_vel = self.asset.data.default_joint_vel.torch[env_ids].clone() + math_utils.sample_uniform(
            *self._fall_joint_vel_range,
            self.asset.data.default_joint_vel.torch[env_ids].shape,
            device=self.device,
        )
        joint_limits = self.asset.data.soft_joint_pos_limits.torch[env_ids]
        fall_joint_pos.clamp_(joint_limits[..., 0], joint_limits[..., 1])

        # Interpolate: (1-alpha)*reference + alpha*fall
        new_pos = (1.0 - alpha) * ref_state.base_pos_w + alpha * fall_pos
        # Batch-compatible quaternion interpolation (nlerp: linear interp + normalize)
        tau = alpha.squeeze(-1).unsqueeze(-1)  # (N, 1)
        q1, q2 = ref_state.base_quat_w, fall_quat
        # Ensure shortest path
        dot = (q1 * q2).sum(dim=-1, keepdim=True)
        q2 = torch.where(dot < 0, -q2, q2)
        new_quat = (1.0 - tau) * q1 + tau * q2
        new_quat = new_quat / new_quat.norm(dim=-1, keepdim=True)
        self.asset.write_root_pose_to_sim_index(root_pose=torch.cat([new_pos, new_quat], dim=-1), env_ids=env_ids)

        new_lin = (1.0 - alpha) * ref_state.base_lin_vel_w + alpha * fall_lin_vel
        new_ang = (1.0 - alpha) * ref_state.base_ang_vel_w + alpha * fall_ang_vel
        self.asset.write_root_velocity_to_sim_index(
            root_velocity=torch.cat([new_lin, new_ang], dim=-1), env_ids=env_ids
        )

        new_jpos = (1.0 - alpha) * ref_state.joint_pos + alpha * fall_joint_pos
        new_jvel = (1.0 - alpha) * ref_state.joint_vel + alpha * fall_joint_vel
        self.asset.write_joint_state_to_sim_index(position=new_jpos, velocity=new_jvel, env_ids=env_ids)
