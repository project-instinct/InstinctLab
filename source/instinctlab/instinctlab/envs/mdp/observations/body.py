from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Sequence

import isaaclab.utils.math as math_utils
from isaaclab.managers import ManagerTermBase, SceneEntityCfg

import instinctlab.utils.math as instinct_math

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.managers import ObservationTermCfg
    from instinctlab.motion_reference import MotionReferenceManager


class base_pos_offset_since_motion_refresh(ManagerTermBase):
    """Compute short-term/local base position offset since the last motion reference refresh. It is a bit more local
    than get base_pos_w directly. But not good for long-term tracking and for policy observation.
    """

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        reference_cfg = cfg.params.get("reference_cfg", SceneEntityCfg("motion_reference"))
        self.motion_reference = env.scene[reference_cfg.name]
        asset_cfg = cfg.params.get("asset_cfg", SceneEntityCfg("robot"))
        self.asset = env.scene[asset_cfg.name]

        self.base_pos_marker = torch.zeros_like(self.asset.data.root_pos_w)  # (num_envs, 3)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self.base_pos_marker[env_ids] = self.asset.data.root_pos_w[env_ids]

    def __call__(
        self,
        env: ManagerBasedEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        reference_cfg: SceneEntityCfg = SceneEntityCfg("motion_reference"),
    ) -> torch.Tensor:
        landmarker_refresh_mask = self.motion_reference.time_passed_from_update < env.step_dt
        # (num_envs, 3)
        self.base_pos_marker[landmarker_refresh_mask] = self.asset.data.root_pos_w[landmarker_refresh_mask]
        # (num_envs, 3)
        base_pos_offset = self.asset.data.root_pos_w - self.base_pos_marker
        return base_pos_offset  # (num_envs, 3)


def base_heading_w(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """The heading direction of the robot base in world frame.
    Returns:
        (num_envs, 1)
    """
    asset: Articulation = env.scene[asset_cfg.name]
    base_heading_w = math_utils.euler_xyz_from_quat(asset.data.root_link_quat_w)[2]
    base_heading_w = math_utils.wrap_to_pi(base_heading_w)  # wrap to [-pi, pi]
    base_heading_w = base_heading_w.unsqueeze(-1)  # (num_envs, 1)
    return base_heading_w


def root_tannorm_w(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """The orientation of the root link in tangent-normal representation.
    Returns:
        (num_envs, 6)
    """
    asset: Articulation = env.scene[asset_cfg.name]
    root_quat_w = asset.data.root_link_quat_w
    root_tannorm = instinct_math.quat_to_tan_norm(root_quat_w)
    return root_tannorm


def _get_body_indexes(motion_ref: MotionReferenceManager, body_names: list[str] | None) -> list[int]:
    """Select body indices in motion reference order."""
    if body_names is None:
        return list(range(len(motion_ref.body_names)))
    name_to_idx = {name: i for i, name in enumerate(motion_ref.body_names)}
    missing = [name for name in body_names if name not in name_to_idx]
    if missing:
        raise ValueError(f"Body names not found in motion reference: {missing}")
    return [name_to_idx[name] for name in body_names]


def _heading_inv_quat(root_quat_w: torch.Tensor) -> torch.Tensor:
    """Yaw-only inverse quaternion."""
    return math_utils.quat_inv(math_utils.yaw_quat(root_quat_w))


def base_height(env: ManagerBasedEnv, command_name: str = "motion_reference") -> torch.Tensor:
    """The height of the robot root in world frame."""
    del command_name  # keep signature aligned with command-based observation terms.
    robot: Articulation = env.scene["robot"]
    return robot.data.root_pos_w[:, 2:3]


def local_body_pos(
    env: ManagerBasedEnv,
    command_name: str = "motion_reference",
    body_names: list[str] | None = None,
    anchor_body_name: str | None = None,
) -> torch.Tensor:
    """Body positions in yaw-heading frame, relative to robot root position."""
    motion_ref: MotionReferenceManager = env.scene[command_name]
    robot: Articulation = env.scene["robot"]

    body_indices = _get_body_indexes(motion_ref, body_names)
    names = [motion_ref.body_names[i] for i in body_indices]
    robot_body_ids, _ = robot.find_bodies(names, preserve_order=True)

    heading_inv = _heading_inv_quat(robot.data.root_quat_w)
    heading_inv_ext = heading_inv.unsqueeze(1).expand(-1, len(robot_body_ids), -1)
    rel_pos_w = robot.data.body_link_pos_w[:, robot_body_ids] - robot.data.root_pos_w.unsqueeze(1)
    local_pos = math_utils.quat_apply(heading_inv_ext, rel_pos_w)

    if anchor_body_name is None:
        anchor_body_name = motion_ref.body_names[0] if motion_ref.body_names else None
    if anchor_body_name is not None and anchor_body_name in names:
        anchor_i = names.index(anchor_body_name)
        keep_mask = torch.ones(len(names), dtype=torch.bool, device=local_pos.device)
        keep_mask[anchor_i] = False
        local_pos = local_pos[:, keep_mask, :]

    return local_pos.reshape(env.num_envs, -1)


def local_body_rot(
    env: ManagerBasedEnv,
    command_name: str = "motion_reference",
    body_names: list[str] | None = None,
) -> torch.Tensor:
    """Body rotations in yaw-heading frame, in tangent-normal representation."""
    motion_ref: MotionReferenceManager = env.scene[command_name]
    robot: Articulation = env.scene["robot"]

    body_indices = _get_body_indexes(motion_ref, body_names)
    names = [motion_ref.body_names[i] for i in body_indices]
    robot_body_ids, _ = robot.find_bodies(names, preserve_order=True)

    heading_inv = _heading_inv_quat(robot.data.root_quat_w)
    heading_inv_ext = heading_inv.unsqueeze(1).expand(-1, len(robot_body_ids), -1)
    local_quat = math_utils.quat_mul(heading_inv_ext, robot.data.body_link_quat_w[:, robot_body_ids])
    return instinct_math.quat_to_tan_norm(local_quat).reshape(env.num_envs, -1)


def local_body_vel(
    env: ManagerBasedEnv,
    command_name: str = "motion_reference",
    body_names: list[str] | None = None,
) -> torch.Tensor:
    """Body linear velocities in yaw-heading frame."""
    motion_ref: MotionReferenceManager = env.scene[command_name]
    robot: Articulation = env.scene["robot"]

    body_indices = _get_body_indexes(motion_ref, body_names)
    names = [motion_ref.body_names[i] for i in body_indices]
    robot_body_ids, _ = robot.find_bodies(names, preserve_order=True)

    heading_inv = _heading_inv_quat(robot.data.root_quat_w)
    heading_inv_ext = heading_inv.unsqueeze(1).expand(-1, len(robot_body_ids), -1)
    local_vel = math_utils.quat_apply(heading_inv_ext, robot.data.body_lin_vel_w[:, robot_body_ids])
    return local_vel.reshape(env.num_envs, -1)


def local_body_ang_vel(
    env: ManagerBasedEnv,
    command_name: str = "motion_reference",
    body_names: list[str] | None = None,
) -> torch.Tensor:
    """Body angular velocities in yaw-heading frame."""
    motion_ref: MotionReferenceManager = env.scene[command_name]
    robot: Articulation = env.scene["robot"]

    body_indices = _get_body_indexes(motion_ref, body_names)
    names = [motion_ref.body_names[i] for i in body_indices]
    robot_body_ids, _ = robot.find_bodies(names, preserve_order=True)

    heading_inv = _heading_inv_quat(robot.data.root_quat_w)
    heading_inv_ext = heading_inv.unsqueeze(1).expand(-1, len(robot_body_ids), -1)
    local_ang_vel = math_utils.quat_apply(heading_inv_ext, robot.data.body_ang_vel_w[:, robot_body_ids])
    return local_ang_vel.reshape(env.num_envs, -1)


def link_pos_b(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), in_base_frame: bool = True
) -> torch.Tensor:
    """The link positions in the robot base frame.
    Returns:
        (num_envs, num_links, 3)
    """
    asset: Articulation = env.scene[asset_cfg.name]
    link_pos_w = asset.data.body_link_pos_w[:, asset_cfg.body_ids]
    if in_base_frame:
        link_pos = math_utils.transform_points(
            link_pos_w,
            *math_utils.subtract_frame_transforms(
                asset.data.root_link_pos_w,
                asset.data.root_link_quat_w,
            ),
        )
    else:
        link_pos = link_pos_w
    return link_pos


def link_quat_b(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), in_base_frame: bool = True
) -> torch.Tensor:
    """The link orientations in the robot base frame.
    Returns:
        (num_envs, num_links, 4)
    """
    asset: Articulation = env.scene[asset_cfg.name]
    link_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids]
    if in_base_frame:
        link_quat = math_utils.quat_mul(
            math_utils.quat_inv(asset.data.root_link_quat_w).unsqueeze(1).expand(-1, link_quat_w.shape[1], -1),
            link_quat_w,
        )
    else:
        link_quat = link_quat_w
    return link_quat


def link_tannorm_b(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), in_base_frame: bool = True
) -> torch.Tensor:
    """The link orientations in tangent-normal representation in the robot base frame.
    Returns:
        (num_envs, num_links, 6)
    """
    asset: Articulation = env.scene[asset_cfg.name]
    link_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids]
    if in_base_frame:
        link_quat = math_utils.quat_mul(
            math_utils.quat_inv(asset.data.root_link_quat_w).unsqueeze(1).expand(-1, link_quat_w.shape[1], -1),
            link_quat_w,
        )
    else:
        link_quat = link_quat_w
    link_tannorm = instinct_math.quat_to_tan_norm(link_quat)
    return link_tannorm
