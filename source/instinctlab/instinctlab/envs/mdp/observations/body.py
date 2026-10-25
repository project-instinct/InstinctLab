from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.managers import ManagerTermBase, SceneEntityCfg

import instinctlab.utils.math as instinct_math

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.managers import ObservationTermCfg


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

        self.base_pos_marker = torch.zeros_like(self.asset.data.root_pos_w.torch)  # (num_envs, 3)

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        self.base_pos_marker[env_ids] = self.asset.data.root_pos_w.torch[env_ids]

    def __call__(
        self,
        env: ManagerBasedEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        reference_cfg: SceneEntityCfg = SceneEntityCfg("motion_reference"),
    ) -> torch.Tensor:
        landmarker_refresh_mask = self.motion_reference.time_passed_from_update < env.step_dt
        # (num_envs, 3)
        self.base_pos_marker[landmarker_refresh_mask] = self.asset.data.root_pos_w.torch[landmarker_refresh_mask]
        # (num_envs, 3)
        base_pos_offset = self.asset.data.root_pos_w.torch - self.base_pos_marker
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
    base_heading_w = math_utils.euler_xyz_from_quat(asset.data.root_link_quat_w.torch)[2]
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
    root_quat_w = asset.data.root_link_quat_w.torch
    root_tannorm = instinct_math.quat_to_tan_norm(root_quat_w)
    return root_tannorm


def link_pos_b(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), in_base_frame: bool = True
) -> torch.Tensor:
    """The link positions in the robot base frame.
    Returns:
        (num_envs, num_links, 3)
    """
    asset: Articulation = env.scene[asset_cfg.name]
    link_pos_w = asset.data.body_link_pos_w.torch[:, asset_cfg.body_ids]
    if in_base_frame:
        link_pos = math_utils.transform_points(
            link_pos_w,
            *math_utils.subtract_frame_transforms(
                asset.data.root_link_pos_w.torch,
                asset.data.root_link_quat_w.torch,
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
    link_quat_w = asset.data.body_link_quat_w.torch[:, asset_cfg.body_ids]
    if in_base_frame:
        link_quat = math_utils.quat_mul(
            math_utils.quat_inv(asset.data.root_link_quat_w.torch).unsqueeze(1).expand(-1, link_quat_w.shape[1], -1),
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
    link_quat_w = asset.data.body_link_quat_w.torch[:, asset_cfg.body_ids]
    if in_base_frame:
        link_quat = math_utils.quat_mul(
            math_utils.quat_inv(asset.data.root_link_quat_w.torch).unsqueeze(1).expand(-1, link_quat_w.shape[1], -1),
            link_quat_w,
        )
    else:
        link_quat = link_quat_w
    link_tannorm = instinct_math.quat_to_tan_norm(link_quat)
    return link_tannorm
