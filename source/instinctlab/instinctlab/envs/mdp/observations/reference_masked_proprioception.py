from __future__ import annotations

import torch
from prettytable import PrettyTable
from typing import TYPE_CHECKING, Sequence

import isaaclab.utils.math as math_utils
from isaaclab.envs.mdp import joint_pos, joint_pos_rel
from isaaclab.managers import ManagerTermBase, ObservationTermCfg, SceneEntityCfg

import instinctlab.utils.math as instinct_math

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedEnv

    from instinctlab.motion_reference.motion_reference_manager import MotionReferenceManagerBase


def joint_pos_reference_masked(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    reference_cfg: SceneEntityCfg = SceneEntityCfg("motion_reference"),
) -> torch.Tensor:
    """Mask the robot joint positions based on the reference frame mask."""

    # get the joint pos from isaaclab default mdp
    joint_pos_ = joint_pos(env, asset_cfg)

    # get the mask from the reference frame
    motion_reference: MotionReferenceManagerBase = env.scene[reference_cfg.name]
    mask = motion_reference.reference_frame.joint_pos_mask[:, 0, asset_cfg.joint_ids]

    return joint_pos_ * mask


def joint_pos_rel_reference_masked(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    reference_cfg: SceneEntityCfg = SceneEntityCfg("motion_reference"),
) -> torch.Tensor:
    """Mask the robot joint positions based on the reference frame mask."""

    # get the joint pos from isaaclab default mdp
    joint_pos_ = joint_pos_rel(env, asset_cfg)

    # get the mask from the reference frame
    motion_reference: MotionReferenceManagerBase = env.scene[reference_cfg.name]
    mask = motion_reference.reference_frame.joint_pos_mask[:, 0, asset_cfg.joint_ids]

    return joint_pos_ * mask


class link_pos_reference_masked(ManagerTermBase):
    """Get link position in the robot base frame, but masked by the reference frame mask."""

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.reference_cfg: SceneEntityCfg = cfg.params.get("reference_cfg", SceneEntityCfg("motion_reference"))
        self.motion_reference: MotionReferenceManagerBase = env.scene[self.reference_cfg.name]
        self.asset_cfg: SceneEntityCfg = cfg.params.get("asset_cfg", SceneEntityCfg("robot"))
        self.asset: Articulation = env.scene[self.asset_cfg.name]

        # check body_ids in asset and in motion_reference
        self.asset_link_names = self.asset_cfg.body_names
        self.reference_link_names = self.motion_reference.cfg.link_of_interests[self.reference_cfg.body_ids]
        # print a pretty table showing the link names
        table = PrettyTable()
        table.title = "link_pos_reference_masked checking link names"
        table.field_names = ["Asset Link Names", "Reference Link Names"]
        for name_i, name_j in zip(self.asset_link_names, self.reference_link_names):
            table.add_row([name_i, name_j])
        print(table)
        # check if the link names are the same
        for name_i, name_j in zip(self.asset_link_names, self.reference_link_names):
            if name_i != name_j:
                raise ValueError(f"Link names do not match: {name_i} != {name_j}")

    def __call__(
        self,
        env: ManagerBasedEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        reference_cfg: SceneEntityCfg = SceneEntityCfg("motion_reference"),
        in_base_frame: bool = True,
    ) -> torch.Tensor:
        """The link positions in the robot base frame.
        Returns:
            (num_envs, num_links, 3)
        """
        # obtain the link position w.r.t the robot base
        link_pos_w = self.asset.data.body_link_pos_w.torch[:, self.asset_cfg.body_ids]
        if in_base_frame:
            link_pos = math_utils.transform_points(
                link_pos_w,
                *math_utils.subtract_frame_transforms(
                    self.asset.data.root_link_pos_w.torch,
                    self.asset.data.root_link_quat_w.torch,
                ),
            )
        else:
            link_pos = link_pos_w
        # obtain the link reference position
        reference_link_mask = self.motion_reference.reference_frame.link_pos_mask[:, 0]
        reference_link_mask = reference_link_mask[:, self.reference_cfg.body_ids]  # (num_envs, num_bodies)

        return link_pos * reference_link_mask.unsqueeze(-1)  # (num_envs, num_links, 3)
