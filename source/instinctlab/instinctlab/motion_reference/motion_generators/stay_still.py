from __future__ import annotations

import inspect
import os
import yaml
from collections.abc import Sequence
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils

from instinctlab.motion_reference import MotionReferenceData, MotionReferenceState
from instinctlab.motion_reference.motion_buffer import MotionBuffer

if TYPE_CHECKING:
    from .stay_still_cfg import StayStillMotionCfg

import numpy as np
import torch


class StayStillMotion(MotionBuffer):
    """Fill the motion buffer to tell the robot to stay still."""

    cfg: StayStillMotionCfg

    generative: bool = True

    def __init__(
        self,
        cfg: StayStillMotionCfg,
        *args,
        **kwargs,
    ):
        super().__init__(cfg, *args, **kwargs)

        # (num_envs, num_joints)
        joint_pos = self.articulation_access.get_dof_positions()
        self.rest_joint_pos = torch.zeros_like(joint_pos)
        self.rest_base_pose = torch.zeros(self.rest_joint_pos.shape[0], 7, device=self.device)

    @property
    def num_trajectories(self) -> int:
        return self.cfg.pseudo_num_trajectories

    def reset(self, env_ids: Sequence[int] | torch.Tensor, symmetric_augmentation_mask_buffer: torch.Tensor) -> None:
        pass

    def fill_init_reference_state(
        self,
        env_ids: Sequence[int] | torch.Tensor,
        env_origins: torch.Tensor,
        state_buffer: MotionReferenceState,
    ) -> None:

        # temporary hacking to see if the code works
        state_buffer.joint_pos[env_ids] = self.rest_joint_pos[env_ids]
        state_buffer.joint_vel[env_ids] = 0.0
        state_buffer.base_pos_w[env_ids] = env_origins[env_ids]
        state_buffer.base_pos_w[env_ids, 2] += self.cfg.resting_pose_spawn_height_offset
        state_buffer.base_quat_w[env_ids] = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device).expand(
            len(env_ids), -1
        )
        state_buffer.base_lin_vel_w[env_ids] = 0.0
        state_buffer.base_ang_vel_w[env_ids] = 0.0

    def fill_motion_data(
        self,
        env_ids: torch.Tensor,
        sample_timestamp: torch.Tensor,
        env_origins: torch.Tensor,
        data_buffer: MotionReferenceData,
    ) -> None:
        """Fill the motion data buffer with the stay still motion."""

        joint_pos = self.articulation_access.get_dof_positions().clone()  # (num_envs, num_dofs)
        joint_pos = joint_pos[env_ids]

        base_pose_w = self.articulation_access.get_root_transforms().clone()[:, :7]
        _base_pos_w = base_pose_w[env_ids][:, :3]
        _base_quat_w = base_pose_w[env_ids][:, 3:7]
        base_pose_w = torch.cat((_base_pos_w, _base_quat_w), dim=-1)

        # determine whether to store the current state as the resting pose
        if isinstance(self.cfg.mark_base_rest_pose_time, (list, tuple)):
            store_sample_pose_mask = (
                sample_timestamp[:, 0] > self.cfg.mark_base_rest_pose_time[0]
                and sample_timestamp[:, 0] < self.cfg.mark_base_rest_pose_time[1]
            )
        else:
            store_sample_pose_mask = sample_timestamp[:, 0] < self.cfg.mark_base_rest_pose_time
        self.rest_joint_pos[env_ids[store_sample_pose_mask]] = joint_pos[store_sample_pose_mask]
        self.rest_base_pose[env_ids[store_sample_pose_mask]] = base_pose_w[store_sample_pose_mask]

        # determine whether to use current robot state or the resting pose as the motion reference
        if isinstance(self.cfg.mark_base_rest_pose_time, (list, tuple)):
            use_rest_pose_mask = sample_timestamp[:, 0] > self.cfg.mark_base_rest_pose_time[1]
        else:
            use_rest_pose_mask = sample_timestamp[:, 0] > self.cfg.mark_base_rest_pose_time
        joint_pos[use_rest_pose_mask] = self.rest_joint_pos[env_ids[use_rest_pose_mask]]
        base_pose_w[use_rest_pose_mask] = self.rest_base_pose[env_ids[use_rest_pose_mask]]

        base_pos_w, base_quat_w = base_pose_w[:, :3], base_pose_w[:, 3:7]

        # can be replaced with all zeros
        link_pos_quat_b = self.forward_kinematics_func(joint_pos)
        link_pos_b = link_pos_quat_b[..., :3]
        link_quat_b = link_pos_quat_b[..., 3:]
        link_pos_w, link_quat_w = math_utils.combine_frame_transforms(
            base_pos_w.unsqueeze(1).expand(-1, link_pos_b.shape[1], -1),
            base_quat_w.unsqueeze(1).expand(-1, link_pos_b.shape[1], -1),
            link_pos_b,
            link_quat_b,
        )

        n_frames = data_buffer.joint_pos.shape[1]
        data_buffer.joint_pos[env_ids] = joint_pos.unsqueeze(1).expand(-1, n_frames, -1)
        data_buffer.joint_vel[env_ids] = 0.0
        data_buffer.validity[env_ids] = 1.0
        data_buffer.base_pos_w[env_ids] = base_pos_w.unsqueeze(1).expand(-1, n_frames, -1)
        data_buffer.base_quat_w[env_ids] = base_quat_w.unsqueeze(1).expand(-1, n_frames, -1)
        data_buffer.link_pos_b[env_ids] = link_pos_b.unsqueeze(1).expand(-1, n_frames, -1, -1)
        data_buffer.link_quat_b[env_ids] = link_quat_b.unsqueeze(1).expand(-1, n_frames, -1, -1)
        data_buffer.link_pos_w[env_ids] = link_pos_w.unsqueeze(1).expand(-1, n_frames, -1, -1)
        data_buffer.link_quat_w[env_ids] = link_quat_w.unsqueeze(1).expand(-1, n_frames, -1, -1)
