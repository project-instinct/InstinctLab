from __future__ import annotations

import inspect
import os
import pickle as pkl
import yaml
from collections.abc import Sequence
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils

from instinctlab.motion_reference import MotionReferenceData, MotionReferenceState, MotionSequence
from instinctlab.motion_reference.motion_buffer import MotionBuffer
from instinctlab.motion_reference.utils import estimate_angular_velocity, estimate_velocity

from .amass_motion import AmassMotion

if TYPE_CHECKING:
    from .emberUcb_cfg import EmberUcbCfg

import numpy as np
import torch
import torch.multiprocessing as mp

import pytorch_kinematics as pk


class EmberUcb(AmassMotion):
    cfg: EmberUcbCfg

    def _read_motion_file(self, motion_file_idx: int) -> MotionSequence | None:
        # Force not using velocity estimation
        self.cfg.velocity_estimation_method = None

        filepath = self._all_motion_files[motion_file_idx]
        motion_data = np.load(filepath)

        framerate = motion_data["fps"].item()
        joint_names = motion_data["dof_names"].tolist()
        joint_pos = torch.as_tensor(motion_data["dof_positions"].astype(np.float32))  # (N, num_joints)
        joint_vel = torch.as_tensor(motion_data["dof_velocities"].astype(np.float32))  # (N, num_joints)

        retargetted_joints_to_output_joints_ids = [joint_names.index(j_name) for j_name in self.isaac_joint_names]
        joint_pos = joint_pos[:, retargetted_joints_to_output_joints_ids]
        joint_vel = joint_vel[:, retargetted_joints_to_output_joints_ids]

        body_names = motion_data["body_names"].tolist()
        all_body_positions = torch.as_tensor(motion_data["body_positions"].astype(np.float32))  # (N, num_bodies, 3)
        all_body_rotations_wxyz = torch.as_tensor(
            motion_data["body_rotations"].astype(np.float32)
        )  # (N, num_bodies, 4) wxyz
        all_body_rotations = math_utils.convert_quat(all_body_rotations_wxyz, to="xyzw")
        all_body_lin_vel = torch.as_tensor(
            motion_data["body_linear_velocities"].astype(np.float32)
        )  # (N, num_bodies, 3)
        all_body_ang_vel = torch.as_tensor(
            motion_data["body_angular_velocities"].astype(np.float32)
        )  # (N, num_bodies, 3)

        key_link_names = self.link_of_interests
        retargetted_link_to_output_link_ids = [body_names.index(l_name) for l_name in key_link_names]
        link_pos_w = all_body_positions[:, retargetted_link_to_output_link_ids, :]  # (N, num_key_links, 3)
        link_rot_w = all_body_rotations[:, retargetted_link_to_output_link_ids, :]  # (N, num_key_links, 4) xyzw
        link_lin_vel_w = all_body_lin_vel[:, retargetted_link_to_output_link_ids, :]  # (N, num_key_links, 3)
        link_ang_vel_w = all_body_ang_vel[:, retargetted_link_to_output_link_ids, :]  # (N, num_key_links, 3)

        base_pos_w = all_body_positions[:, body_names.index(self.cfg.base_link_name), :]  # (N, 3)
        base_rot_w = all_body_rotations[:, body_names.index(self.cfg.base_link_name), :]  # (N, 4) xyzw
        base_lin_vel_w = all_body_lin_vel[:, body_names.index(self.cfg.base_link_name), :]  # (N, 3)
        base_ang_vel_w = all_body_ang_vel[:, body_names.index(self.cfg.base_link_name), :]  # (N, 3)

        # compute the link pos/rot/vel in base frame represented in the base frame
        base_rot_w_conj = math_utils.quat_conjugate(base_rot_w)
        link_pos_b = math_utils.quat_apply(base_rot_w_conj[:, None, :], link_pos_w - base_pos_w[:, None, :])
        link_rot_b = math_utils.quat_mul(base_rot_w_conj[:, None, :].expand(-1, link_rot_w.shape[1], -1), link_rot_w)
        link_lin_vel_b = math_utils.quat_apply(
            base_rot_w_conj[:, None, :], link_lin_vel_w - base_lin_vel_w[:, None, :]
        ) - torch.cross(link_ang_vel_w, link_pos_b)
        link_ang_vel_b = math_utils.quat_apply(base_rot_w_conj[:, None, :], link_ang_vel_w)

        return MotionSequence(
            joint_pos=joint_pos.to(device=self.buffer_device),
            joint_vel=joint_vel.to(device=self.buffer_device),
            base_pos_w=base_pos_w.to(device=self.buffer_device),
            base_lin_vel_w=base_lin_vel_w.to(device=self.buffer_device),
            base_quat_w=base_rot_w.to(device=self.buffer_device),
            base_ang_vel_w=base_ang_vel_w.to(device=self.buffer_device),
            link_pos_b=link_pos_b.to(device=self.buffer_device),
            link_quat_b=link_rot_b.to(device=self.buffer_device),
            link_pos_w=link_pos_w.to(device=self.buffer_device),
            link_quat_w=link_rot_w.to(device=self.buffer_device),
            link_lin_vel_b=link_lin_vel_b.to(device=self.buffer_device),
            link_ang_vel_b=link_ang_vel_b.to(device=self.buffer_device),
            link_lin_vel_w=link_lin_vel_w.to(device=self.buffer_device),
            link_ang_vel_w=link_ang_vel_w.to(device=self.buffer_device),
            framerate=torch.as_tensor(framerate),
            buffer_length=torch.as_tensor(joint_pos.shape[0]),
        )
