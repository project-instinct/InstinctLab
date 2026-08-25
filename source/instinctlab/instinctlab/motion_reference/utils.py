""" Some utilities to compute the statistics of the robot w.r.t the reference data. """

from __future__ import annotations

import numpy as np
import torch
from typing import TYPE_CHECKING, Literal

import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg

from instinctlab.utils.math import quat_angular_velocity, quat_slerp_batch

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.envs import ManagerBasedEnv

    from .motion_reference_manager import MotionReferenceManagerBase


def get_base_position_distance(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    reference_cfg: SceneEntityCfg = SceneEntityCfg("motion_reference"),
    masking: bool = True,
    squared: bool = False,
    return_diff: bool = False,
) -> torch.Tensor:
    """
    Args:
        masking: whether to use the reference data's position mask in computing distance
        return_diff: if True, return the difference (3d) rather than the distance (scalar)
    Returns:
        diff/distance:
        (batch_size, 3) if return_diff is True
        (batch_size,) if return_diff is False
    """
    # extract useful elements
    asset: RigidObject = env.scene[asset_cfg.name]
    motion_reference: MotionReferenceManagerBase = env.scene[reference_cfg.name]

    # obtain the base position of the robot
    base_pos = asset.data.root_pos_w.torch
    # obtain the reference position
    ref_pos = motion_reference.data.base_pos_w
    ref_pos = ref_pos[motion_reference.ALL_INDICES, motion_reference.aiming_frame_idx]

    # calculate the distance between the base position and the reference position
    diff = base_pos - ref_pos  # (batch_size, 3)
    if masking:
        diff[:, :2] *= motion_reference.data.base_pos_plane_mask[
            motion_reference.ALL_INDICES, motion_reference.aiming_frame_idx
        ].unsqueeze(1)
        diff[:, 2] *= motion_reference.data.base_pos_height_mask[
            motion_reference.ALL_INDICES, motion_reference.aiming_frame_idx
        ]
    if return_diff:
        return diff

    if squared:
        distance = torch.sum(torch.square(diff), dim=-1)
    else:
        distance = torch.norm(diff, dim=-1)  # (batch_size,)
    return distance


def get_base_rotation_distance(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    reference_cfg: SceneEntityCfg = SceneEntityCfg("motion_reference"),
    difference_mode: Literal["box_minus", "axis_angle"] = "axis_angle",
    masking: bool = True,
) -> torch.Tensor:
    """
    Args:
        masking: whether to use the reference data's rotation mask in computing distance
    Returns:
        (batch_size,) if return_diff is False
    """
    # extract useful elements
    asset: RigidObject = env.scene[asset_cfg.name]
    motion_reference: MotionReferenceManagerBase = env.scene[reference_cfg.name]

    # obtain the base rotation of the robot
    base_rot = asset.data.root_quat_w.torch
    # obtain the reference rotation
    ref_rot = motion_reference.data.base_quat_w[motion_reference.ALL_INDICES, motion_reference.aiming_frame_idx]

    # calculate the difference between the base rotation and the reference rotation
    if difference_mode == "box_minus":
        diff = torch.norm(
            math_utils.quat_box_minus(base_rot.reshape(-1, 4), ref_rot.reshape(-1, 4)).reshape(-1, 3), dim=-1
        )  # (batch_size,)
    elif difference_mode == "axis_angle":
        diff = math_utils.quat_error_magnitude(base_rot.reshape(-1, 4), ref_rot.reshape(-1, 4))  # (batch_size,)

    if masking:
        diff *= motion_reference.data.validity[motion_reference.ALL_INDICES, motion_reference.aiming_frame_idx]

    return diff  # (batch_size,)


def get_base_velocity_difference(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    reference_cfg: SceneEntityCfg = SceneEntityCfg("motion_reference"),
    masking: bool = True,
    anchor_frame: Literal["world", "robot", "reference"] = "world",
    return_diff: bool = False,
) -> torch.Tensor:
    """
    Args:
        masking: whether to use the reference data's velocity mask in computing distance
        anchor_frame: the frame to calculate the velocity difference, can be "world", "robot" or "reference"
    Returns:
        velocity_diff: (batch_size, 3) if return_diff is True
    """
    # extract useful elements
    asset: RigidObject = env.scene[asset_cfg.name]
    motion_reference: MotionReferenceManagerBase = env.scene[reference_cfg.name]

    if anchor_frame == "world":
        # obtain the base velocity of the robot in world frame
        base_vel = asset.data.root_lin_vel_w.torch
        ref_base_vel = motion_reference.data.base_lin_vel_w[
            motion_reference.ALL_INDICES, motion_reference.aiming_frame_idx
        ]
    elif anchor_frame == "robot":
        # obtain the base velocity of the robot in robot frame
        base_vel = asset.data.root_lin_vel_b.torch
        anchor_quat = asset.data.root_quat_w.torch
        ref_base_vel = math_utils.quat_apply_inverse(
            anchor_quat,
            motion_reference.data.base_lin_vel_w[motion_reference.ALL_INDICES, motion_reference.aiming_frame_idx],
        )
    elif anchor_frame == "reference":
        # obtain the base velocity of the robot in reference frame
        anchor_quat = motion_reference.data.base_quat_w[motion_reference.ALL_INDICES, motion_reference.aiming_frame_idx]
        base_vel = math_utils.quat_apply_inverse(anchor_quat, asset.data.root_lin_vel_w.torch)
        ref_base_vel = math_utils.quat_apply_inverse(
            anchor_quat,
            motion_reference.data.base_lin_vel_w[motion_reference.ALL_INDICES, motion_reference.aiming_frame_idx],
        )
    else:
        raise ValueError(f"Unknown anchor frame: {anchor_frame}. Must be one of 'world', 'robot', or 'reference'.")
    # shape (batch_size, 3)

    # calculate the distance between the base velocity and the reference base velocity
    vel_diff = base_vel - ref_base_vel  # (batch_size, 3)
    if masking:
        vel_diff *= motion_reference.data.validity[
            motion_reference.ALL_INDICES, motion_reference.aiming_frame_idx
        ].unsqueeze(1)
    if return_diff:
        return vel_diff  # (batch_size, 3)

    return torch.norm(vel_diff, dim=-1)  # (batch_size,)


def get_joint_position_difference(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    reference_cfg: SceneEntityCfg = SceneEntityCfg("motion_reference"),
    masking: bool = True,
) -> torch.Tensor:
    # extract useful elements
    asset: Articulation = env.scene[asset_cfg.name]
    motion_reference: MotionReferenceManagerBase = env.scene[reference_cfg.name]

    # obtain the joint position of the robot
    joint_pos = asset.data.joint_pos.torch
    # obtain the reference joint position
    ref_joint_pos = motion_reference.data.joint_pos
    ref_joint_pos = ref_joint_pos[motion_reference.ALL_INDICES, motion_reference.aiming_frame_idx]

    # calculate the distance between the joint position and the reference joint position
    joint_diff = joint_pos - ref_joint_pos  # (batch_size, num_joints)
    if masking:
        joint_diff *= motion_reference.data.joint_pos_mask[
            motion_reference.ALL_INDICES, motion_reference.aiming_frame_idx
        ]
    joint_diff = joint_diff[:, reference_cfg.joint_ids]
    return joint_diff  # (batch_size, num_joints)


def get_joint_velocity_difference(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    reference_cfg: SceneEntityCfg = SceneEntityCfg("motion_reference"),
    masking: bool = True,
) -> torch.Tensor:
    # extract useful elements
    asset: Articulation = env.scene[asset_cfg.name]
    motion_reference: MotionReferenceManagerBase = env.scene[reference_cfg.name]

    # obtain the joint velocity of the robot
    joint_vel = asset.data.joint_vel.torch
    # obtain the reference joint velocity
    ref_joint_vel = motion_reference.data.joint_vel
    ref_joint_vel = ref_joint_vel[motion_reference.ALL_INDICES, motion_reference.aiming_frame_idx]

    # calculate the distance between the joint velocity and the reference joint velocity
    joint_diff = joint_vel - ref_joint_vel  # (batch_size, num_joints)
    if masking:
        joint_diff *= motion_reference.data.joint_vel_mask[
            motion_reference.ALL_INDICES, motion_reference.aiming_frame_idx
        ]
        joint_diff *= motion_reference.data.validity[
            motion_reference.ALL_INDICES, motion_reference.aiming_frame_idx
        ].unsqueeze(1)
    joint_diff = joint_diff[:, reference_cfg.joint_ids]
    if torch.isnan(joint_diff).any():
        raise ValueError("Joint velocity difference contains NaN values.")
    return joint_diff  # (batch_size, num_joints)


def get_link_position_distance(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    reference_cfg: SceneEntityCfg = SceneEntityCfg("motion_reference"),
    in_base_frame: bool = False,
    masking: bool = True,
    squared: bool = False,
    return_diff: bool = False,
) -> torch.Tensor:
    # extract useful elements
    asset: Articulation = env.scene[asset_cfg.name]
    motion_reference: MotionReferenceManagerBase = env.scene[reference_cfg.name]

    # obtain the link position w.r.t the robot base
    links = asset.find_bodies(motion_reference.cfg.link_of_interests, preserve_order=True)
    link_indices = links[0]
    links_pos_w = asset.data.body_link_pos_w.torch[:, link_indices]  # (batch_size, num_links, 3)
    root_pos_w = asset.data.root_pos_w.torch
    root_quat_w = asset.data.root_quat_w.torch
    root_pos_w_inv, root_quat_w_inv = math_utils.subtract_frame_transforms(root_pos_w, root_quat_w)
    if in_base_frame:
        links_pos = math_utils.transform_points(
            links_pos_w,
            root_pos_w_inv,
            root_quat_w_inv,
        )
        # obtain the link reference position
        ref_links_pos = motion_reference.data.link_pos_b[
            motion_reference.ALL_INDICES,
            motion_reference.aiming_frame_idx,
        ]
    else:
        links_pos = links_pos_w
        ref_links_pos = motion_reference.data.link_pos_w[
            motion_reference.ALL_INDICES,
            motion_reference.aiming_frame_idx,
        ]

    # calculate the distances between link and reference link
    link_pos_diff = links_pos - ref_links_pos  # (batch_size, num_links, 3)
    if masking:
        link_pos_diff *= motion_reference.data.link_pos_mask[
            motion_reference.ALL_INDICES,
            motion_reference.aiming_frame_idx,
        ].unsqueeze(-1)
    link_pos_diff = link_pos_diff[:, reference_cfg.body_ids]  # (batch_size, num_links, 3)
    if return_diff:
        return link_pos_diff
    if squared:
        return torch.sum(torch.square(link_pos_diff), dim=-1)
    else:
        return torch.norm(link_pos_diff, dim=-1)  # (batch_size, num_links)


def get_link_rotation_distance(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    reference_cfg: SceneEntityCfg = SceneEntityCfg("motion_reference"),
    in_base_frame: bool = False,
    difference_mode: Literal["box_minus", "axis_angle"] = "axis_angle",
    masking: bool = True,
    squared: bool = False,
) -> torch.Tensor:
    # extract useful elements
    asset: Articulation = env.scene[asset_cfg.name]
    motion_reference: MotionReferenceManagerBase = env.scene[reference_cfg.name]

    # obtain the link rotation w.r.t the robot base
    links = asset.find_bodies(motion_reference.cfg.link_of_interests, preserve_order=True)
    link_indices = links[0]

    if in_base_frame:
        links_rot_w = asset.data.body_link_quat_w.torch[:, link_indices]
        root_quat_w_inv = math_utils.quat_inv(asset.data.root_quat_w.torch)
        links_quat = math_utils.quat_mul(root_quat_w_inv.unsqueeze(-2).expand(-1, len(link_indices), -1), links_rot_w)
        links_quat_ref = motion_reference.data.link_quat_b[
            motion_reference.ALL_INDICES,
            motion_reference.aiming_frame_idx,
        ]
    else:
        links_quat = asset.data.body_link_quat_w.torch[:, link_indices]
        links_quat_ref = motion_reference.data.link_quat_w[
            motion_reference.ALL_INDICES,
            motion_reference.aiming_frame_idx,
        ]
    # calculate the difference between link rotation and reference link rotation
    if difference_mode == "box_minus":
        link_rot_diff = torch.norm(
            math_utils.quat_box_minus(links_quat.reshape(-1, 4), links_quat_ref.reshape(-1, 4)).reshape(
                -1, len(link_indices), 3
            ),
            dim=-1,
        )  # (batch_size, num_links)
    elif difference_mode == "axis_angle":
        link_rot_diff = math_utils.quat_error_magnitude(
            links_quat.reshape(-1, 4), links_quat_ref.reshape(-1, 4)
        ).reshape(
            -1, len(link_indices)
        )  # (batch_size, num_links)

    if masking:
        link_rot_diff *= motion_reference.data.link_rot_mask[
            motion_reference.ALL_INDICES,
            motion_reference.aiming_frame_idx,
        ]
    link_rot_diff = link_rot_diff[:, reference_cfg.body_ids]
    if squared:
        return torch.square(link_rot_diff)  # (batch_size, num_links)
    else:
        return link_rot_diff  # (batch_size, num_links)


def matching_reference_timing(
    env: ManagerBasedEnv,
    buffer: torch.Tensor,
    motion_reference: MotionReferenceManagerBase,
    check_at_keyframe_threshold: float,
    multiply_by_frame_interval: bool,
) -> torch.Tensor:
    """Multiple the buffer by keyframe timing and (maybe) frame interval."""
    if check_at_keyframe_threshold >= 0:
        # return nonzero reward when the time_to_aiming_frame is less than the threshold
        buffer = buffer * (motion_reference.time_to_aiming_frame <= check_at_keyframe_threshold).float()

    if multiply_by_frame_interval:
        buffer *= motion_reference.frame_interval_s / env.step_dt

    # maskout the reward when the reference if all frames are done/invalid
    buffer = buffer * (motion_reference.aiming_frame_idx >= 0)

    return buffer


def motion_interpolate_bilinear(
    root_trans: torch.Tensor,
    root_quat: torch.Tensor,
    joint_pos: torch.Tensor,
    source_framerate: float,
    target_framerate: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The bilinear interpolation function for the motion data. Interface referring to `.motion_reference_cfg.MotionReferenceManagerCfg.motion_interpolate_func`."""

    # shape[0] - 1 because the time is measured by the frame interval. The 0-th frame does not count as time.
    time_max = (root_trans.shape[0] - 1) / source_framerate
    # NOTE: don't delete the frame_idx clipping, because the frame_idx might be greater than root_trans.shape[0] - 1
    num_frames_ = np.ceil(time_max * target_framerate)
    frame_idx = (
        torch.arange(num_frames_).to(dtype=torch.float, device=root_trans.device) / target_framerate * source_framerate
    )
    # Must be no greater than the last frame index, NOTE: assuming ascending order of the frame_idx
    frame_idx = frame_idx[frame_idx <= (root_trans.shape[0] - 1)]

    front_frame_idx = frame_idx.floor().long()
    back_frame_idx = frame_idx.ceil().long()
    ratio = frame_idx - front_frame_idx.float()
    _root_trans = (
        ratio.unsqueeze(-1) * root_trans[back_frame_idx] + (1 - ratio).unsqueeze(-1) * root_trans[front_frame_idx]
    )
    _root_quat = quat_slerp_batch(
        root_quat[front_frame_idx],
        root_quat[back_frame_idx],
        ratio,
    )
    _joint_pos = (
        ratio.unsqueeze(-1) * joint_pos[back_frame_idx] + (1 - ratio).unsqueeze(-1) * joint_pos[front_frame_idx]
    )

    return _root_trans, _root_quat, _joint_pos


def pose_interpolate_bilinear(
    pos: torch.Tensor,
    quat: torch.Tensor,
    validity: torch.Tensor | None,
    source_framerate: float,
    target_framerate: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """The bilinear interpolation function for generic pose data (e.g. objects).

    Why this needs an individual function:
    1. Object data often has an extra batch dimension (T, N, 3) for multiple objects, whereas robot motion is usually (T, 3).
    2. Object data includes validity flags that need nearest-neighbor interpolation, which isn't handled in standard robot motion interpolation.
    3. Separation of concerns: keeps the robot-specific kinematic interpolation separate from generic pose trajectory interpolation.

    Args:
        pos: (T, N, 3) or (T, 3) position
        quat: (T, N, 4) or (T, 4) quaternion
        validity: (T, N) or (T,) or None validity mask
        source_framerate: source data framerate
        target_framerate: target data framerate

    Returns:
        interpolated_pos, interpolated_quat, interpolated_validity
    """
    # Time indices calculation
    time_max = (pos.shape[0] - 1) / source_framerate
    num_frames_ = np.ceil(time_max * target_framerate)
    frame_idx = torch.arange(num_frames_).to(dtype=torch.float, device=pos.device) / target_framerate * source_framerate
    frame_idx = frame_idx[frame_idx <= (pos.shape[0] - 1)]

    front_frame_idx = frame_idx.floor().long()
    back_frame_idx = frame_idx.ceil().long()
    ratio = frame_idx - front_frame_idx.float()

    # Position Interpolation (Linear)
    # Expand ratio to match pos dimensions: (T, 1, ..., 1)
    ratio_pos = ratio.view(-1, *([1] * (pos.ndim - 1)))
    _pos = pos[back_frame_idx] * ratio_pos + pos[front_frame_idx] * (1 - ratio_pos)

    # Rotation Interpolation (Slerp)
    # Flatten extra dimensions for slerp: (T * ..., 4)
    q0 = quat[front_frame_idx]
    q1 = quat[back_frame_idx]
    batch_shape = q0.shape[:-1]

    # Expand ratio to match batch shape
    ratio_quat = ratio.view(-1, *([1] * (len(batch_shape) - 1))).expand(batch_shape)

    _quat = quat_slerp_batch(q0.reshape(-1, 4), q1.reshape(-1, 4), ratio_quat.reshape(-1)).reshape(*batch_shape, 4)

    # Validity Interpolation (Nearest Neighbor)
    _validity = None
    if validity is not None:
        idx_nearest = frame_idx.round().long().clamp(max=pos.shape[0] - 1)
        _validity = validity[idx_nearest]

    return _pos, _quat, _validity


def estimate_velocity(
    positions: torch.Tensor,  # (batch_size, num_frames, data_dim)
    dt: float,
    estimation_type: Literal["frontward", "backward", "frontbackward"] = "frontbackward",
):
    """Estimate the linear / joint velocity of the robot by using the position difference between two frames.
    Args:
        positions: the position data to be estimated, shape (batch_size, num_frames, data_dim)
        dt: the time interval between two frames
        estimation_type: the type of estimation, can be "frontward", "backward" or "frontbackward"
            - "frontward": estimate the velocity by using the position difference between the current frame and the next frame
            - "backward": estimate the velocity by using the position difference between the current frame and the previous frame
            - "frontbackward": estimate the velocity by using the position difference between the next frame and the previous frame
    Returns:
        velocity: the estimated velocity, shape (batch_size, num_frames, data_dim)
    """
    assert len(positions.shape) == 3, "The position data must be in shape (batch_size, num_frames, data_dim)."
    if estimation_type == "frontward":
        next_frame = torch.roll(positions, -1, dims=1)
        next_frame[:, -1] = positions[:, -1]
        prev_frame = positions
        velocity = (next_frame - prev_frame) / dt
    elif estimation_type == "backward":
        prev_frame = torch.roll(positions, 1, dims=1)
        prev_frame[:, 0] = positions[:, 0]
        next_frame = positions
        velocity = (next_frame - prev_frame) / dt
    elif estimation_type == "frontbackward":
        prev_frame = torch.roll(positions, 1, dims=1)
        prev_frame[:, 0] = positions[:, 0]
        next_frame = torch.roll(positions, -1, dims=1)
        next_frame[:, -1] = positions[:, -1]
        velocity = (next_frame - prev_frame) / (2 * dt)
    else:
        raise ValueError(f"Unknown estimation type: {estimation_type}.")

    return velocity


def estimate_angular_velocity(
    quaternions: torch.Tensor,  # (batch_size, num_frames, 4)
    dt: float,
    estimation_type: Literal["frontward", "backward", "frontbackward"] = "frontbackward",
):
    """Estimate the angular velocity of the robot by using the quaternion difference between two frames.
    Args:
        quaternions: the quaternion data to be estimated, shape (batch_size, num_frames, 4)
        dt: the time interval between two frames
        estimation_type: the type of estimation, can be "frontward", "backward" or "frontbackward"
            - "frontward": estimate the angular velocity by using the quaternion difference between the current frame and the next frame
            - "backward": estimate the angular velocity by using the quaternion difference between the current frame and the previous frame
            - "frontbackward": estimate the angular velocity by using the quaternion difference between the next frame and the previous frame
    Returns:
        angular_velocity: the estimated angular velocity, shape (batch_size, num_frames, 3)
    """
    assert quaternions.shape[-1] == 4, "The last dimension of the quaternion data must be 4."
    assert len(quaternions.shape) == 3, "The quaternion data must be in shape (batch_size, num_frames, 4)."
    if estimation_type == "frontward":
        next_frame = torch.roll(quaternions, -1, dims=1)
        next_frame[:, -1] = quaternions[:, -1]
        prev_frame = quaternions
        angular_velocity = quat_angular_velocity(
            prev_frame.reshape(-1, 4),
            next_frame.reshape(-1, 4),
            dt,
        ).reshape(-1, quaternions.shape[1], 3)
    elif estimation_type == "backward":
        prev_frame = torch.roll(quaternions, 1, dims=1)
        prev_frame[:, 0] = quaternions[:, 0]
        next_frame = quaternions
        angular_velocity = quat_angular_velocity(
            prev_frame.reshape(-1, 4),
            next_frame.reshape(-1, 4),
            dt,
        ).reshape(-1, quaternions.shape[1], 3)
    elif estimation_type == "frontbackward":
        prev_frame = torch.roll(quaternions, 1, dims=1)
        prev_frame[:, 0] = quaternions[:, 0]
        next_frame = torch.roll(quaternions, -1, dims=1)
        next_frame[:, -1] = quaternions[:, -1]
        angular_velocity = quat_angular_velocity(
            prev_frame.reshape(-1, 4),
            next_frame.reshape(-1, 4),
            2 * dt,
        ).reshape(-1, quaternions.shape[1], 3)
    else:
        raise ValueError(f"Unknown estimation type: {estimation_type}.")

    return angular_velocity
