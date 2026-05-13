from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Literal

import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.envs import ManagerBasedEnv

    from instinctlab.motion_reference import MotionReferenceData, MotionReferenceManager

"""
Sampling masks.
"""


def resample_base_plane_pos_ref_mask(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    motion_ref_cfg: SceneEntityCfg = SceneEntityCfg("motion_reference"),
    maskout_ratio: float = 0.5,
):
    motion_reference: MotionReferenceManager = env.scene[motion_ref_cfg.name]
    data: MotionReferenceData = motion_reference.data
    data.base_pos_plane_mask[env_ids] = ~(torch.rand_like(data.base_pos_plane_mask[env_ids]) < maskout_ratio)


def resample_base_height_pos_ref_mask(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    motion_ref_cfg: SceneEntityCfg = SceneEntityCfg("motion_reference"),
    maskout_ratio: float = 0.5,
):
    motion_reference: MotionReferenceManager = env.scene[motion_ref_cfg.name]
    data: MotionReferenceData = motion_reference.data
    data.base_pos_height_mask[env_ids] = ~(torch.rand_like(data.base_pos_height_mask[env_ids]) < maskout_ratio)


def resample_base_position_ref_mask(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    motion_ref_cfg: SceneEntityCfg = SceneEntityCfg("motion_reference"),
    maskout_ratio: float = 0.5,
):
    """Sample and maskout both pos_plane_mask and pos_height_mask together."""
    motion_reference: MotionReferenceManager = env.scene[motion_ref_cfg.name]
    data: MotionReferenceData = motion_reference.data
    mask = torch.rand_like(data.base_pos_plane_mask[env_ids]) < maskout_ratio
    data.base_pos_plane_mask[env_ids] = ~mask
    data.base_pos_height_mask[env_ids] = ~mask


def resample_base_orientation_ref_mask(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    motion_ref_cfg: SceneEntityCfg = SceneEntityCfg("motion_reference"),
    maskout_ratio: float = 0.5,
):
    motion_reference: MotionReferenceManager = env.scene[motion_ref_cfg.name]
    data: MotionReferenceData = motion_reference.data
    data.base_orientation_mask[env_ids] = ~(torch.rand_like(data.base_orientation_mask[env_ids]) < maskout_ratio)


def resample_base_heading_ref_mask(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    motion_ref_cfg: SceneEntityCfg = SceneEntityCfg("motion_reference"),
    maskout_ratio: float = 0.5,
):
    motion_reference: MotionReferenceManager = env.scene[motion_ref_cfg.name]
    data: MotionReferenceData = motion_reference.data
    data.base_heading_mask[env_ids] = ~(torch.rand_like(data.base_heading_mask[env_ids]) < maskout_ratio)


def resample_base_rotation_ref_mask(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    motion_ref_cfg: SceneEntityCfg = SceneEntityCfg("motion_reference"),
    maskout_ratio: float = 0.5,
):
    """Sample and maskout both orientation_mask and heading_mask together."""
    motion_reference: MotionReferenceManager = env.scene[motion_ref_cfg.name]
    data: MotionReferenceData = motion_reference.data
    mask = torch.rand_like(data.base_orientation_mask[env_ids]) < maskout_ratio
    data.base_orientation_mask[env_ids] = ~mask
    data.base_heading_mask[env_ids] = ~mask


"""
Compute Mask based on the reference data.
"""


def maskout_base_height_pos_ref_on_orientation(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    motion_ref_cfg: SceneEntityCfg = SceneEntityCfg("motion_reference"),
    maskout_threshold: float = 0.5,
):
    """Mask out the base_pos_height_mask when the reference roll/pitch is larger than the
    threshold. Recommend to use in `interval` mode and apply in each env_step.

    Args:
        maskout_threshold: the threshold in radians.
    """
    motion_reference: MotionReferenceManager = env.scene[motion_ref_cfg.name]
    data: MotionReferenceData = motion_reference.data
    leading_dims = data.base_quat_w[env_ids].shape[:-1]
    roll_ref, pitch_ref, _ = math_utils.euler_xyz_from_quat(data.base_quat_w[env_ids].reshape(-1, 4))  # (B*,)
    roll_ref = math_utils.wrap_to_pi(roll_ref).reshape(leading_dims)  # (num_envs, num_keyframes)
    pitch_ref = math_utils.wrap_to_pi(pitch_ref).reshape(leading_dims)  # (num_envs, num_keyframes)
    maskout = (torch.abs(roll_ref) > maskout_threshold) | (torch.abs(pitch_ref) > maskout_threshold)
    data.base_pos_height_mask[env_ids] = ~maskout


def maskout_base_pos_ref_on_orientation(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    motion_ref_cfg: SceneEntityCfg = SceneEntityCfg("motion_reference"),
    maskout_threshold: float = 0.5,
    maskout_prob: float = 1.0,
):
    """Mask out the base_pos_plane_mask and base_pos_height_mask at a given probability when the reference roll/pitch
    is larger than the threshold. Recommend to use in `interval` mode and apply in each env_step.
    """
    motion_reference: MotionReferenceManager = env.scene[motion_ref_cfg.name]
    data: MotionReferenceData = motion_reference.data
    leading_dims = data.base_quat_w[env_ids].shape[:-1]
    roll_ref, pitch_ref, _ = math_utils.euler_xyz_from_quat(data.base_quat_w[env_ids].reshape(-1, 4))  # (B*,)
    roll_ref = math_utils.wrap_to_pi(roll_ref).reshape(leading_dims)  # (num_envs, num_keyframes)
    pitch_ref = math_utils.wrap_to_pi(pitch_ref).reshape(leading_dims)  # (num_envs, num_keyframes)
    maskout = (torch.abs(roll_ref) > maskout_threshold) | (torch.abs(pitch_ref) > maskout_threshold)
    maskout = maskout & (torch.rand_like(roll_ref) < maskout_prob)  # (num_envs, num_keyframes)
    data.base_pos_height_mask[env_ids] = ~maskout
    data.base_pos_plane_mask[env_ids] = ~maskout


def maskout_base_plane_pos_ref_on_height(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    motion_ref_cfg: SceneEntityCfg = SceneEntityCfg("motion_reference"),
    maskout_threshold: float = 0.5,  # height threshold in meters
):
    motion_reference: MotionReferenceManager = env.scene[motion_ref_cfg.name]
    data: MotionReferenceData = motion_reference.data
    leading_dims = data.base_quat_w[env_ids].shape[:-1]
    height_ref = data.base_pos_w[env_ids, :, 2]  # (num_envs, num_keyframes)
    maskout = height_ref < maskout_threshold
    data.base_pos_plane_mask[env_ids] = ~maskout


def maskout_base_pos_ref(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    motion_ref_cfg: SceneEntityCfg = SceneEntityCfg("motion_reference"),
):
    """Mask out the base pos reference data on every motion reference."""
    motion_reference: MotionReferenceManager = env.scene[motion_ref_cfg.name]
    data: MotionReferenceData = motion_reference.data
    data.base_pos_plane_mask[env_ids] = 0
    data.base_pos_height_mask[env_ids] = 0


def maskout_joint_ref(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    motion_ref_cfg: SceneEntityCfg = SceneEntityCfg("motion_reference"),
):
    """Mask out the wrist reference data on every motion reference.

    Args:
        wrist_name_expr: the regular expression to match the wrist names.
    """
    motion_reference: MotionReferenceManager = env.scene[motion_ref_cfg.name]
    joint_ids = motion_ref_cfg.joint_ids
    # generate meshgrid for env_ids and joint_ids
    if isinstance(joint_ids, (list, tuple)):
        joint_ids = torch.tensor(joint_ids, dtype=torch.long, device=motion_reference.device)
    env_ids_, joint_ids_ = torch.meshgrid(env_ids, joint_ids)
    env_ids_ = env_ids_.reshape(-1)
    joint_ids_ = joint_ids_.reshape(-1)
    # mask out the wrist reference data
    data: MotionReferenceData = motion_reference.data
    data.joint_pos_mask[env_ids_, :, joint_ids_] = 0
    data.joint_vel_mask[env_ids_, :, joint_ids_] = 0
    # mask out the reference_frame
    reference_frame: MotionReferenceData = motion_reference.reference_frame
    reference_frame.joint_pos_mask[env_ids_, :, joint_ids_] = 0
    reference_frame.joint_vel_mask[env_ids_, :, joint_ids_] = 0


def maskout_link_ref(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    motion_ref_cfg: SceneEntityCfg = SceneEntityCfg("motion_reference"),
):
    """Mask out the link reference data on every motion reference.
    NOTE: This implementation uses motion_ref_cfg.link_ids. There is NO need to assign asset_cfg.link_names
    """
    motion_reference: MotionReferenceManager = env.scene[motion_ref_cfg.name]
    # generate meshgrid for env_ids and link_ids
    link_ids = torch.tensor(motion_ref_cfg.body_ids, dtype=torch.long, device=env_ids.device)
    env_ids_, link_ids_ = torch.meshgrid(env_ids, link_ids)
    env_ids_ = env_ids_.reshape(-1)
    link_ids_ = link_ids_.reshape(-1)
    # mask out the link reference data
    data: MotionReferenceData = motion_reference.data
    data.link_pos_mask[env_ids_, :, link_ids_] = 0
    data.link_rot_mask[env_ids_, :, link_ids_] = 0
    # mask out the reference_frame
    reference_frame: MotionReferenceData = motion_reference.reference_frame
    reference_frame.link_pos_mask[env_ids_, :, link_ids_] = 0
    reference_frame.link_rot_mask[env_ids_, :, link_ids_] = 0


def reset_robot_state_by_reference_gaussian_randomization_scale(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    motion_ref_cfg: SceneEntityCfg = SceneEntityCfg("motion_reference"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    position_offset: list = [0.0, 0.0, 0.0],
    dof_vel_ratio: float = 1.0,
    base_lin_vel_ratio: float = 1.0,
    base_ang_vel_ratio: float = 1.0,
    randomize_pose_range: dict[str, tuple[float, float]] = {},
    randomize_velocity_range: dict[str, tuple[float, float]] = {},
    randomize_joint_pos_range: tuple[float, float] = (0.0, 0.0),
    gaussian_center: float = 0.5,
    gaussian_std: float = 0.2,
    reverse_gaussian: bool = True,  # higher scale at start/end, lower scale at middle
):
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    motion_ref: MotionReferenceManager = env.scene[motion_ref_cfg.name]

    # no reset the motion reference object
    # motion reference (as sensor) is already reset(ed) in scene.reset(...)
    # motion_ref.reset(env_ids)
    start_from_middle_ratio = 1.0 - (motion_ref.assigned_motion_lengths / motion_ref.complete_motion_lengths)
    start_from_middle_ratio = start_from_middle_ratio[env_ids]
    # compute the randomization scaling function
    # gaussian distribution centered at 0.5 with std = gaussian_std
    randomization_scale = torch.exp(
        -((start_from_middle_ratio - gaussian_center) ** 2) / (2 * gaussian_std**2)
    )  # (num_envs,)
    if reverse_gaussian:
        randomization_scale = 1.0 - randomization_scale

    # reset the robot state based on motion reference data
    data: MotionReferenceData = motion_ref.data  # triggering the motion reference to update the data.
    motion_ref_init_state = motion_ref.get_init_reference_state(env_ids)

    # reset the robot state
    base_pos_w = motion_ref_init_state.base_pos_w
    base_pos_w = motion_ref_init_state.base_pos_w + torch.tensor(
        position_offset, device=motion_ref_init_state.base_pos_w.device
    ).unsqueeze(0)

    # apply randomizations on the reset states if specified
    if randomize_pose_range:
        # Apply pose randomization similar to reset_root_state_uniform
        range_list = [randomize_pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=base_pos_w.device)
        rand_samples = math_utils.sample_uniform(
            ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=base_pos_w.device
        )
        rand_samples = rand_samples * randomization_scale.unsqueeze(1)

        # Add position randomization
        base_pos_w += rand_samples[:, 0:3]

        # Add orientation randomization
        orientations_delta = math_utils.quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
        base_quat_w = math_utils.quat_mul(orientations_delta, motion_ref_init_state.base_quat_w)
    else:
        base_quat_w = motion_ref_init_state.base_quat_w

    # write the root pose to the simulation
    asset.write_root_pose_to_sim(
        torch.cat(
            [
                base_pos_w,
                base_quat_w,
            ],
            dim=-1,
        ),
        env_ids=env_ids,
    )

    # apply velocity randomization if specified
    base_lin_vel_w = motion_ref_init_state.base_lin_vel_w * base_lin_vel_ratio
    base_ang_vel_w = motion_ref_init_state.base_ang_vel_w * base_ang_vel_ratio
    if randomize_velocity_range:
        # Apply velocity randomization
        range_list = [randomize_velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=base_lin_vel_w.device)
        vel_samples = math_utils.sample_uniform(
            ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=base_lin_vel_w.device
        )
        vel_samples = vel_samples * randomization_scale.unsqueeze(1)

        # Add velocity randomization
        base_lin_vel_w += vel_samples[:, 0:3]
        base_ang_vel_w += vel_samples[:, 3:6]

    # write the root velocity to the simulation
    asset.write_root_velocity_to_sim(
        torch.cat(
            [
                base_lin_vel_w,
                base_ang_vel_w,
            ],
            dim=-1,
        ),
        env_ids=env_ids,
    )

    # apply joint position randomization if specified
    joint_pos = motion_ref_init_state.joint_pos
    joint_vel = motion_ref_init_state.joint_vel * dof_vel_ratio
    if randomize_joint_pos_range != (0.0, 0.0):
        # Apply joint position randomization
        joint_pos_noise = math_utils.sample_uniform(
            randomize_joint_pos_range[0], randomize_joint_pos_range[1], joint_pos.shape, device=joint_pos.device
        )
        joint_pos_noise = joint_pos_noise * randomization_scale.unsqueeze(1)
        joint_pos += joint_pos_noise

    # write the joint state to the simulation
    asset.write_joint_state_to_sim(
        joint_pos,
        joint_vel,
        env_ids=env_ids,
    )
