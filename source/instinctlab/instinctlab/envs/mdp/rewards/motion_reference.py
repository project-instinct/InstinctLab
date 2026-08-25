from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Literal

import isaaclab.utils.math as math_utils
from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg

import instinctlab.motion_reference.utils as motion_reference_utils

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.envs import ManagerBasedRLEnv

    from instinctlab.motion_reference.motion_reference_manager import MotionReferenceManagerBase

"""
Reward functions
"""


def base_position_tracking_gauss(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    reference_cfg: SceneEntityCfg = SceneEntityCfg("motion_reference"),
    check_at_keyframe_threshold: float = -1,
    multiply_by_frame_interval: bool = False,
    tracking_torlerance: float = 0.0,
    tracking_sigma: float = 0.2,
) -> torch.Tensor:
    """Reward for tracking the base position of the robot to the reference position.
    In exponential gaussian form.

    Args:
        env: The environment object.
        asset_cfg: The configuration for the scene entity. Default is "robot".
        reference_cfg: The configuration for the scene entity. Default is "motion_reference".
        check_at_keyframe_threshold: Whether to check the base position at keyframe. Default is False.
            Only be non-zero when the reference is a keyframe.
        multiply_by_frame_interval: Whether to multiply the reward by the frame interval.
            Default is False.
    """
    motion_reference: MotionReferenceManagerBase = env.scene[reference_cfg.name]
    distance = motion_reference_utils.get_base_position_distance(
        env,
        asset_cfg,
        reference_cfg,
        squared=False,
    )
    if tracking_torlerance > 0:
        distance = torch.clamp(distance - tracking_torlerance, min=0)
    distance = torch.square(distance)

    rewards = torch.exp(-distance / (tracking_sigma * tracking_sigma))

    rewards = motion_reference_utils.matching_reference_timing(
        env,
        rewards,
        motion_reference,
        check_at_keyframe_threshold=check_at_keyframe_threshold,
        multiply_by_frame_interval=multiply_by_frame_interval,
    )
    return rewards


def base_position_tracking_neg_log(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    reference_cfg: SceneEntityCfg = SceneEntityCfg("motion_reference"),
    check_at_keyframe_threshold: float = -1,
    multiply_by_frame_interval: bool = False,
    tracking_torlerance: float = 1e-3,
) -> torch.Tensor:
    """Reward for tracking the base position of the robot to the reference position.
    In negative log form.

    Args:
        tracking_torlerance: The tolerance for tracking the base position, which also prevents
            the reward from becoming infinite.
    """
    motion_reference: MotionReferenceManagerBase = env.scene[reference_cfg.name]
    distance = motion_reference_utils.get_base_position_distance(
        env,
        asset_cfg,
        reference_cfg,
        squared=False,
    )
    if tracking_torlerance > 0:
        distance = torch.clamp(distance - tracking_torlerance, min=0)
    distance = torch.square(distance)

    rewards = -torch.log(distance + tracking_torlerance)

    rewards = motion_reference_utils.matching_reference_timing(
        env,
        rewards,
        motion_reference,
        check_at_keyframe_threshold=check_at_keyframe_threshold,
        multiply_by_frame_interval=multiply_by_frame_interval,
    )
    return rewards


def base_position_imitation_gauss(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    reference_cfg: SceneEntityCfg = SceneEntityCfg("motion_reference"),
    std: float = 0.1,
):
    motion_reference: MotionReferenceManagerBase = env.scene[reference_cfg.name]
    robot: Articulation = env.scene[asset_cfg.name]
    base_pos_ref = motion_reference.reference_frame.base_pos_w[:, 0]  # (batch_size, 3)
    base_pos = robot.data.root_pos_w.torch  # (batch_size, 3)
    base_pos_diff = base_pos - base_pos_ref  # (batch_size, 3)
    rewards = torch.exp(-torch.sum(torch.square(base_pos_diff), dim=-1) / (std * std))  # (batch_size,)
    # Apply validity mask from the reference frame
    rewards *= motion_reference.reference_frame.validity[:, 0]  # (batch_size,)
    return rewards


class base_position_offset_imitation_gauss(ManagerTermBase):
    """Reward for base position offset imitation using a Gaussian kernel.
    This reward is designed to encourage the agent to match the base position offset of the reference motion in EVERY timestep.
    Args:
        env: The environment object.
        asset_cfg: The configuration for the scene entity. Default is "robot".
        reference_cfg: The configuration for the scene entity. Default is "motion_reference".
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.asset_cfg = self.cfg.params.get("asset_cfg", SceneEntityCfg("robot"))
        self.asset = self._env.scene[self.asset_cfg.name]
        self.reference_cfg = self.cfg.params.get("reference_cfg", SceneEntityCfg("motion_reference"))
        self.motion_reference: MotionReferenceManagerBase = self._env.scene[self.reference_cfg.name]
        self.asset_position_marker = torch.ones_like(self.asset.data.root_pos_w.torch) * torch.nan  # (batch_size, 3)
        self.asset_quat_marker = torch.ones_like(self.asset.data.root_quat_w.torch) * torch.nan  # (batch_size, 4)
        self.reference_position_marker = (
            torch.ones_like(self.motion_reference.reference_frame.base_pos_w[:, 0]) * torch.nan
        )  # (batch_size, 3)
        self.reference_quat_marker = (
            torch.ones_like(self.motion_reference.reference_frame.base_quat_w[:, 0]) * torch.nan
        )

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        reference_cfg: SceneEntityCfg = SceneEntityCfg("motion_reference"),
        std: float = 0.1,
        in_base_frame: bool = True,
    ):
        self.refresh_markers()
        asset_pos_offset, reference_pos_offset = self.compute_base_position_offset(in_base_frame=in_base_frame)

        pos_offset_diff = asset_pos_offset - reference_pos_offset  # (batch_size, 3)
        rewards = torch.exp(-torch.sum(torch.square(pos_offset_diff), dim=-1) / (std * std))
        rewards[torch.isnan(rewards)] = 0.0  # Handle NaN values
        return rewards

    def compute_base_position_offset(self, in_base_frame: bool):
        """compute the base position offset"""
        asset_pos_offset = self.asset.data.root_pos_w.torch - self.asset_position_marker  # (batch_size, 3)
        reference_pos_offset = (
            self.motion_reference.reference_frame.base_pos_w[:, 0] - self.reference_position_marker
        )  # (batch_size, 3)
        if in_base_frame:
            asset_pos_offset = math_utils.quat_apply_inverse(
                self.asset_quat_marker,
                asset_pos_offset,
            )  # (batch_size, 3)
            reference_pos_offset = math_utils.quat_apply_inverse(
                self.reference_quat_marker,
                reference_pos_offset,
            )  # (batch_size, 3)
        return asset_pos_offset, reference_pos_offset

    def refresh_markers(self):
        landmarker_refresh_mask = self.motion_reference.time_passed_from_update < self._env.step_dt  # (batch_size, 1)
        self.asset_position_marker[landmarker_refresh_mask] = self.asset.data.root_pos_w.torch[landmarker_refresh_mask]
        self.asset_quat_marker[landmarker_refresh_mask] = self.asset.data.root_quat_w.torch[landmarker_refresh_mask]
        self.reference_position_marker[landmarker_refresh_mask] = self.motion_reference.reference_frame.base_pos_w[
            landmarker_refresh_mask
        ][:, 0]
        self.reference_quat_marker[landmarker_refresh_mask] = self.motion_reference.reference_frame.base_quat_w[
            landmarker_refresh_mask
        ][:, 0]


class base_position_offset_imitation_square(base_position_offset_imitation_gauss):
    """Reward for base position offset imitation using a square error.
    This reward is designed to encourage the agent to match the base position offset of the reference motion in EVERY timestep.
    Args:
        env: The environment object.
        asset_cfg: The configuration for the scene entity. Default is "robot".
        reference_cfg: The configuration for the scene entity. Default is "motion_reference".
    """

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        reference_cfg: SceneEntityCfg = SceneEntityCfg("motion_reference"),
        std: float = 0.1,
        in_base_frame: bool = True,
    ):
        self.refresh_markers()
        asset_pos_offset, reference_pos_offset = self.compute_base_position_offset(in_base_frame=in_base_frame)
        pos_offset_diff = asset_pos_offset - reference_pos_offset  # (batch_size, 3)
        rewards = torch.sum(torch.square(pos_offset_diff), dim=-1)  # (batch_size,)
        rewards[torch.isnan(rewards)] = 0.0  # Handle NaN values
        return rewards


def base_velocity_imitation_gauss(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    reference_cfg: SceneEntityCfg = SceneEntityCfg("motion_reference"),
    std: float = 0.5,
    in_base_frame: bool = True,
):
    """Reward for base velocity imitation using a Gaussian kernel.
    This reward is designed to encourage the agent to match the base velocity of the reference motion in EVERY timestep.
    Args:
        env: The environment object.
        asset_cfg: The configuration for the scene entity. Default is "robot".
        reference_cfg: The configuration for the scene entity. Default is "motion_reference".
    """
    motion_reference: MotionReferenceManagerBase = env.scene[reference_cfg.name]
    asset: RigidObject = env.scene[asset_cfg.name]
    base_vel_ref_w = motion_reference.reference_frame.base_lin_vel_w[:, 0]  # (batch_size, 6)
    if in_base_frame:
        root_quat_w = motion_reference.reference_frame.base_quat_w[:, 0]  # (batch_size, 4)
        base_vel_ref = math_utils.quat_apply_inverse(
            root_quat_w,
            base_vel_ref_w,
        )
        base_vel = asset.data.root_lin_vel_b.torch  # (batch_size, 3)
    else:
        base_vel_ref = base_vel_ref_w
        base_vel = asset.data.root_lin_vel_w.torch  # (batch_size, 3)
    base_vel_diff = base_vel - base_vel_ref  # (batch_size, 3)

    rewards = torch.exp(-torch.sum(torch.square(base_vel_diff), dim=-1) / (std * std))

    return rewards


def base_velocity_imitation_square(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    reference_cfg: SceneEntityCfg = SceneEntityCfg("motion_reference"),
    in_base_frame: bool = True,
):
    """Reward for base velocity imitation using a square error.
    This reward is designed to encourage the agent to match the base velocity of the reference motion in EVERY timestep.
    Args:
        env: The environment object.
        asset_cfg: The configuration for the scene entity. Default is "robot".
        reference_cfg: The configuration for the scene entity. Default is "motion_reference".
    """
    motion_reference: MotionReferenceManagerBase = env.scene[reference_cfg.name]
    asset: RigidObject = env.scene[asset_cfg.name]
    if in_base_frame:
        base_vel_ref = math_utils.quat_apply_inverse(
            motion_reference.reference_frame.base_quat_w[:, 0],
            motion_reference.reference_frame.base_lin_vel_w[:, 0],  # (batch_size, 3)
        )
        base_vel = asset.data.root_lin_vel_b.torch  # (batch_size, 3)
    else:
        base_vel_ref = motion_reference.reference_frame.base_lin_vel_w[:, 0]  # (batch_size, 3)
        base_vel = asset.data.root_lin_vel_w.torch  # (batch_size, 3)
    base_vel_diff = base_vel - base_vel_ref  # (batch_size, 3)

    rewards = torch.sum(torch.square(base_vel_diff), dim=-1)  # (batch_size,)

    return rewards


def base_rot_tracking_cos(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    reference_cfg: SceneEntityCfg = SceneEntityCfg("motion_reference"),
    check_at_keyframe_threshold: float = -1,
    multiply_by_frame_interval: bool = False,
) -> torch.Tensor:
    # extract useful elements
    asset: RigidObject = env.scene[asset_cfg.name]
    motion_reference: MotionReferenceManagerBase = env.scene[reference_cfg.name]

    # obtain the robot rotation and reference rotation
    quat = asset.data.root_state_w.torch[:, 3:7]
    quat_ref = motion_reference.data.base_quat_w[motion_reference.ALL_INDICES, motion_reference.aiming_frame_idx]

    # compute the cosine difference
    quat_diff = math_utils.quat_mul(quat_ref, math_utils.quat_conjugate(quat))
    vec = torch.zeros_like(quat_ref[:, :3])  # (num_envs, 3)
    vec[:, 0] = 1
    proj_vec = math_utils.quat_apply(quat_diff, vec)
    rewards = proj_vec[:, 0]  # (num_envs, 3)

    rewards = motion_reference_utils.matching_reference_timing(
        env,
        rewards,
        motion_reference,
        check_at_keyframe_threshold=check_at_keyframe_threshold,
        multiply_by_frame_interval=multiply_by_frame_interval,
    )
    return rewards


def base_rot_tracking_gauss(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    reference_cfg: SceneEntityCfg = SceneEntityCfg("motion_reference"),
    check_at_keyframe_threshold: float = -1,
    multiply_by_frame_interval: bool = False,
    tracking_torlerance: float = 0.0,
    tracking_sigma: float = 0.3,
):
    # extract useful elements
    asset: RigidObject = env.scene[asset_cfg.name]
    motion_reference: MotionReferenceManagerBase = env.scene[reference_cfg.name]

    # obtain the robot rotation and reference rotation
    quat = asset.data.root_state_w.torch[:, 3:7]
    quat_ref = motion_reference.data.base_quat_w[motion_reference.ALL_INDICES, motion_reference.aiming_frame_idx]

    # quaternion to euler
    angle = math_utils.quat_error_magnitude(quat_ref.reshape(-1, 4), quat.reshape(-1, 4))  # (batch_size,)
    if tracking_torlerance > 0:
        angle = torch.clamp(angle - tracking_torlerance, min=0)
    angle_square = torch.square(angle)
    rewards = torch.exp(-angle_square / (tracking_sigma * tracking_sigma))

    rewards = motion_reference_utils.matching_reference_timing(
        env,
        rewards,
        motion_reference,
        check_at_keyframe_threshold=check_at_keyframe_threshold,
        multiply_by_frame_interval=multiply_by_frame_interval,
    )
    return rewards


def base_rot_imitation_gauss(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    reference_cfg: SceneEntityCfg = SceneEntityCfg("motion_reference"),
    difference_type: Literal["axis_angle", "box_minus"] = "axis_angle",
    std: float = 0.3,
):
    """Reward for base rotation imitation using a Gaussian kernel.
    This reward is designed to encourage the agent to match the base rotation of the reference motion in EVERY timestep.
    Args:
        env: The environment object.
        asset_cfg: The configuration for the scene entity. Default is "robot".
        reference_cfg: The configuration for the scene entity. Default is "motion_reference".
    """
    motion_reference: MotionReferenceManagerBase = env.scene[reference_cfg.name]
    asset: RigidObject = env.scene[asset_cfg.name]
    quat_ref = motion_reference.reference_frame.base_quat_w[:, 0]  # (batch_size, 4)
    quat = asset.data.root_quat_w.torch  # (batch_size, 4)
    if difference_type == "axis_angle":
        quat_diff = math_utils.quat_mul(quat_ref, math_utils.quat_conjugate(quat))  # (batch_size, 4)
        axisang = math_utils.axis_angle_from_quat(quat_diff)  # (batch_size, 3)
        rot_error = torch.norm(axisang, dim=-1)  # (batch_size,)
    elif difference_type == "box_minus":
        quat_box_minus = math_utils.quat_box_minus(quat_ref, quat)
        rot_error = torch.norm(quat_box_minus, dim=-1)  # (batch_size,)
    else:
        raise ValueError(f"Unsupported difference method: {difference_type}.")

    rewards = torch.exp(-torch.square(rot_error) / (std * std))

    return rewards


def base_rot_imitation_square(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    reference_cfg: SceneEntityCfg = SceneEntityCfg("motion_reference"),
    difference_type: Literal["axis_angle", "box_minus"] = "axis_angle",
):
    """Reward for base rotation imitation using a square error.
    This reward is designed to encourage the agent to match the base rotation of the reference motion in EVERY timestep.
    Args:
        env: The environment object.
        asset_cfg: The configuration for the scene entity. Default is "robot".
        reference_cfg: The configuration for the scene entity. Default is "motion_reference".
    """
    motion_reference: MotionReferenceManagerBase = env.scene[reference_cfg.name]
    asset: RigidObject = env.scene[asset_cfg.name]
    quat_ref = motion_reference.reference_frame.base_quat_w[:, 0]  # (batch_size, 4)
    quat = asset.data.root_quat_w.torch  # (batch_size, 4)
    if difference_type == "axis_angle":
        rot_error = math_utils.quat_error_magnitude(quat_ref.reshape(-1, 4), quat.reshape(-1, 4))
    elif difference_type == "box_minus":
        quat_box_minus = math_utils.quat_box_minus(quat_ref, quat)
        rot_error = torch.norm(quat_box_minus, dim=-1)  # (batch_size,)
    else:
        raise ValueError(f"Unsupported difference method: {difference_type}.")

    rewards = torch.square(rot_error)

    return rewards


def base_projected_gravity_tracking_gauss(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    reference_cfg: SceneEntityCfg = SceneEntityCfg("motion_reference"),
    check_at_keyframe_threshold: float = -1,
    multiply_by_frame_interval: bool = False,
    tracking_torlerance: float = 0.0,
    tracking_sigma: float = 0.2,
):
    """Reward for tracking the projected gravity of the robot to the reference projected gravity.
    In exponential gaussian form.

    Args:
        env: The environment object.
        asset_cfg: The configuration for the scene entity. Default is "robot".
        reference_cfg: The configuration for the scene entity. Default is "motion_reference".
        check_at_keyframe_threshold: Whether to check the base position at keyframe. Default is False.
            Only be non-zero when the reference is a keyframe.
        multiply_by_frame_interval: Whether to multiply the reward by the frame interval.
            Default is False.
    """
    # extract useful elements
    asset: RigidObject = env.scene[asset_cfg.name]
    motion_reference: MotionReferenceManagerBase = env.scene[reference_cfg.name]

    # obtain the robot rotation and reference rotation
    quat = asset.data.root_state_w.torch[:, 3:7]
    quat_ref = motion_reference.data.base_quat_w[motion_reference.ALL_INDICES, motion_reference.aiming_frame_idx]
    GRAVITY_VEC_W = asset.data.GRAVITY_VEC_W.torch
    projected_gravity = math_utils.quat_apply_inverse(
        quat, GRAVITY_VEC_W
    )  # (num_envs, 3), projected gravity in the robot's local frame
    projected_gravity_ref = math_utils.quat_apply_inverse(
        quat_ref, GRAVITY_VEC_W
    )  # (num_envs, 3), projected gravity in the robot's local frame

    reward = projected_gravity - projected_gravity_ref  # (num_envs, 3)
    if tracking_torlerance > 0:
        reward = torch.clamp(torch.abs(reward) - tracking_torlerance, min=0)
    reward = torch.square(reward)  # (num_envs, 3)
    rewards = torch.exp(-torch.sum(reward, dim=-1) / (tracking_sigma * tracking_sigma))
    rewards = motion_reference_utils.matching_reference_timing(
        env,
        rewards,
        motion_reference,
        check_at_keyframe_threshold=check_at_keyframe_threshold,
        multiply_by_frame_interval=multiply_by_frame_interval,
    )
    return rewards


def joint_pos_tracking_square(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    reference_cfg: SceneEntityCfg = SceneEntityCfg("motion_reference"),
    check_at_keyframe_threshold: float = -1,
    multiply_by_frame_interval: bool = False,
):
    motion_reference: MotionReferenceManagerBase = env.scene[reference_cfg.name]
    joint_pos_diff = motion_reference_utils.get_joint_position_difference(env, asset_cfg, reference_cfg)

    rewards = torch.sum(torch.square(joint_pos_diff), dim=-1)

    rewards = motion_reference_utils.matching_reference_timing(
        env,
        rewards,
        motion_reference,
        check_at_keyframe_threshold=check_at_keyframe_threshold,
        multiply_by_frame_interval=multiply_by_frame_interval,
    )
    return rewards


def joint_pos_tracking_gauss(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    reference_cfg: SceneEntityCfg = SceneEntityCfg("motion_reference"),
    check_at_keyframe_threshold: float = -1,
    multiply_by_frame_interval: bool = False,
    normalize_by_num_joints: bool = False,
    tracking_torlerance: float = 0.0,
    tracking_sigma: float = 0.7,
    combine_method: str = "prod",
    zero_on_none_masked_on: bool = False,
):
    """Reward for joint position tracking using a Gaussian kernel.
    ### NOTE:
        - Please assign subset of joints using `reference_cfg.joint_names` instead of anything in `asset_cfg`
    Args:
        combine_method: The method to combine all joints error. Default is "prod".
            - "prod": multiply rewards for each joint's gaussian reward.
            - "sum": sum rewards for each joint's gaussian reward.
        zero_on_none_masked_on: If True, the reward will be zeroed out if the reference motion does not have a mask for
            any joint.
    """
    motion_reference: MotionReferenceManagerBase = env.scene[reference_cfg.name]
    joint_pos_diff = motion_reference_utils.get_joint_position_difference(env, asset_cfg, reference_cfg, masking=False)
    joint_pos_diff_mask = motion_reference.data.joint_pos_mask[
        motion_reference.ALL_INDICES, motion_reference.aiming_frame_idx
    ]  # (batch_size, num_joints)
    joint_pos_diff_mask = joint_pos_diff_mask[:, reference_cfg.joint_ids]  # (batch_size, num_joints_selected)
    if tracking_torlerance > 0:
        joint_pos_diff = torch.clamp(torch.abs(joint_pos_diff) - tracking_torlerance, min=0)
    joint_pos_err = torch.square(joint_pos_diff)

    if combine_method == "prod":
        # error is already masked and will not affect the exponential operation.
        joint_pos_err = torch.sum(joint_pos_err * joint_pos_diff_mask, dim=-1)  # (batch_size,)
        if normalize_by_num_joints:
            joint_pos_err = joint_pos_err / joint_pos_diff.shape[-1]

    rewards = torch.exp(-joint_pos_err / (tracking_sigma * tracking_sigma))

    if combine_method == "sum":
        rewards = torch.sum(rewards * joint_pos_diff_mask, dim=-1)  # (batch_size,)
        if normalize_by_num_joints:
            rewards = rewards / joint_pos_diff.shape[-1]

    rewards = motion_reference_utils.matching_reference_timing(
        env,
        rewards,
        motion_reference,
        check_at_keyframe_threshold=check_at_keyframe_threshold,
        multiply_by_frame_interval=multiply_by_frame_interval,
    )
    if zero_on_none_masked_on:
        # If the reference motion does not have a mask for any joint, the reward will be zeroed out.
        rewards[joint_pos_diff_mask.any(dim=-1) == False] = 0.0
    return rewards


class joint_pos_tracking_gauss_normalized(ManagerTermBase):
    """joint_pos_tracking with gauss kernel, the sigma is computed dynamically for each joints across all samples."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        reference_cfg: SceneEntityCfg = SceneEntityCfg("motion_reference"),
        check_at_keyframe_threshold: float = -1,
        multiply_by_frame_interval: bool = False,
        normalize_by_num_joints: bool = False,
        tracking_torlerance: float = 0.0,
        sigma_times: float = 1.2,
        combine_method: str = "prod",
    ):
        motion_reference: MotionReferenceManagerBase = env.scene[reference_cfg.name]
        joint_pos_diff = motion_reference_utils.get_joint_position_difference(env, asset_cfg, reference_cfg)

        tracking_var = torch.var(joint_pos_diff, dim=0, keepdim=(combine_method == "sum")) * sigma_times * sigma_times

        if tracking_torlerance > 0:
            joint_pos_diff = torch.clamp(torch.abs(joint_pos_diff) - tracking_torlerance, min=0)
        joint_pos_err = torch.square(joint_pos_diff)

        if combine_method == "prod":
            joint_pos_err = torch.sum(joint_pos_err, dim=-1)  # (batch_size,)
            if normalize_by_num_joints:
                joint_pos_err = joint_pos_err / joint_pos_diff.shape[-1]

        rewards = torch.exp(-joint_pos_err / tracking_var)

        rewards = motion_reference_utils.matching_reference_timing(
            env,
            rewards,
            motion_reference,
            check_at_keyframe_threshold=check_at_keyframe_threshold,
            multiply_by_frame_interval=multiply_by_frame_interval,
        )

        if combine_method == "sum":
            rewards = torch.sum(rewards, dim=-1)  # (batch_size,)
            if normalize_by_num_joints:
                rewards = rewards / joint_pos_diff.shape[-1]


def joint_pos_imitation_gauss(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    reference_cfg: SceneEntityCfg = SceneEntityCfg("motion_reference"),
    std: float = 0.7,
    masked: bool = False,
    combine_method: Literal["prod", "sum", "mean_prod"] = "prod",
):
    """
    Reward for joint position imitation using a Gaussian kernel.
    This reward is designed to encourage the agent to match the joint positions of the reference motion in EVERY timestep.

    Args:
        env: The environment object.
        asset_cfg: The configuration for the scene entity. Default is "robot".
        reference_cfg: The configuration for the scene entity. Default is "motion_reference".
        masked: If True, only compute rewards for joints that are not masked in the reference motion.
    """
    motion_reference: MotionReferenceManagerBase = env.scene[reference_cfg.name]
    robot: Articulation = env.scene[asset_cfg.name]
    joint_pos_ids = asset_cfg.joint_ids
    joint_pos_ref = motion_reference.reference_frame.joint_pos[:, 0, joint_pos_ids]  # (batch_size, num_joints_selected)
    joint_pos = robot.data.joint_pos.torch[:, joint_pos_ids]  # (batch_size, num_joints_selected)
    joint_pos_diff = joint_pos - joint_pos_ref  # (batch_size, num_joints_selected)
    joint_pos_square = torch.square(joint_pos_diff)

    if masked:
        joint_pos_mask = motion_reference.reference_frame.joint_pos_mask[:, 0, joint_pos_ids]
    else:
        joint_pos_mask = 1

    if combine_method == "prod":
        # (batch_size, num_joints_selected) -> (batch_size,)
        joint_pos_square = torch.sum(joint_pos_square * joint_pos_mask, dim=-1)
    elif combine_method == "mean_prod":
        # (batch_size, num_joints_selected) -> (batch_size,)
        joint_pos_square = torch.mean(joint_pos_square * joint_pos_mask, dim=-1)

    rewards = torch.exp(-joint_pos_square / (std * std))

    if combine_method == "sum":
        # (batch_size, num_joints_selected) -> (batch_size,)
        rewards = torch.sum(rewards * joint_pos_mask, dim=-1)

    return rewards


def joint_pos_imitation_square(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    reference_cfg: SceneEntityCfg = SceneEntityCfg("motion_reference"),
    masked: bool = False,
):
    """Reward for joint position imitation using a square error.
    This reward is designed to encourage the agent to match the joint positions of the reference motion in EVERY timestep.
    Args:
        env: The environment object.
        asset_cfg: The configuration for the scene entity. Default is "robot".
        reference_cfg: The configuration for the scene entity. Default is "motion_reference".
        masked: If True, only compute rewards for joints that are not masked in the reference motion.
    """
    motion_reference: MotionReferenceManagerBase = env.scene[reference_cfg.name]
    robot: Articulation = env.scene[asset_cfg.name]
    joint_pos_ids = asset_cfg.joint_ids
    joint_pos_ref = motion_reference.reference_frame.joint_pos[:, 0, joint_pos_ids]  # (batch_size, num_joints_selected)
    joint_pos = robot.data.joint_pos.torch[:, joint_pos_ids]  # (batch_size, num_joints_selected)
    joint_pos_diff = joint_pos - joint_pos_ref  # (batch_size, num_joints_selected)
    if masked:
        joint_pos_mask = motion_reference.reference_frame.joint_pos_mask[:, 0, joint_pos_ids]
        joint_pos_diff = joint_pos_diff * joint_pos_mask

    rewards = torch.sum(torch.square(joint_pos_diff), dim=-1)  # (batch_size,)

    return rewards


def joint_vel_imitation_gauss(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    reference_cfg: SceneEntityCfg = SceneEntityCfg("motion_reference"),
    std: float = 10.0,
    masked: bool = False,
    combine_method: Literal["prod", "sum", "mean_prod"] = "prod",
):
    """
    Reward for joint velocity imitation using a Gaussian kernel.
    This reward is designed to encourage the agent to match the joint velocities of the reference motion in EVERY timestep.

    Args:
        env: The environment object.
        asset_cfg: The configuration for the scene entity. Default is "robot".
        reference_cfg: The configuration for the scene entity. Default is "motion_reference".
        masked: If True, only compute rewards for joints that are not masked in the reference motion.
    """
    motion_reference: MotionReferenceManagerBase = env.scene[reference_cfg.name]
    robot: Articulation = env.scene[asset_cfg.name]
    joint_vel_ids = asset_cfg.joint_ids
    joint_vel_ref = motion_reference.reference_frame.joint_vel[:, 0, joint_vel_ids]  # (batch_size, num_joints_selected)
    joint_vel = robot.data.joint_vel.torch[:, joint_vel_ids]  # (batch_size, num_joints_selected)
    joint_vel_diff = joint_vel - joint_vel_ref  # (batch_size, num_joints_selected)
    joint_vel_square = torch.square(joint_vel_diff)  # (batch_size, num_joints_selected)
    if masked:
        joint_vel_mask = motion_reference.reference_frame.joint_vel_mask[:, 0, joint_vel_ids]
    else:
        joint_vel_mask = 1

    if combine_method == "prod":
        # (batch_size, num_joints_selected) -> (batch_size,)
        joint_vel_square = torch.sum(joint_vel_square * joint_vel_mask, dim=-1)
    elif combine_method == "mean_prod":
        # (batch_size, num_joints_selected) -> (batch_size,)
        joint_vel_square = torch.mean(joint_vel_square * joint_vel_mask, dim=-1)

    rewards = torch.exp(-joint_vel_square / (std * std))

    if combine_method == "sum":
        # (batch_size, num_joints_selected) -> (batch_size,)
        rewards = torch.sum(rewards * joint_vel_mask, dim=-1)

    return rewards


def joint_vel_imitation_square(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    reference_cfg: SceneEntityCfg = SceneEntityCfg("motion_reference"),
    masked: bool = False,
):
    """Reward for joint velocity tracking using a square error.
    This reward is designed to encourage the agent to match the joint velocities of the reference motion in EVERY timestep.
    Args:
        env: The environment object.
        asset_cfg: The configuration for the scene entity. Default is "robot".
        reference_cfg: The configuration for the scene entity. Default is "motion_reference".
        masked: If True, only compute rewards for joints that are not masked in the reference motion.
    """
    motion_reference: MotionReferenceManagerBase = env.scene[reference_cfg.name]
    robot: Articulation = env.scene[asset_cfg.name]
    joint_vel_ids = asset_cfg.joint_ids
    joint_vel_ref = motion_reference.reference_frame.joint_vel[:, 0, joint_vel_ids]  # (batch_size, num_joints_selected)
    joint_vel = robot.data.joint_vel.torch[:, joint_vel_ids]  # (batch_size, num_joints_selected)
    joint_vel_diff = joint_vel - joint_vel_ref  # (batch_size, num_joints_selected)
    if masked:
        joint_vel_mask = motion_reference.reference_frame.joint_vel_mask[:, 0, joint_vel_ids]
        joint_vel_diff = joint_vel_diff * joint_vel_mask

    rewards = torch.sum(torch.square(joint_vel_diff), dim=-1)  # (batch_size,)

    return rewards


def link_pos_tracking_square(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    reference_cfg: SceneEntityCfg = SceneEntityCfg("motion_reference"),
    in_base_frame: bool = True,
    check_at_keyframe_threshold: float = -1,
    multiply_by_frame_interval: bool = False,
):
    motion_reference: MotionReferenceManagerBase = env.scene[reference_cfg.name]
    link_pos_distance = motion_reference_utils.get_link_position_distance(
        env,
        asset_cfg,
        reference_cfg,
        in_base_frame=in_base_frame,
        squared=True,
    )  # (batch_size, num_links)

    rewards = torch.sum(link_pos_distance, dim=-1)

    rewards = motion_reference_utils.matching_reference_timing(
        env,
        rewards,
        motion_reference,
        check_at_keyframe_threshold=check_at_keyframe_threshold,
        multiply_by_frame_interval=multiply_by_frame_interval,
    )
    return rewards


def link_pos_tracking_gauss(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    reference_cfg: SceneEntityCfg = SceneEntityCfg("motion_reference"),
    in_base_frame: bool = True,
    check_at_keyframe_threshold: float = -1,
    multiply_by_frame_interval: bool = False,
    normalize_by_num_links: bool = False,
    tracking_torlerance: float = 0.2,
    tracking_sigma: float = 0.7,
    combine_method: str = "prod",
    zero_on_none_masked_on: bool = False,
):
    """Reward for link position tracking using a Gaussian kernel.
    ### NOTE:
        - Please assign subset of links using `reference_cfg.body_names` instead of anything in `asset_cfg`
    Args:
        combine_method: The method to combine all links error. Default is "prod".
            - "prod": multiply rewards for each link's gaussian reward.
            - "sum": sum rewards for each link's gaussian reward.
    """
    motion_reference: MotionReferenceManagerBase = env.scene[reference_cfg.name]
    link_pos_distance = motion_reference_utils.get_link_position_distance(
        env,
        asset_cfg,
        reference_cfg,
        in_base_frame=in_base_frame,
        masking=False,
        squared=False,
    )  # (batch_size, num_links)
    assert len(link_pos_distance.shape) == 2
    link_pos_distance_mask = motion_reference.data.link_pos_mask[
        motion_reference.ALL_INDICES, motion_reference.aiming_frame_idx
    ]  # (batch_size, num_links)
    link_pos_distance_mask = link_pos_distance_mask[:, reference_cfg.body_ids]  # (batch_size, num_links_selected)

    if tracking_torlerance > 0:
        link_pos_distance = torch.clamp(link_pos_distance - tracking_torlerance, min=0)
    link_pos_err = torch.square(link_pos_distance)

    if combine_method == "prod":
        link_pos_err = torch.sum(link_pos_err * link_pos_distance_mask, dim=-1)  # (batch_size,)
        if normalize_by_num_links:
            link_pos_err = link_pos_err / link_pos_distance.shape[-1]

    rewards = torch.exp(-link_pos_err / (tracking_sigma * tracking_sigma))

    if combine_method == "sum":
        rewards = torch.sum(rewards * link_pos_distance_mask, dim=-1)  # (batch_size,)
        if normalize_by_num_links:
            rewards = rewards / link_pos_distance.shape[-1]

    rewards = motion_reference_utils.matching_reference_timing(
        env,
        rewards,
        motion_reference,
        check_at_keyframe_threshold=check_at_keyframe_threshold,
        multiply_by_frame_interval=multiply_by_frame_interval,
    )
    if zero_on_none_masked_on:
        # If the reference motion does not have a mask for any link, the reward will be zeroed out.
        rewards[link_pos_distance_mask.any(dim=-1) == False] = 0.0
    return rewards


def link_rot_tracking_gauss(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    reference_cfg: SceneEntityCfg = SceneEntityCfg("motion_reference"),
    in_base_frame: bool = True,
    check_at_keyframe_threshold: float = -1,
    multiply_by_frame_interval: bool = False,
    normalize_by_num_links: bool = False,
    tracking_sigma: float = 0.7,
    combine_method: str = "prod",
):
    """
    Args:
        combine_method: The method to combine all links error. Default is "prod".
            - "prod": multiply rewards for each link's gaussian reward.
            - "sum": sum rewards for each link's gaussian reward.
    """
    motion_reference: MotionReferenceManagerBase = env.scene[reference_cfg.name]
    link_rot_distance = motion_reference_utils.get_link_rotation_distance(
        env,
        asset_cfg,
        reference_cfg,
        in_base_frame=in_base_frame,
    )  # (batch_size, num_links)

    link_rot_err = torch.square(link_rot_distance)

    if combine_method == "prod":
        link_rot_err = torch.sum(link_rot_err, dim=-1)  # (batch_size,)
        if normalize_by_num_links:
            link_rot_err = link_rot_err / link_rot_distance.shape[-1]

    rewards = torch.exp(-link_rot_err / (tracking_sigma * tracking_sigma))

    if combine_method == "sum":
        rewards = torch.sum(rewards, dim=-1)  # (batch_size,)
        if normalize_by_num_links:
            rewards = rewards / link_rot_distance.shape[-1]

    rewards = motion_reference_utils.matching_reference_timing(
        env,
        rewards,
        motion_reference,
        check_at_keyframe_threshold=check_at_keyframe_threshold,
        multiply_by_frame_interval=multiply_by_frame_interval,
    )
    return rewards


def link_pos_imitation_gauss(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    reference_cfg: SceneEntityCfg = SceneEntityCfg("motion_reference"),
    in_base_frame: bool = True,
    in_relative_world_frame: bool = False,
    std: float = 0.1,
    masked: bool = False,
    combine_method: Literal["prod", "sum", "mean_prod"] = "prod",
):
    """
    Reward for link position imitation using a Gaussian kernel.
    This reward is designed to encourage the agent to match the link positions of the reference motion in EVERY timestep.

    Args:
        env: The environment object.
        asset_cfg: The configuration for the scene entity. Default is "robot".
        reference_cfg: The configuration for the scene entity. Default is "motion_reference".
        in_base_frame: Whether to compute the link position in the base frame of the robot.
        in_relative_world_frame: Whether to compute the link position difference between the reference link pose,
            which are moved to robot's relative position.
        masked: If True, only compute rewards for links that are not masked in the reference motion.
    """
    motion_reference: MotionReferenceManagerBase = env.scene[reference_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]
    links = asset.find_bodies(motion_reference.cfg.link_of_interests, preserve_order=True)
    link_indices = links[0]
    links_pos_w = asset.data.body_link_pos_w.torch[:, link_indices]  # (batch_size, num_links, 3)
    if in_base_frame:
        root_pos_w = asset.data.root_pos_w.torch
        root_quat_w = asset.data.root_quat_w.torch
        root_pos_w_inv, root_quat_w_inv = math_utils.subtract_frame_transforms(root_pos_w, root_quat_w)
        links_pos = math_utils.transform_points(
            links_pos_w,
            root_pos_w_inv,
            root_quat_w_inv,
        )
        link_pos_ref = motion_reference.reference_frame.link_pos_b[:, 0]
    elif in_relative_world_frame:
        links_pos = links_pos_w
        link_pos_ref = motion_reference.reference_link_pos_relative_w
    else:
        links_pos = links_pos_w
        link_pos_ref = motion_reference.reference_frame.link_pos_w[:, 0]

    link_pos_diff = links_pos - link_pos_ref  # (batch_size, num_links, 3)
    link_pos_square = torch.sum(torch.square(link_pos_diff), dim=-1)  # (batch_size, num_links)
    if masked:
        # (batch_size, num_links_selected)
        link_mask = motion_reference.reference_frame.link_pos_mask[:, 0, reference_cfg.body_ids]
    else:
        link_mask = 1

    # select a subset of link_of_interests
    link_pos_square = link_pos_square[:, reference_cfg.body_ids]  # (batch_size, num_links_selected)

    if combine_method == "prod":
        # (batch_size, num_links_selected) -> (batch_size,)
        link_pos_square = torch.sum(link_pos_square * link_mask, dim=-1)
    if combine_method == "mean_prod":
        # (batch_size, num_links_selected) -> (batch_size,)
        link_pos_square = torch.mean(link_pos_square * link_mask, dim=-1)

    rewards = torch.exp(-link_pos_square / (std * std))

    if combine_method == "sum":
        # (batch_size, num_links_selected) -> (batch_size,)
        rewards = torch.sum(rewards * link_mask, dim=-1)

    return rewards


def link_pos_imitation_square(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    reference_cfg: SceneEntityCfg = SceneEntityCfg("motion_reference"),
    in_base_frame: bool = True,
    masked: bool = False,
):
    """Reward for link position imitation using a square error.
    This reward is designed to encourage the agent to match the link positions of the reference motion in EVERY timestep.
    Args:
        env: The environment object.
        asset_cfg: The configuration for the scene entity. Default is "robot".
        reference_cfg: The configuration for the scene entity. Default is "motion_reference".
        in_base_frame: Whether to compute the link position in the base frame of the robot.
        masked: If True, only compute rewards for links that are not masked in the reference motion.
    """
    motion_reference: MotionReferenceManagerBase = env.scene[reference_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]
    links = asset.find_bodies(motion_reference.cfg.link_of_interests, preserve_order=True)
    link_indices = links[0]
    links_pos_w = asset.data.body_link_pos_w.torch[:, link_indices]  # (batch_size, num_links, 3)
    if in_base_frame:
        root_pos_w = asset.data.root_pos_w.torch
        root_quat_w = asset.data.root_quat_w.torch
        root_pos_w_inv, root_quat_w_inv = math_utils.subtract_frame_transforms(root_pos_w, root_quat_w)
        links_pos = math_utils.transform_points(
            links_pos_w,
            root_pos_w_inv,
            root_quat_w_inv,
        )  # (batch_size, num_links, 3)
        link_pos_ref = motion_reference.reference_frame.link_pos_b[:, 0]
    else:
        links_pos = links_pos_w  # (batch_size, num_links, 3)
        link_pos_ref = motion_reference.reference_frame.link_pos_w[:, 0]  # (batch_size, num_links, 3)
    link_pos_diff = links_pos - link_pos_ref  # (batch_size, num_links, 3)
    link_pos_dist = torch.norm(link_pos_diff, dim=-1)  # (batch_size, num_links)
    if masked:
        link_mask = motion_reference.reference_frame.link_pos_mask[:, 0]
        link_pos_dist = link_pos_dist * link_mask  # (batch_size, num_links)
    # select a subset of link_of_interests
    link_pos_dist = link_pos_dist[:, reference_cfg.body_ids]  # (batch_size, num_links_selected)
    rewards = torch.sum(torch.square(link_pos_dist), dim=-1)  # (batch_size, num_links_selected)
    return rewards


def link_pos_imitation_neg_log(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    reference_cfg: SceneEntityCfg = SceneEntityCfg("motion_reference"),
    in_base_frame: bool = True,
    masked: bool = False,
    min_error: float = 1e-3,  # to avoid log(0) issues
):
    """Reward for computing the negative log of the link position error in every timestep."""
    motion_reference: MotionReferenceManagerBase = env.scene[reference_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]
    links = asset.find_bodies(motion_reference.cfg.link_of_interests, preserve_order=True)
    link_indices = links[0]
    links_pos_w = asset.data.body_link_pos_w.torch[:, link_indices]  # (batch_size, num_links, 3)
    if in_base_frame:
        root_pos_w = asset.data.root_pos_w.torch
        root_quat_w = asset.data.root_quat_w.torch
        root_pos_w_inv, root_quat_w_inv = math_utils.subtract_frame_transforms(root_pos_w, root_quat_w)
        links_pos = math_utils.transform_points(
            links_pos_w,
            root_pos_w_inv,
            root_quat_w_inv,
        )  # (batch_size, num_links, 3)
        link_pos_ref = motion_reference.reference_frame.link_pos_b[:, 0]
    else:
        links_pos = links_pos_w  # (batch_size, num_links, 3)
        link_pos_ref = motion_reference.reference_frame.link_pos_w[:, 0]  # (batch_size, num_links, 3)
    link_pos_diff = links_pos - link_pos_ref  # (batch_size, num_links, 3)
    link_pos_dist = torch.norm(link_pos_diff, dim=-1)  # (batch_size, num_links)

    # Compute the negative log of the link position error
    rewards = -torch.log(link_pos_dist + min_error)  # shape (batch_size, num_links)
    if masked:
        link_mask = motion_reference.reference_frame.link_pos_mask[:, 0]
        rewards = rewards * link_mask  # (batch_size, num_links)
    rewards = rewards[:, reference_cfg.body_ids]  # (batch_size, num_links_selected)
    rewards = torch.mean(rewards, dim=-1)  # (batch_size,)

    return rewards


def link_rot_imitation_gauss(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    reference_cfg: SceneEntityCfg = SceneEntityCfg("motion_reference"),
    in_base_frame: bool = True,
    in_relative_world_frame: bool = False,
    std: float = 0.4,
    masked: bool = False,
    combine_method: Literal["prod", "sum", "mean_prod"] = "prod",
):
    """
    Reward for link rotation imitation using a Gaussian kernel.
    This reward is designed to encourage the agent to match the link rotations of the reference motion in EVERY timestep.
    Args:
        env: The environment object.
        asset_cfg: The configuration for the scene entity. Default is "robot".
        reference_cfg: The configuration for the scene entity. Default is "motion_reference".
        in_base_frame: Whether to compute the link rotation in the base frame of the robot.
        in_relative_world_frame: Whether to compute the link position difference between the reference link pose,
            which are moved to robot's relative position.
        masked: If True, only compute rewards for links that are not masked in the reference motion.
    """
    motion_reference: MotionReferenceManagerBase = env.scene[reference_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]
    links = asset.find_bodies(motion_reference.cfg.link_of_interests, preserve_order=True)  # type: ignore
    link_indices = links[0]
    links_rot_w = asset.data.body_link_quat_w.torch[:, link_indices]  # (batch_size, num_links, 4)
    if in_base_frame:
        root_quat_w = asset.data.root_quat_w.torch
        root_quat_w_inv = math_utils.quat_inv(root_quat_w)  # (batch_size, 4)
        link_rot = math_utils.quat_mul(root_quat_w_inv.unsqueeze(1).expand(-1, links_rot_w.shape[1], -1), links_rot_w)
        link_rot_ref = motion_reference.reference_frame.link_quat_b[:, 0]
    elif in_relative_world_frame:
        link_rot = links_rot_w
        link_rot_ref = motion_reference.reference_link_quat_relative_w
    else:
        link_rot = links_rot_w
        link_rot_ref = motion_reference.reference_frame.link_quat_w[:, 0]
    link_rot_error_magnitude = math_utils.quat_error_magnitude(
        link_rot.reshape(-1, 4), link_rot_ref.reshape(-1, 4)
    ).reshape(
        link_rot.shape[0], link_rot.shape[1]
    )  # (batch_size, num_links)

    # select a subset of link_of_interests
    link_rot_error_magnitude = link_rot_error_magnitude[:, reference_cfg.body_ids]
    link_rot_error_square = torch.square(link_rot_error_magnitude)  # (batch_size, num_links_selected)
    if masked:
        # (batch_size, num_links_selected)
        link_rot_mask = motion_reference.reference_frame.link_rot_mask[:, 0, reference_cfg.body_ids]
    else:
        link_rot_mask = 1

    if combine_method == "prod":
        # (batch_size, num_links_selected) -> (batch_size,)
        link_rot_error_square = torch.sum(link_rot_error_square * link_rot_mask, dim=-1)
    if combine_method == "mean_prod":
        # (batch_size, num_links_selected) -> (batch_size,)
        link_rot_error_square = torch.mean(link_rot_error_square * link_rot_mask, dim=-1)

    rewards = torch.exp(-link_rot_error_square / (std * std))

    if combine_method == "sum":
        # (batch_size, num_links_selected) -> (batch_size,)
        rewards = torch.sum(rewards * link_rot_mask, dim=-1)

    return rewards


def link_lin_vel_imitation_gauss(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    reference_cfg: SceneEntityCfg = SceneEntityCfg("motion_reference"),
    in_base_frame: bool = False,
    std: float = 0.4,
    masked: bool = False,
    combine_method: Literal["prod", "sum", "mean_prod"] = "prod",
):
    """
    Reward for link linear velocity imitation using a Gaussian kernel.
    This reward is designed to encourage the agent to match the link linear velocities of the reference motion in EVERY timestep.
    Args:
        env: The environment object.
        asset_cfg: The configuration for the scene entity. Default is "robot".
        reference_cfg: The configuration for the scene entity. Default is "motion_reference".
        in_base_frame: Whether to compute the link linear velocity in the base frame of the robot.
        masked: If True, only compute rewards for links that are not masked in the reference motion.
    """
    motion_reference: MotionReferenceManagerBase = env.scene[reference_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]
    links = asset.find_bodies(motion_reference.cfg.link_of_interests, preserve_order=True)  # type: ignore
    link_indices = links[0]
    links_lin_vel_w = asset.data.body_link_lin_vel_w.torch[:, link_indices]  # (batch_size, num_links, 3)
    if in_base_frame:
        root_pos_w = asset.data.root_pos_w.torch
        root_quat_w = asset.data.root_quat_w.torch
        root_lin_vel_w = asset.data.root_lin_vel_w.torch
        root_ang_vel_w = asset.data.root_ang_vel_w.torch
        links_pos_w = asset.data.body_link_pos_w.torch[:, link_indices]  # (batch_size, num_links, 3)
        link_pos_offset_w = links_pos_w - root_pos_w.unsqueeze(1)  # (batch_size, num_links, 3)
        link_lin_vel = math_utils.quat_apply_inverse(
            root_quat_w.unsqueeze(1).expand(-1, links_lin_vel_w.shape[1], -1),
            (
                links_lin_vel_w
                - root_lin_vel_w.unsqueeze(1)
                - torch.cross(root_ang_vel_w.unsqueeze(1), link_pos_offset_w, dim=-1)
            ),
        )  # (batch_size, num_links, 3)
        link_lin_vel_ref = motion_reference.reference_frame.link_lin_vel_b[:, 0]
    else:
        link_lin_vel = links_lin_vel_w
        link_lin_vel_ref = motion_reference.reference_frame.link_lin_vel_w[:, 0]
    link_lin_vel_diff = link_lin_vel - link_lin_vel_ref  # (batch_size, num_links, 3)
    link_lin_vel_square = torch.sum(torch.square(link_lin_vel_diff), dim=-1)  # (batch_size, num_links)
    if masked:
        # (batch_size, num_links_selected)
        link_lin_vel_mask = motion_reference.reference_frame.link_pos_mask[:, 0, reference_cfg.body_ids]
    else:
        link_lin_vel_mask = 1

    # select a subset of link_of_interests
    link_lin_vel_square = link_lin_vel_square[:, reference_cfg.body_ids]  # (batch_size, num_links_selected)
    if combine_method == "prod":
        # (batch_size, num_links_selected) -> (batch_size,)
        link_lin_vel_square = torch.sum(link_lin_vel_square * link_lin_vel_mask, dim=-1)
    if combine_method == "mean_prod":
        # (batch_size, num_links_selected) -> (batch_size,)
        link_lin_vel_square = torch.mean(link_lin_vel_square * link_lin_vel_mask, dim=-1)

    rewards = torch.exp(-link_lin_vel_square / (std * std))

    if combine_method == "sum":
        # (batch_size, num_links_selected) -> (batch_size,)
        rewards = torch.sum(rewards * link_lin_vel_mask, dim=-1)

    return rewards


def link_ang_vel_imitation_gauss(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    reference_cfg: SceneEntityCfg = SceneEntityCfg("motion_reference"),
    in_base_frame: bool = False,
    std: float = 0.4,
    masked: bool = False,
    combine_method: Literal["prod", "sum", "mean_prod"] = "prod",
):
    """
    Reward for link angular velocity imitation using a Gaussian kernel.
    This reward is designed to encourage the agent to match the link angular velocities of the reference motion in EVERY timestep.
    Args:
        env: The environment object.
        asset_cfg: The configuration for the scene entity. Default is "robot".
        reference_cfg: The configuration for the scene entity. Default is "motion_reference".
        in_base_frame: Whether to compute the link angular velocity in the base frame of the robot.
        masked: If True, only compute rewards for links that are not masked in the reference motion.
    """
    motion_reference: MotionReferenceManagerBase = env.scene[reference_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]
    links = asset.find_bodies(motion_reference.cfg.link_of_interests, preserve_order=True)  # type: ignore
    link_indices = links[0]
    links_ang_vel_w = asset.data.body_link_ang_vel_w.torch[:, link_indices]  # (batch_size, num_links, 3)
    if in_base_frame:
        root_quat_w = asset.data.root_quat_w.torch
        root_ang_vel_w = asset.data.root_ang_vel_w.torch
        link_ang_vel = math_utils.quat_apply_inverse(
            root_quat_w.unsqueeze(1).expand(-1, links_ang_vel_w.shape[1], -1),
            links_ang_vel_w - root_ang_vel_w.unsqueeze(1),
        )  # (batch_size, num_links, 3)
        link_ang_vel_ref = motion_reference.reference_frame.link_ang_vel_b[:, 0]
    else:
        link_ang_vel = links_ang_vel_w
        link_ang_vel_ref = motion_reference.reference_frame.link_ang_vel_w[:, 0]
    link_ang_vel_diff = link_ang_vel - link_ang_vel_ref  # (batch_size, num_links, 3)
    link_ang_vel_square = torch.sum(torch.square(link_ang_vel_diff), dim=-1)  # (batch_size, num_links)
    if masked:
        # (batch_size, num_links_selected)
        link_ang_vel_mask = motion_reference.reference_frame.link_rot_mask[:, 0, reference_cfg.body_ids]
    else:
        link_ang_vel_mask = 1

    # select a subset of link_of_interests
    link_ang_vel_square = link_ang_vel_square[:, reference_cfg.body_ids]  # (batch_size, num_links_selected)
    if combine_method == "prod":
        # (batch_size, num_links_selected) -> (batch_size,)
        link_ang_vel_square = torch.sum(link_ang_vel_square * link_ang_vel_mask, dim=-1)
    if combine_method == "mean_prod":
        # (batch_size, num_links_selected) -> (batch_size,)
        link_ang_vel_square = torch.mean(link_ang_vel_square * link_ang_vel_mask, dim=-1)

    rewards = torch.exp(-link_ang_vel_square / (std * std))

    if combine_method == "sum":
        # (batch_size, num_links_selected) -> (batch_size,)
        rewards = torch.sum(rewards * link_ang_vel_mask, dim=-1)

    return rewards
