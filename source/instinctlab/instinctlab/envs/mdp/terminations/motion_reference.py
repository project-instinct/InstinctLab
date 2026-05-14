from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.managers import ManagerTermBase, SceneEntityCfg

import instinctlab.motion_reference.utils as motion_reference_utils

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv

    from instinctlab.motion_reference import MotionReferenceManager


def pos_far_from_ref(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    reference_cfg: SceneEntityCfg = SceneEntityCfg("motion_reference"),
    distance_threshold: float = 0.5,
    check_at_keyframe_threshold: float = 0.01,
    print_reason: bool = False,
    height_only: bool = False,
) -> torch.Tensor:
    """Check if the position is far from the reference position.

    Args:
        env: The environment object.
        distance_threshold: The distance threshold.
        check_at_keyframe: only check when the time is close to the keyframe.
        height_only: only check the height (z-axis) difference.

    Returns:
        True if the position is far from the reference position, False otherwise.
    """
    distance = motion_reference_utils.get_base_position_distance(
        env,
        asset_cfg,
        reference_cfg,
        masking=True,
        squared=False,
        return_diff=height_only,
    )
    if height_only:
        distance = distance[..., 2].abs()  # only check z-axis
    distance = motion_reference_utils.matching_reference_timing(
        env,
        distance,
        env.scene[reference_cfg.name],
        check_at_keyframe_threshold=check_at_keyframe_threshold,
        multiply_by_frame_interval=False,
    )

    return_ = distance > distance_threshold
    if print_reason and return_.any():
        motion_reference: MotionReferenceManager = env.scene[reference_cfg.name]
        env_ids = torch.where(return_)[0]
        motion_identifiers = motion_reference.get_current_motion_identifiers(env_ids=env_ids)
        print(f"pos_far_from_ref: {return_.sum()} at {motion_identifiers}")
    return return_


def bad_global_body_pos(
    env: ManagerBasedRLEnv,
    command_name: str = "motion_reference",
    threshold: float = 0.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    body_names: list[str] | None = None,
    disable_flag: bool = False,
) -> torch.Tensor:
    """Terminate if any tracked body is too far from reference in world frame.

    InstinctLab counterpart of ScaleTrack's ``bad_global_body_pos``:
    compute :math:`||p_ref^w - p_robot^w||` per body and terminate if any exceeds threshold.

    Args:
        env: The environment object.
        command_name: Scene entity name of the motion reference manager.
        threshold: Per-body Euclidean distance threshold in meters.
        asset_cfg: Robot articulation.
        body_names: Optional subset of body names to check. If None, use the motion reference's body_names.
        disable_flag: If True, always return False (no termination).
    """
    if disable_flag:
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)

    motion_ref: MotionReferenceManager = env.scene[command_name]
    robot: Articulation = env.scene[asset_cfg.name]

    names = body_names if body_names is not None else motion_ref.body_names
    name_to_ref_idx = {name: i for i, name in enumerate(motion_ref.body_names)}
    try:
        ref_indices = [name_to_ref_idx[name] for name in names]
    except KeyError as exc:
        missing = exc.args[0] if exc.args else exc
        raise ValueError(
            f"Requested body name {missing!r} not found in motion reference body_names."
        ) from exc
    body_ids, _ = robot.find_bodies(names, preserve_order=True)

    ref_pos_w = motion_ref.reference_frame.link_pos_w[:, 0, ref_indices]
    robot_pos_w = robot.data.body_link_pos_w[:, body_ids]

    error = torch.norm(ref_pos_w - robot_pos_w, dim=-1)
    return torch.any(error > threshold, dim=-1)


class joint_pos_far_from_ref(ManagerTermBase):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.exclude_joint_indices = []
        if cfg.params.get("exclude_joint_names", []):
            robot_asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
            for joint_name in cfg.params["exclude_joint_names"]:
                self.exclude_joint_indices += robot_asset.find_joints(joint_name)[0]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        reference_cfg: SceneEntityCfg = SceneEntityCfg("motion_reference"),
        difference_threshold: float = 0.5,
        check_at_keyframe_threshold: float = 0.01,
        exclude_joint_names=[],  # list of joint names to exclude
        print_reason: bool = False,
    ) -> torch.Tensor:
        """Check if the joint position is far from the reference joint position."""
        difference_ = motion_reference_utils.get_joint_position_difference(
            env,
            asset_cfg,
            reference_cfg,
            masking=True,
        )

        # set exclude joints to zero
        if self.exclude_joint_indices:
            difference_[:, self.exclude_joint_indices] = 0.0

        difference = motion_reference_utils.matching_reference_timing(
            env,
            torch.abs(difference_).max(-1).values,
            env.scene[reference_cfg.name],
            check_at_keyframe_threshold=check_at_keyframe_threshold,
            multiply_by_frame_interval=False,
        )
        return_ = torch.abs(difference) > difference_threshold
        if print_reason and return_.any():
            print(
                "joint_pos_far_from_ref: ",
                return_.sum(),
                "max diff: ",
                torch.abs(difference_).max(),
                "at joint:",
                difference_.argmax(),
            )
        return return_


def link_pos_far_from_ref(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    reference_cfg: SceneEntityCfg = SceneEntityCfg("motion_reference"),
    distance_threshold: float = 0.5,
    in_base_frame: bool = False,
    check_at_keyframe_threshold: float = 0.01,
    height_only: bool = False,  # recommend using with `in_base_frame=False`
    print_reason: bool = False,
) -> torch.Tensor:
    distance = motion_reference_utils.get_link_position_distance(
        env,
        asset_cfg,
        reference_cfg,
        masking=True,
        squared=False,
        in_base_frame=in_base_frame,
        return_diff=height_only,
    )  # (batch_size, num_links)
    if height_only:
        distance = distance[..., 2].abs()  # only check z-axis
    distance = distance.max(-1).values
    distance = motion_reference_utils.matching_reference_timing(
        env,
        distance,
        env.scene[reference_cfg.name],
        check_at_keyframe_threshold=check_at_keyframe_threshold,
        multiply_by_frame_interval=False,
    )

    return_ = distance > distance_threshold
    if print_reason and return_.any():
        motion_reference: MotionReferenceManager = env.scene[reference_cfg.name]
        env_ids = torch.where(return_)[0]
        motion_identifiers = motion_reference.get_current_motion_identifiers(env_ids=env_ids)
        print(f"link_pos_far_from_ref: {return_.sum()} at {motion_identifiers}")
    return return_


def rot_far_from_ref(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    reference_cfg: SceneEntityCfg = SceneEntityCfg("motion_reference"),
    rot_axis_threshold: float = 1.5714 / 2,
    check_at_keyframe_threshold: float = 0.01,
    print_reason: bool = False,
) -> torch.Tensor:
    """Check if the rotation is far from the reference rotation.

    Args:
        env: The environment object.
        rot_axis_threshold: difference threshold in terms of the rotation axis norm.
        check_at_keyframe: only check when the timt is close to the keyframe.

    Returns:
        True if the rotation is far from the reference rotation, False otherwise.
    """
    robot: Articulation = env.scene[asset_cfg.name]
    motion_reference: MotionReferenceManager = env.scene[reference_cfg.name]

    # get the current rotation
    rot = robot.data.root_state_w[:, 3:7]  # shape: [N, 4]
    ref_rot = motion_reference.data.base_quat_w  # shape: [N, n_frames, 4]
    ref_rot = ref_rot[motion_reference.ALL_INDICES, motion_reference.aiming_frame_idx]

    # calculate the axis-angle norm as the difference
    rot_diff = math_utils.quat_mul(math_utils.quat_inv(ref_rot), rot)  # shape: [N, 4]
    rot_diff_axis = math_utils.axis_angle_from_quat(rot_diff)  # shape: [N, 3]
    rot_diff_norm = torch.norm(rot_diff_axis, dim=-1)  # shape: [N,]

    if check_at_keyframe_threshold >= 0:
        near_keyframe = (motion_reference.time_to_aiming_frame <= check_at_keyframe_threshold).float()
        rot_diff_norm = rot_diff_norm * near_keyframe

    return_ = rot_diff_norm > rot_axis_threshold
    if print_reason and return_.any():
        print("rot_far_from_ref: ", return_.sum())
    return return_


def projected_gravity_far_from_ref(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    reference_cfg: SceneEntityCfg = SceneEntityCfg("motion_reference"),
    projected_gravity_threshold: float = 0.5,
    check_at_keyframe_threshold: float = 0.01,
    z_only: bool = True,
    print_reason: bool = False,
) -> torch.Tensor:
    """Check if the projected gravity is far from the reference projected gravity."""

    robot: Articulation = env.scene[asset_cfg.name]
    motion_reference: MotionReferenceManager = env.scene[reference_cfg.name]

    # get the current rotation
    rot = robot.data.root_state_w[:, 3:7]  # shape: [N, 4]
    ref_rot = motion_reference.data.base_quat_w  # shape: [N, n_frames, 4]
    ref_rot = ref_rot[motion_reference.ALL_INDICES, motion_reference.aiming_frame_idx]

    pg = math_utils.quat_apply_inverse(rot, robot.data.GRAVITY_VEC_W)
    ref_pg = math_utils.quat_apply_inverse(ref_rot, robot.data.GRAVITY_VEC_W)

    if z_only:
        diff = (pg[:, 2] - ref_pg[:, 2]).abs()
    else:
        diff = torch.norm(pg - ref_pg, dim=-1)

    if check_at_keyframe_threshold >= 0:
        near_keyframe = (motion_reference.time_to_aiming_frame <= check_at_keyframe_threshold).float()
        diff = diff * near_keyframe

    return_ = diff > projected_gravity_threshold
    if print_reason and return_.any():
        env_ids = torch.where(return_)[0]
        motion_identifiers = motion_reference.get_current_motion_identifiers(env_ids=env_ids)
        print(f"projected_gravity_far_from_ref: {return_.sum()} at {motion_identifiers}")
    return return_
