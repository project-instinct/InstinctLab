from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Sequence

import isaaclab.utils.math as math_utils
from isaaclab.managers import ManagerTermBase, ObservationTermCfg, SceneEntityCfg

import instinctlab.utils.math as instinct_math

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedEnv

    from instinctlab.motion_reference.motion_reference_manager import MotionReferenceManagerBase


class base_pos_offset_reference_as_state(ManagerTermBase):
    """Return the shorten base position. Aka. the position *offset* at the reference position when the
    motion reference data is refreshed.
    """

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        asset_cfg = cfg.params.get("asset_cfg", SceneEntityCfg("motion_reference"))
        self.motion_reference: MotionReferenceManagerBase = env.scene[asset_cfg.name]
        self.base_pos_marker = torch.zeros_like(self.motion_reference.reference_frame.base_pos_w[:, 0])  # (num_envs, 3)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self.base_pos_marker[env_ids] = self.motion_reference.reference_frame.base_pos_w[env_ids, 0]

    def __call__(
        self,
        env: ManagerBasedEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("motion_reference"),
    ) -> torch.Tensor:
        # refresh the base position landmarker when the motion reference data is refreshed
        # (num_envs, 3)
        base_pos_w = self.motion_reference.reference_frame.base_pos_w[:, 0]
        # refresh the landmarker if the refreshed time is smaller than env.step_dt
        landmarker_refresh_mask = self.motion_reference.time_passed_from_update < env.step_dt
        self.base_pos_marker[landmarker_refresh_mask] = base_pos_w[landmarker_refresh_mask]

        # (num_envs, 3)
        base_pos_w_offset = base_pos_w - self.base_pos_marker
        return base_pos_w_offset  # (num_envs, 3)


def base_pos_z_reference_as_state(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("motion_reference"),
):
    """The height of the root link. (typically used in flat ground scenarios)
    Returns:
        (num_envs, 1)
    """
    motion_reference: MotionReferenceManagerBase = env.scene[asset_cfg.name]
    root_pos_w = motion_reference.reference_frame.base_pos_w[:, 0]
    return root_pos_w[:, 2:3]


def base_lin_vel_reference_as_state(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("motion_reference"),
):
    """The linear velocity of the root link in the robot base frame as state.
    Returns:
        (num_envs, 3)
    """
    motion_reference: MotionReferenceManagerBase = env.scene[asset_cfg.name]
    base_quat_w = motion_reference.reference_frame.base_quat_w[:, 0]
    base_lin_vel_w = motion_reference.reference_frame.base_lin_vel_w[:, 0]
    base_lin_vel_b = math_utils.quat_apply_inverse(base_quat_w, base_lin_vel_w)
    return base_lin_vel_b


def base_ang_vel_reference_as_state(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("motion_reference"),
):
    """The angular velocity of the root link in the robot base frame as state.
    Returns:
        (num_envs, 3)
    """
    motion_reference: MotionReferenceManagerBase = env.scene[asset_cfg.name]
    base_quat_w = motion_reference.reference_frame.base_quat_w[:, 0]
    base_ang_vel_w = motion_reference.reference_frame.base_ang_vel_w[:, 0]
    base_ang_vel_b = math_utils.quat_apply_inverse(base_quat_w, base_ang_vel_w)
    return base_ang_vel_b


def heading_reference_as_state(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("motion_reference"),
):
    """The heading of the root link in the robot base frame as state.
    Returns:
        (num_envs, 1)
    """
    motion_reference: MotionReferenceManagerBase = env.scene[asset_cfg.name]
    root_quat_w = motion_reference.reference_frame.base_quat_w[:, 0]
    root_heading_w = math_utils.wrap_to_pi(math_utils.euler_xyz_from_quat(root_quat_w)[2])
    return root_heading_w


def root_tannorm_reference_as_state(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("motion_reference"),
):
    """The orientation of the root link in tangent-normal representation.
    Returns:
        (num_envs, 6)
    """
    motion_reference: MotionReferenceManagerBase = env.scene[asset_cfg.name]
    root_quat_w = motion_reference.reference_frame.base_quat_w[:, 0]
    root_tannorm_w = instinct_math.quat_to_tan_norm(root_quat_w)
    return root_tannorm_w


class projected_gravity_reference_as_state(ManagerTermBase):
    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        self.motion_ref: MotionReferenceManagerBase = (
            env.scene[cfg.params["asset_cfg"].name] if "asset_cfg" in cfg.params else env.scene["motion_reference"]
        )

        gravity_dir = torch.tensor(env.cfg.sim.gravity, device=self.device)
        gravity_dir = math_utils.normalize(gravity_dir.unsqueeze(0)).squeeze(0)

        self.GRAVITY_VEC_W = gravity_dir.repeat(len(self.motion_ref.ALL_INDICES), 1)

    def __call__(
        self,
        env: ManagerBasedEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("motion_reference"),
    ):
        """The projected gravity in the robot base frame as state.
        Returns:
            (num_envs, 3)
        """
        motion_reference: MotionReferenceManagerBase = env.scene[asset_cfg.name]
        root_quat_w = motion_reference.reference_frame.base_quat_w[:, 0]
        projected_gravity_w = math_utils.quat_apply_inverse(root_quat_w, self.GRAVITY_VEC_W)
        return projected_gravity_w


def joint_pos_reference_as_state(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("motion_reference"),
):
    """Joint positions in the robot base frame as state, but from motion reference"""
    motion_reference: MotionReferenceManagerBase = env.scene[asset_cfg.name]
    return (
        motion_reference.reference_frame.joint_pos[:, 0, asset_cfg.joint_ids]
        * motion_reference.reference_frame.joint_pos_mask[:, 0, asset_cfg.joint_ids]
    )


def joint_pos_rel_reference_as_state(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("motion_reference"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Joint positions (relative to robot's default joint pos) in the robot base frame as state, but from motion reference
    Returns:
        (num_envs, num_joints)
    """
    motion_reference: MotionReferenceManagerBase = env.scene[asset_cfg.name]
    joint_pos = motion_reference.reference_frame.joint_pos[:, 0, asset_cfg.joint_ids]
    robot: Articulation = env.scene[robot_cfg.name]
    joint_pos_rel = joint_pos - robot.data.default_joint_pos.torch[:, asset_cfg.joint_ids]
    return joint_pos_rel * motion_reference.reference_frame.joint_pos_mask[:, 0, asset_cfg.joint_ids]


def joint_pos_err_reference_as_state(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("motion_reference"),
    state_as_last_reference: bool = False,
    mask: bool = True,
):
    """Joint position error in the robot base frame as state, but from motion reference.
    It is the complementary observation as from JointPosErrRefCommand.
    Args:
        state_as_last_reference: If True, the current state is added to the last frame as the reference.
    """
    motion_reference: MotionReferenceManagerBase = env.scene[asset_cfg.name]
    joint_pos_b = motion_reference.reference_frame.joint_pos[:, 0, asset_cfg.joint_ids]  # (num_envs, num_joints)
    joint_pos_ref_b = motion_reference.data.joint_pos[:, :, asset_cfg.joint_ids]

    if state_as_last_reference:
        joint_pos_ref_b = torch.cat(
            [
                joint_pos_ref_b,
                joint_pos_b.unsqueeze(1),
            ],
            dim=1,
        )

    joint_pos_err = joint_pos_ref_b - joint_pos_b.unsqueeze(1)
    num_reference_frames = motion_reference.num_frames
    if mask:
        joint_pos_err[:, :num_reference_frames] *= motion_reference.reference_frame.joint_pos_mask[
            :, :, asset_cfg.joint_ids
        ]
    joint_pos_err[:, :num_reference_frames] *= motion_reference.data.validity.unsqueeze(-1)
    return joint_pos_err  # (num_envs, num_frames, num_joints)


def joint_vel_reference_as_state(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("motion_reference"),
    mask: bool = True,
):
    """Joint velocities in the robot base frame as state, but from motion reference"""
    motion_reference: MotionReferenceManagerBase = env.scene[asset_cfg.name]
    joint_vel = motion_reference.reference_frame.joint_vel[:, 0, asset_cfg.joint_ids]
    if mask:
        joint_vel *= motion_reference.reference_frame.joint_vel_mask[:, 0, asset_cfg.joint_ids]
    return joint_vel


def joint_vel_rel_reference_as_state(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("motion_reference"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    mask: bool = True,
):
    """Joint velocities (relative to robot's default joint vel) in the robot base frame as state, but from motion reference
    Returns:
        (num_envs, num_joints)
    """
    motion_reference: MotionReferenceManagerBase = env.scene[asset_cfg.name]
    joint_vel = motion_reference.reference_frame.joint_vel[:, 0, asset_cfg.joint_ids]
    if mask:
        joint_vel *= motion_reference.reference_frame.joint_vel_mask[:, 0, asset_cfg.joint_ids]
    robot: Articulation = env.scene[robot_cfg.name]
    joint_vel_rel = joint_vel - robot.data.default_joint_vel.torch[:, asset_cfg.joint_ids]
    return joint_vel_rel


def link_pos_reference_as_state(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("motion_reference"),
):
    """Link positions in the robot base frame as state, but from motion reference"""
    motion_reference: MotionReferenceManagerBase = env.scene[asset_cfg.name]
    return motion_reference.reference_frame.link_pos_b[
        :, 0, asset_cfg.body_ids
    ] * motion_reference.reference_frame.link_pos_mask[:, 0, asset_cfg.body_ids].unsqueeze(-1)


def link_pos_err_reference_as_state(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("motion_reference"),
    state_as_last_reference: bool = False,
):
    """Link position error in the robot base frame as state, but from motion reference.
    It is the complementary observation as from LinkPosErrRefCommand.
    Args:
        state_as_last_reference: If True, the current state is added to the last frame as the reference.
    """
    motion_reference: MotionReferenceManagerBase = env.scene[asset_cfg.name]
    link_pos_b = motion_reference.reference_frame.link_pos_b[:, 0, asset_cfg.body_ids]  # (num_envs, num_links, 3)
    link_pos_ref_b = motion_reference.data.link_pos_b[:, :, asset_cfg.body_ids]

    if state_as_last_reference:
        link_pos_ref_b = torch.cat(
            [
                link_pos_ref_b,
                link_pos_b.unsqueeze(1),
            ],
            dim=1,
        )

    link_pos_err = link_pos_ref_b - link_pos_b.unsqueeze(1)
    # NOTE: motion_reference.reference_frame.link_pos_mask is not used here.
    num_reference_frames = motion_reference.num_frames
    link_pos_err[:, :num_reference_frames] *= motion_reference.data.link_pos_mask[:, :, asset_cfg.body_ids].unsqueeze(
        -1
    )
    link_pos_err[:, :num_reference_frames] *= motion_reference.data.validity.unsqueeze(-1).unsqueeze(-1)
    return link_pos_err  # (num_envs, num_frames, num_links, 3)


def link_quat_reference_as_state(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("motion_reference"),
):
    """Link quaternion in the robot base frame as state, but from motion reference"""
    motion_reference: MotionReferenceManagerBase = env.scene[asset_cfg.name]
    return motion_reference.reference_frame.link_quat_b[
        :, 0, asset_cfg.body_ids
    ] * motion_reference.reference_frame.link_rot_mask[:, 0, asset_cfg.body_ids].unsqueeze(-1)


def link_tannorm_reference_as_state(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("motion_reference"),
):
    """Link orientations in tangent-normal in the robot base frame as state, but from motion reference"""
    motion_reference: MotionReferenceManagerBase = env.scene[asset_cfg.name]
    link_quat_b = motion_reference.reference_frame.link_quat_b[:, 0, asset_cfg.body_ids]
    link_tannorm_b = instinct_math.quat_to_tan_norm(link_quat_b)
    return link_tannorm_b * motion_reference.reference_frame.link_rot_mask[:, 0, asset_cfg.body_ids].unsqueeze(-1)


def link_tannorm_err_reference_as_state(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("motion_reference"),
    in_base_frame: bool = True,
    state_as_last_reference: bool = False,
):
    """Link orientation error in tangent-normal in the robot base frame as state, but from motion reference.
    It is the complementary observation as from LinkRotErrRefCommand.
    NOTE: Make sure the rotation mode is the same as the config in your LinkRotErrRefCommandCfg.
    Args:
        state_as_last_reference: If True, the current state is added to the last frame as the reference.
    """
    motion_reference: MotionReferenceManagerBase = env.scene[asset_cfg.name]
    link_quat_w = motion_reference.reference_frame.link_quat_w[:, 0, asset_cfg.body_ids]
    if in_base_frame:
        link_quat_ref_ = motion_reference.data.link_quat_b[:, :, asset_cfg.body_ids]
        link_quat_ = math_utils.quat_mul(
            math_utils.quat_inv(motion_reference.reference_frame.base_quat_w).expand(
                -1, link_quat_w.shape[1], -1
            ),  # No unsqueeze here because the time dimension can be reused as num_link dimension
            link_quat_w,
        )
    else:
        link_quat_ref_ = motion_reference.data.link_quat_w[:, :, asset_cfg.body_ids]
        link_quat_ = link_quat_w
    if state_as_last_reference:
        link_quat_ref_ = torch.cat(
            [
                link_quat_ref_,
                link_quat_.unsqueeze(1),
            ],
            dim=1,
        )
    link_quat_err = math_utils.quat_mul(
        math_utils.quat_inv(link_quat_).unsqueeze(1).expand(-1, link_quat_ref_.shape[1], -1, -1),
        link_quat_ref_,
    )
    link_tannorm_err = instinct_math.quat_to_tan_norm(link_quat_err)
    # NOTE: motion_reference.reference_frame.link_rot_mask is not used here.
    num_reference_frames = motion_reference.num_frames
    link_tannorm_err[:, :num_reference_frames] *= motion_reference.data.link_rot_mask[
        :, :, asset_cfg.body_ids
    ].unsqueeze(-1)
    link_tannorm_err[:, :num_reference_frames] *= motion_reference.data.validity.unsqueeze(-1).unsqueeze(-1)
    return link_tannorm_err  # (num_envs, num_frames, num_links, 6)
