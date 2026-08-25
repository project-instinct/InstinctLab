from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Literal

import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv

    from instinctlab.motion_reference.motion_reference_manager import MotionReferenceManagerBase


def reference_progress(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("motion_reference"),
) -> torch.Tensor:
    """The progress of the reference motion, from 0 to 1.
    Return shape: (num_envs, 1)
    """
    motion_reference: MotionReferenceManagerBase = env.scene[asset_cfg.name]
    motion_length = motion_reference.complete_motion_lengths
    timestamp = env.episode_length_buf if hasattr(env, "episode_length_buf") else torch.zeros_like(motion_length)
    timestamp *= env.step_dt
    # (num_envs, 1)
    return (timestamp / motion_length).unsqueeze(-1)


def time_from_reference_update(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("motion_reference"),
    multi_frames: bool = False,
    current_state_mask_at_last: bool = False,
) -> torch.Tensor:
    """Time from the last reference refresh.
    Args:
        multi_frames: whether to expand the last dimension to meet the (N, T, D) shape.
        current_state_mask_at_last: whether to append the current state mask at the last frame, which is
            an all-trues mask. Only effective in multi_frames mode.
    """
    motion_reference: MotionReferenceManagerBase = env.scene[asset_cfg.name]
    return_ = motion_reference.time_passed_from_update.unsqueeze(-1)
    if multi_frames:
        # (num_envs, 1) -> (num_envs, num_frames, 1)
        return_ = return_.unsqueeze(-1).expand(-1, motion_reference.cfg.num_frames, -1)
        if current_state_mask_at_last:
            return_ = torch.cat(
                [
                    return_,
                    torch.zeros_like(return_[:, -1:], dtype=torch.float32),
                ],
                dim=1,
            )
    return return_


def ref_frame_interval(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("motion_reference")
) -> torch.Tensor:
    """A legacy reference time observation.
    Combing motion frame interval (s) and the time to target frame
    """
    motion_reference: MotionReferenceManagerBase = env.scene[asset_cfg.name]
    frame_interval = motion_reference.data.time_to_target_frame[..., :1]
    time_from_reference_update = motion_reference.time_passed_from_update.unsqueeze(-1)
    time_to_target = frame_interval - time_from_reference_update
    return torch.cat(
        [
            frame_interval,
            # I know this value start from frame_interval to negative values...
            # It is not designed well in the isaacgym version.
            time_to_target,
        ],
        dim=-1,
    )  # (num_envs, 2)


def motion_reference_mask(
    env: ManagerBasedEnv,
    data_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("motion_reference"),
    current_state_mask_at_last: bool = False,
) -> torch.Tensor:
    """
    Args:
        current_state_mask_at_last: whether to append the current state mask at the last frame, which is
            an all-trues mask.
    """
    motion_reference: MotionReferenceManagerBase = env.scene[asset_cfg.name]
    return_ = getattr(motion_reference.data, data_name).to(torch.float32)  # (num_envs, num_frames, D)
    if current_state_mask_at_last:
        return_ = torch.cat(
            [
                return_,
                torch.ones_like(return_[:, -1:], dtype=torch.float32),
            ],
            dim=1,
        )
    return return_


def pose_ref_mask(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("motion_reference"),
    current_state_mask_at_last: bool = False,
) -> torch.Tensor:
    """
    Args:
        current_state_mask_at_last: whether to append the current state mask at the last frame, which is
            an all-trues mask.
    """
    motion_reference: MotionReferenceManagerBase = env.scene[asset_cfg.name]
    return_ = torch.stack(
        [
            motion_reference.data.base_pos_plane_mask,
            motion_reference.data.base_pos_height_mask,
            motion_reference.data.base_orientation_mask,
            motion_reference.data.base_heading_mask,
        ],
        dim=-1,
    ).to(
        torch.float
    )  # (num_envs, num_frames, 4)
    if current_state_mask_at_last:
        return_ = torch.cat(
            [
                return_,
                torch.ones_like(return_[:, -1:], dtype=torch.float32),
            ],
            dim=1,
        )
    return return_


def pose_ref_mask_legacy(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("motion_reference")
) -> torch.Tensor:
    motion_reference: MotionReferenceManagerBase = env.scene[asset_cfg.name]
    return torch.stack(
        [
            motion_reference.data.base_pos_plane_mask,
            motion_reference.data.base_pos_height_mask,
            motion_reference.data.base_orientation_mask & motion_reference.data.base_heading_mask,
        ],
        dim=-1,
    ).to(
        torch.float
    )  # (num_envs, num_frames, 3)
