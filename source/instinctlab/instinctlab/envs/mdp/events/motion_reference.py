from __future__ import annotations

import math
import torch
import tqdm
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg

from instinctlab.motion_reference.motion_reference_hoi_data import HoiMotionReferenceData, HoiMotionReferenceState

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject, RigidObjectCollection
    from isaaclab.envs import ManagerBasedEnv

    from instinctlab.envs.mdp import BeyondMimicAdaptiveWeighting
    from instinctlab.motion_reference import MotionReferenceData
    from instinctlab.motion_reference.motion_reference_manager import MotionReferenceManagerBase


def match_motion_ref_with_scene(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    motion_ref_cfg: SceneEntityCfg = SceneEntityCfg("motion_reference"),
):
    """Match information between motion reference and the scene, use at 'startup' stage."""
    print("[Event] Match motion reference with scene.")
    motion_ref: MotionReferenceManagerBase = env.scene[motion_ref_cfg.name]
    motion_ref.match_scene(env.scene)


def reset_robot_state_by_reference(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    motion_ref_cfg: SceneEntityCfg = SceneEntityCfg("motion_reference"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    position_offset: list = [0.0, 0.0, 0.1],
    dof_vel_ratio: float = 0.5,
    base_lin_vel_ratio: float = 0.5,
    base_ang_vel_ratio: float = 0.5,
    randomize_pose_range: dict[str, tuple[float, float]] = {},
    randomize_velocity_range: dict[str, tuple[float, float]] = {},
    randomize_joint_pos_range: tuple[float, float] = (0.0, 0.0),
):
    """Reset robot state based on motion reference with optional randomization.

    Args:
        env: The environment instance.
        env_ids: Environment IDs to reset.
        motion_ref_cfg: Motion reference configuration.
        asset_cfg: Robot asset configuration.
        position_offset: Position offset to apply [x, y, z].
        dof_vel_ratio: Ratio to scale joint velocities.
        base_lin_vel_ratio: Ratio to scale base linear velocities.
        base_ang_vel_ratio: Ratio to scale base angular velocities.
        randomize_pose_range: Optional pose randomization ranges for ["x", "y", "z", "roll", "pitch", "yaw"].
        randomize_velocity_range: Optional velocity randomization ranges for ["x", "y", "z", "roll", "pitch", "yaw"].
        randomize_joint_pos_range: Optional joint position randomization range (min, max).
    """
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    motion_ref: MotionReferenceManagerBase = env.scene[motion_ref_cfg.name]

    # reset the motion reference object
    # motion reference (as sensor) is already reset(ed) in scene.reset(...)
    # motion_ref.reset(env_ids)

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

        # Add position randomization
        base_pos_w += rand_samples[:, 0:3]

        # Add orientation randomization
        orientations_delta = math_utils.quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
        base_quat_w = math_utils.quat_mul(orientations_delta, motion_ref_init_state.base_quat_w)
    else:
        base_quat_w = motion_ref_init_state.base_quat_w

    # write the root pose to the simulation
    asset.write_root_pose_to_sim_index(
        root_pose=torch.cat(
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

        # Add velocity randomization
        base_lin_vel_w += vel_samples[:, 0:3]
        base_ang_vel_w += vel_samples[:, 3:6]

    # write the root velocity to the simulation
    asset.write_root_velocity_to_sim_index(
        root_velocity=torch.cat(
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
        joint_pos += joint_pos_noise

    # write the joint state to the simulation
    asset.write_joint_state_to_sim_index(
        position=joint_pos,
        velocity=joint_vel,
        env_ids=env_ids,
    )


def _apply_rigid_object_states(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    object_pos: torch.Tensor,
    object_quat: torch.Tensor,
    object_lin_vel: torch.Tensor,
    object_ang_vel: torch.Tensor,
    object_validity: torch.Tensor,
    scene_object_names: list[str],
    rigid_object_collection_cfg: SceneEntityCfg | None,
    invalid_object_pos: tuple[float, float, float] | None = None,
):
    """Helper to apply rigid object states to simulation.

    Args:
        env: The environment instance.
        env_ids: Environment IDs to update.
        object_pos: Object positions [N_env, N_obj, 3].
        object_quat: Object orientations [N_env, N_obj, 4].
        object_lin_vel: Object linear velocities [N_env, N_obj, 3].
        object_ang_vel: Object angular velocities [N_env, N_obj, 3].
        object_validity: Object validity mask [N_env, N_obj].
        scene_object_names: List of object names in the scene.
        rigid_object_collection_cfg: Configuration for rigid object collection (optional).
        invalid_object_pos: If not None, set invalid objects to this position (e.g. to hide them).
            If None, invalid objects are not updated.
    """
    from isaaclab.assets import RigidObject

    # Ensure env_ids is on the same device as data
    if env_ids.device != object_pos.device:
        env_ids = env_ids.to(object_pos.device)

    device = object_pos.device
    identity_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device, dtype=object_quat.dtype)
    zero_vel = torch.zeros(3, device=device, dtype=object_lin_vel.dtype)

    if rigid_object_collection_cfg is not None:
        collection: RigidObjectCollection = env.scene[rigid_object_collection_cfg.name]
        # Iterate over objects in the collection
        num_objects = min(object_validity.shape[1], len(scene_object_names), collection.num_objects)

        for obj_idx in range(num_objects):
            valid_mask = object_validity[:, obj_idx]

            # Update valid objects
            if valid_mask.any():
                valid_env_ids = env_ids[valid_mask].contiguous()

                # Prepare pose and velocity
                root_pose = torch.cat(
                    [object_pos[valid_mask, obj_idx], object_quat[valid_mask, obj_idx]],
                    dim=-1,
                )
                root_velocity = torch.cat(
                    [object_lin_vel[valid_mask, obj_idx], object_ang_vel[valid_mask, obj_idx]],
                    dim=-1,
                )

                collection.write_body_link_pose_to_sim_index(
                    body_poses=root_pose.unsqueeze(1), env_ids=valid_env_ids, body_ids=[obj_idx]
                )
                collection.write_body_link_velocity_to_sim_index(
                    body_velocities=root_velocity.unsqueeze(1), env_ids=valid_env_ids, body_ids=[obj_idx]
                )

            # Optionally set invalid objects to a fixed position
            if invalid_object_pos is not None and (~valid_mask).any():
                invalid_env_ids = env_ids[~valid_mask].contiguous()
                n_invalid = invalid_env_ids.shape[0]
                invalid_pos = (
                    torch.tensor(invalid_object_pos, device=device, dtype=object_pos.dtype)
                    .unsqueeze(0)
                    .expand(n_invalid, 3)
                )
                invalid_quat = identity_quat.unsqueeze(0).expand(n_invalid, 4)
                invalid_lin_vel = zero_vel.unsqueeze(0).expand(n_invalid, 3)
                invalid_ang_vel = zero_vel.unsqueeze(0).expand(n_invalid, 3)
                root_pose = torch.cat([invalid_pos, invalid_quat], dim=-1)
                root_velocity = torch.cat([invalid_lin_vel, invalid_ang_vel], dim=-1)
                collection.write_body_link_pose_to_sim_index(
                    body_poses=root_pose.unsqueeze(1), env_ids=invalid_env_ids, body_ids=[obj_idx]
                )
                collection.write_body_link_velocity_to_sim_index(
                    body_velocities=root_velocity.unsqueeze(1), env_ids=invalid_env_ids, body_ids=[obj_idx]
                )
        return

    # Individual objects case
    scene_keys = set(env.scene.keys())
    for obj_idx, entity_name in enumerate(scene_object_names):
        if obj_idx >= object_validity.shape[1]:
            break

        # Use scene.keys() for membership check; env.scene[key] raises KeyError for invalid keys
        # (e.g. numeric strings like '0' from object index slots that are not scene entity names)
        if entity_name not in scene_keys:
            continue
        asset = env.scene[entity_name]
        if not isinstance(asset, RigidObject):
            continue

        valid_mask = object_validity[:, obj_idx]

        # Update valid objects
        if valid_mask.any():
            valid_env_ids = env_ids[valid_mask].contiguous()

            root_pose = torch.cat(
                [object_pos[valid_mask, obj_idx], object_quat[valid_mask, obj_idx]],
                dim=-1,
            )
            root_velocity = torch.cat(
                [object_lin_vel[valid_mask, obj_idx], object_ang_vel[valid_mask, obj_idx]],
                dim=-1,
            )

            asset.write_root_pose_to_sim_index(root_pose=root_pose, env_ids=valid_env_ids)
            asset.write_root_velocity_to_sim_index(root_velocity=root_velocity, env_ids=valid_env_ids)

        # Optionally set invalid objects to a fixed position
        if invalid_object_pos is not None and (~valid_mask).any():
            invalid_env_ids = env_ids[~valid_mask].contiguous()
            n_invalid = invalid_env_ids.shape[0]
            invalid_pos = (
                torch.tensor(invalid_object_pos, device=device, dtype=object_pos.dtype)
                .unsqueeze(0)
                .expand(n_invalid, 3)
            )
            invalid_quat = identity_quat.unsqueeze(0).expand(n_invalid, 4)
            invalid_lin_vel = zero_vel.unsqueeze(0).expand(n_invalid, 3)
            invalid_ang_vel = zero_vel.unsqueeze(0).expand(n_invalid, 3)
            root_pose = torch.cat([invalid_pos, invalid_quat], dim=-1)
            root_velocity = torch.cat([invalid_lin_vel, invalid_ang_vel], dim=-1)
            asset.write_root_pose_to_sim_index(root_pose=root_pose, env_ids=invalid_env_ids)
            asset.write_root_velocity_to_sim_index(root_velocity=root_velocity, env_ids=invalid_env_ids)


def reset_rigid_objects_state_by_reference(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    motion_ref_cfg: SceneEntityCfg = SceneEntityCfg("motion_reference"),
    rigid_object_collection_cfg: SceneEntityCfg | None = None,
):
    """Reset rigid object states in scene from HOI init reference state.

    This function intentionally uses object index order and ignores object-name matching.
    """
    motion_ref: MotionReferenceManagerBase = env.scene[motion_ref_cfg.name]
    init_state = motion_ref.get_init_reference_state(env_ids)

    if not isinstance(init_state, HoiMotionReferenceState):
        return

    scene_object_names = getattr(motion_ref.cfg, "scene_object_names", None)
    if not scene_object_names:
        return

    # init_state is already indexed by env_ids (get_init_reference_state returns state[env_ids]),
    # so its tensors have shape [len(env_ids), ...] with row i mapping to env_ids[i].
    # Do NOT index again with env_ids—that treats env IDs (e.g. 2) as positional indices
    # and causes out-of-bounds when a subset of envs reset (e.g. env_ids=[2] → tensor has 1 row).
    env_ids_t = torch.as_tensor(env_ids, device=init_state.object_pos_w.device)

    _apply_rigid_object_states(
        env,
        env_ids_t,
        init_state.object_pos_w,
        init_state.object_quat_w,
        init_state.object_lin_vel_w,
        init_state.object_ang_vel_w,
        init_state.object_validity,
        scene_object_names,
        rigid_object_collection_cfg,
    )


def update_rigid_objects_state_by_reference(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    motion_ref_cfg: SceneEntityCfg = SceneEntityCfg("motion_reference"),
    rigid_object_collection_cfg: SceneEntityCfg | None = None,
    invalid_object_pos: tuple[float, float, float] | None = None,
):
    """Set rigid object states from the current HOI reference frame every step.

    This function uses object index order and ignores object-name matching.

    Args:
        invalid_object_pos: If not None, set invalid (non-visible) objects to this position
            (e.g. to hide them far from the scene). If None, invalid objects are not updated.
    """
    motion_ref: MotionReferenceManagerBase = env.scene[motion_ref_cfg.name]
    data: HoiMotionReferenceData = motion_ref.data
    scene_object_names = getattr(motion_ref.cfg, "scene_object_names", None)
    if not scene_object_names:
        return

    if not all(
        hasattr(data, attr)
        for attr in ("object_pos_w", "object_quat_w", "object_lin_vel_w", "object_ang_vel_w", "object_validity")
    ):
        return

    # Always use all env ids to avoid block index / out-of-bounds when interval event passes a subset.
    # Object pose updates must run for all envs to keep motion reference in sync.
    num_envs = env.scene.num_envs
    all_env_ids = torch.arange(num_envs, device=data.object_pos_w.torch.device, dtype=torch.long)

    # data shape is [N, T, O, ...]; use the first target frame.
    _apply_rigid_object_states(
        env,
        all_env_ids,
        data.object_pos_w[:, 0],
        data.object_quat_w[:, 0],
        data.object_lin_vel_w[:, 0],
        data.object_ang_vel_w[:, 0],
        data.object_validity[:, 0],
        scene_object_names,
        rigid_object_collection_cfg,
        invalid_object_pos,
    )


def beyondmimic_bin_fail_counter_smoothing(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    curriculum_name: str,
):
    # Acquire the curriculum term instance, which should be a ManagerTermBase
    curriculum: BeyondMimicAdaptiveWeighting = env.curriculum_manager._term_cfgs[
        env.curriculum_manager._term_names.index(curriculum_name)
    ].func

    curriculum.motion_bin_fail_counter._concatenated_tensor.mul_(1 - curriculum.adaptive_alpha)
    curriculum.motion_bin_fail_counter._concatenated_tensor.add_(
        curriculum.adaptive_alpha * curriculum.current_motion_bin_fail_counter._concatenated_tensor
    )
    curriculum.current_motion_bin_fail_counter._concatenated_tensor.fill_(0)
