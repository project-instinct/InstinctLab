""" Additinoal common termination functions that are not implemented in isaaclab. """

from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Sequence

from isaaclab.managers import ManagerTermBase, ManagerTermBaseCfg, SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.assets import RigidObject
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors import ContactSensor

    from instinctlab.motion_reference.motion_reference_manager import MotionReferenceManagerBase


def dataset_exhausted(
    env: ManagerBasedRLEnv,
    reference_cfg: SceneEntityCfg = SceneEntityCfg("motion_reference"),
    reset_without_notice: bool = False,
    check_all_frames: bool = False,
    obs_term_to_reset: Sequence[str] = (),
    print_reason: bool = False,
) -> torch.Tensor:
    """Check if the dataset is exhausted.

    Args:
        env: The environment object.
        reset_without_notice: whether to reset the environment without returning True.
        check_all_frames: If True, an environment is marked exhausted when *any* frame in its
            reference data is invalid/out of bounds. If False (default), only the current
            aiming frame is checked.
        obs_term_to_reset: Observation terms whose callable state and history should be reset when
            the dataset is exhausted without notifying the environment. Typically this is only
            provided when ``reset_without_notice`` is true. Each entry must use the
            ``"{observation_group_name}:{observation_term_name}"`` format.
    Returns:
        True if the dataset is exhausted, False otherwise.
    """
    motion_reference: MotionReferenceManagerBase = env.scene[reference_cfg.name]
    if check_all_frames:
        return_ = torch.logical_not(motion_reference.data.validity[motion_reference.ALL_INDICES]).any(
            dim=-1
        )  # shape: [N,]
    else:
        return_ = torch.logical_not(
            motion_reference.data.validity[motion_reference.ALL_INDICES, motion_reference.aiming_frame_idx]
        )  # shape: [N,]
    if print_reason and return_.any():
        print("dataset_exhausted: ", return_.sum())
    if obs_term_to_reset:
        env_ids = return_.nonzero(as_tuple=True)[0]
        for term_entry in obs_term_to_reset:
            group_name, term_name = term_entry.split(":")
            term_index = env.observation_manager._group_obs_term_names[group_name].index(term_name)
            term_cfg = env.observation_manager._group_obs_term_cfgs[group_name][term_index]
            if isinstance(term_cfg.func, ManagerTermBase):
                term_cfg.func.reset(env_ids=env_ids)
            env.observation_manager._group_obs_term_history_buffer[group_name][term_name].reset(batch_ids=env_ids)
    if reset_without_notice:
        motion_reference.reset(env_ids=return_.nonzero(as_tuple=True)[0])
        return_[:] = False
    return return_


def terrain_out_of_bounds(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    distance_buffer: float = 3.0,
    print_reason: bool = False,
) -> torch.Tensor:
    """Terminate when the actor move too close to the edge of the terrain.

    If the actor moves too close to the edge of the terrain, the termination is activated. The distance
    to the edge of the terrain is calculated based on the size of the terrain and the distance buffer.
    """
    if env.scene.cfg.terrain.terrain_type == "plane":
        return torch.zeros(
            (env.num_envs,), device=env.device, dtype=torch.bool
        )  # we have infinite terrain because it is a plane
    elif env.scene.cfg.terrain.terrain_type == "generator":
        # obtain the size of the sub-terrains
        terrain_gen_cfg = env.scene.terrain.cfg.terrain_generator
        grid_width, grid_length = terrain_gen_cfg.size
        n_rows, n_cols = terrain_gen_cfg.num_rows, terrain_gen_cfg.num_cols
        border_width = terrain_gen_cfg.border_width
        # compute the size of the map
        map_width = n_rows * grid_width + 2 * border_width
        map_height = n_cols * grid_length + 2 * border_width

        # extract the used quantities (to enable type-hinting)
        asset: RigidObject = env.scene[asset_cfg.name]

        # check if the agent is out of bounds
        x_out_of_bounds = torch.abs(asset.data.root_pos_w.torch[:, 0]) > 0.5 * map_width - distance_buffer
        y_out_of_bounds = torch.abs(asset.data.root_pos_w.torch[:, 1]) > 0.5 * map_height - distance_buffer
        return_ = torch.logical_or(x_out_of_bounds, y_out_of_bounds)
        if print_reason and return_.any():
            print(f"The base is out of the terrain border:", return_.sum())
        return return_
    else:
        raise ValueError("Received unsupported terrain type, must be either 'plane' or 'generator'.")


def abnormal_lin_vel(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    max_value: float = 40.0,  # [m/s]
    print_reason: bool = False,
):
    asset = env.scene[asset_cfg.name]
    return_ = torch.norm(asset.data.root_lin_vel_w.torch, dim=-1) > max_value
    if print_reason and return_.any():
        print(f"abnormal_lin_vel: terminating {return_.sum().item()} envs")
    return return_


def abnormal_ang_vel(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    max_value: float = 40.0,  # [rad/s]
    print_reason: bool = False,
):
    asset = env.scene[asset_cfg.name]
    return_ = torch.norm(asset.data.root_ang_vel_w.torch, dim=-1) > max_value
    if print_reason and return_.any():
        print(f"abnormal_ang_vel: terminating {return_.sum().item()} envs")
    return return_


def abnormal_joint_vel(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    max_value: float = 40.0,  # [rad/s]
    print_reason: bool = False,
):
    asset = env.scene[asset_cfg.name]
    return_ = torch.any(torch.abs(asset.data.joint_vel.torch) > max_value, dim=-1)
    if print_reason and return_.any():
        print(f"abnormal_joint_vel: terminating {return_.sum().item()} envs")
    return return_


def abnormal_joint_acc(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    max_value: float = 4000.0,  # [rad/s^2]
    print_reason: bool = False,
):
    asset = env.scene[asset_cfg.name]
    return_ = torch.any(torch.abs(asset.data.joint_acc.torch) > max_value, dim=-1)
    if print_reason and return_.any():
        print(f"abnormal_joint_acc: terminating {return_.sum().item()} envs")
    return return_


def abnormal_body_vel(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    max_value: float = 100.0,  # [m/s or rad/s, norm of body spatial velocity]
    print_reason: bool = False,
):
    """Terminate environments whose fastest body spatial-velocity norm is implausibly large."""
    asset = env.scene[asset_cfg.name]
    body_vel = asset.data.body_vel_w.torch
    if asset_cfg.body_ids is not None:
        body_vel = body_vel[:, asset_cfg.body_ids]
    return_ = torch.any(torch.norm(body_vel, dim=-1) > max_value, dim=-1)
    if print_reason and return_.any():
        print(f"abnormal_body_vel: terminating {return_.sum().item()} envs")
    return return_


def abnormal_body_acc(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    max_value: float = 10000.0,  # [m/s^2 or rad/s^2, norm of body spatial acceleration]
    print_reason: bool = False,
):
    """Terminate environments whose fastest body spatial-acceleration norm is implausibly large."""
    asset = env.scene[asset_cfg.name]
    body_acc = asset.data.body_acc_w.torch
    if asset_cfg.body_ids is not None:
        body_acc = body_acc[:, asset_cfg.body_ids]
    return_ = torch.any(torch.norm(body_acc, dim=-1) > max_value, dim=-1)
    if print_reason and return_.any():
        print(f"abnormal_body_acc: terminating {return_.sum().item()} envs")
    return return_


def nan_guard(env: ManagerBasedRLEnv, print_reason: bool = False) -> torch.Tensor:
    """Terminate environments whose canonical scene state contains NaN or Inf.

    The state is obtained from :meth:`InteractiveScene.get_state`, which covers the restorable
    state of articulations, rigid objects, deformable objects, and surface grippers.

    Args:
        env: The environment object.
        print_reason: Whether to print which entities triggered the termination.

    Returns:
        A boolean tensor indicating which environments to terminate.
    """
    terminated = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    scene_state = env.scene.get_state(is_relative=False)

    for entity_group in scene_state.values():
        for entity_name, entity_state in entity_group.items():
            fields = entity_state.items() if isinstance(entity_state, dict) else (("state", entity_state),)
            bad_envs = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
            for field_name, tensor in fields:
                if (
                    not isinstance(tensor, torch.Tensor)
                    or not tensor.is_floating_point()
                    or tensor.ndim < 1
                    or tensor.shape[0] != env.num_envs
                ):
                    continue
                non_finite = ~tensor.reshape(env.num_envs, -1).isfinite().all(dim=1)
                bad_envs |= non_finite
                if print_reason and non_finite.any():
                    print(
                        f"nan_guard: '{entity_name}.{field_name}' has non-finite data in {non_finite.sum().item()} envs"
                    )

            if print_reason and bad_envs.any():
                print(f"nan_guard: '{entity_name}' has non-finite data in {bad_envs.sum().item()} envs")
            terminated |= bad_envs

    if print_reason and terminated.any():
        print(f"nan_guard: terminating {terminated.sum().item()} envs")
    return terminated


class illegal_reset_contact(ManagerTermBase):
    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.threshold = cfg.params["threshold"]
        self.sensor_cfg = cfg.params["sensor_cfg"]
        self.print_reason = cfg.params.get("print_reason", False)
        self.episode_length_threshold = cfg.params.get("episode_length_threshold", 1)
        self.illegal_contact_counter = torch.zeros(env.num_envs, device=env.device, dtype=torch.int)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        threshold: float,
        sensor_cfg: SceneEntityCfg,
        print_reason: bool = False,
        episode_length_threshold: int = 1,
    ) -> torch.Tensor:
        """Timeout if the robot is reset with some undesired penetration with the environment.
        within the first episode_length_threshold steps.
        """
        contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
        net_contact_forces = contact_sensor.data.net_forces_w_history.torch
        contacts = torch.any(
            torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold, dim=1
        )
        self.illegal_contact_counter += contacts.int()
        return_ = torch.logical_and(
            self.illegal_contact_counter >= episode_length_threshold,
            env.episode_length_buf <= episode_length_threshold,
        )
        if return_.any() and print_reason:
            print(f"illegal_reset_contact: {return_.sum()} envs")
        return return_

    def reset(self, env_ids: Sequence[int] | slice | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self.illegal_contact_counter[env_ids] = 0
