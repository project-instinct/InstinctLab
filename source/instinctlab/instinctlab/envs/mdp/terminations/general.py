""" Additinoal common termination functions that are not implemented in isaaclab. """

from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Sequence

from isaaclab.managers import ManagerTermBase, ManagerTermBaseCfg, SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.envs import ManagerBasedRLEnv

    from instinctlab.motion_reference import MotionReferenceManager


def dataset_exhausted(
    env: ManagerBasedRLEnv,
    reference_cfg: SceneEntityCfg = SceneEntityCfg("motion_reference"),
    reset_without_notice: bool = False,
    print_reason: bool = False,
) -> torch.Tensor:
    """Check if the dataset is exhausted.

    Args:
        env: The environment object.
        reset_without_notice: whether to reset the environment without returning True.
    Returns:
        True if the dataset is exhausted, False otherwise.
    """
    motion_reference: MotionReferenceManager = env.scene[reference_cfg.name]
    return_ = torch.logical_not(
        motion_reference.data.validity[motion_reference.ALL_INDICES, motion_reference.aiming_frame_idx]
    )  # shape: [N,]
    if print_reason and return_.any():
        print("dataset_exhausted: ", return_.sum())
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
        x_out_of_bounds = torch.abs(asset.data.root_pos_w[:, 0]) > 0.5 * map_width - distance_buffer
        y_out_of_bounds = torch.abs(asset.data.root_pos_w[:, 1]) > 0.5 * map_height - distance_buffer
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
):
    asset = env.scene[asset_cfg.name]
    return torch.norm(asset.data.root_lin_vel_w, dim=-1) > max_value


def abnormal_ang_vel(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    max_value: float = 40.0,  # [rad/s]
):
    asset = env.scene[asset_cfg.name]
    return torch.norm(asset.data.root_ang_vel_w, dim=-1) > max_value


def abnormal_joint_vel(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    max_value: float = 40.0,  # [rad/s]
):
    asset = env.scene[asset_cfg.name]
    return torch.any(torch.abs(asset.data.joint_vel) > max_value, dim=-1)


def abnormal_joint_acc(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    max_value: float = 4000.0,  # [rad/s^2]
):
    asset = env.scene[asset_cfg.name]
    return torch.any(torch.abs(asset.data.joint_acc) > max_value, dim=-1)


class bad_global_body_pos_from_default(ManagerTermBase):
    """Terminate if tracked body positions drift too far from per-episode reset pose."""

    def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._default_body_pos_w = None

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        threshold: float = 0.5,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        disable_flag: bool = False,
        print_reason: bool = False,
    ) -> torch.Tensor:
        if disable_flag:
            return torch.zeros((env.num_envs,), device=env.device, dtype=torch.bool)

        robot: Articulation = env.scene[asset_cfg.name]
        curr_body_pos_w = robot.data.body_link_pos_w[:, asset_cfg.body_ids]

        if self._default_body_pos_w is None or self._default_body_pos_w.shape != curr_body_pos_w.shape:
            self._default_body_pos_w = torch.zeros_like(curr_body_pos_w)

        new_episode_mask = env.episode_length_buf <= 1
        if torch.any(new_episode_mask):
            self._default_body_pos_w[new_episode_mask] = curr_body_pos_w[new_episode_mask]

        error = torch.norm(curr_body_pos_w - self._default_body_pos_w, dim=-1)
        return_ = torch.any(error > threshold, dim=-1)

        if print_reason and return_.any():
            print(f"bad_global_body_pos_from_default: {return_.sum()} envs")
        return return_

    def reset(self, env_ids: Sequence[int] | slice | None = None) -> None:
        if self._default_body_pos_w is None:
            return
        if env_ids is None:
            env_ids = slice(None)
        self._default_body_pos_w[env_ids] = 0.0


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
        net_contact_forces = contact_sensor.data.net_forces_w_history
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
