from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Sequence

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors import ContactSensor

    from instinctlab.sensors.volume_points.volume_points import VolumePoints


def volume_points_penetration(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, tolerance: float = 0.0
) -> torch.Tensor:
    """Penalize the penetration of volume points into the environment."""
    # extract the used quantities (to enable type-hinting)
    volume_sensor: VolumePoints = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    penetration = volume_sensor.data.penetration_offset.torch  # (N, B_, P_, 3), varying B_ and P_
    penetration = penetration.flatten(1, 2)  # (N, B_*P_, 3)
    penetration_depth = torch.norm(penetration, dim=-1)  # (N, B_*P_)
    in_obstacle = (penetration_depth > tolerance).float()  # (N, B_*P_)
    points_vel = volume_sensor.data.points_vel_w.torch  # (N, B_, P_, 3), varying B_ and P_
    points_vel = points_vel.flatten(1, 2)  # (N, B_*P_, 3)
    points_vel_norm = torch.norm(points_vel, dim=-1)  # (N, B_*P_)
    velocity_times_penetration = in_obstacle * (points_vel_norm + 1e-6) * penetration_depth  # (N, B_*P_)

    return torch.sum(velocity_times_penetration, dim=-1)


def step_safety(
    env: ManagerBasedRLEnv,
    volume_points_cfg: SceneEntityCfg,
    contact_forces_cfg: SceneEntityCfg,
    epsilon: float = 1e-5,
    once: bool = False,
) -> torch.Tensor:
    """A log based reward to encourage the robot to make contacts with no penetration to the virtual obstacles.
    Inspired by Deep Tracking Control and Robot Parkour Learning and Humanoid Parkour Learning.
    NOTE: make sure the contact forces sensor is selected for that the volume points sensors are interested in.
    aka. The number of selected bodies in the contact forces sensor should be the same as the number of selected bodies
    in all volume points sensors.
    NOTE: Be aware of the body order.
    """
    # extract the used quantities (to enable type-hinting)
    volume_sensor: VolumePoints = env.scene.sensors[volume_points_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[contact_forces_cfg.name]
    # compute the reward
    penetration = volume_sensor.data.penetration_offset.torch  # (N, B_, P_, 3), varying B_ and P_
    penetration_depth = torch.norm(penetration, dim=-1)  # (N, B_, P_)
    penetration_depth_max = torch.max(penetration_depth, dim=-1)[0]  # (N, B_)
    if once:
        contacts = contact_sensor.compute_first_contact(env.step_dt)[:, contact_forces_cfg.body_ids]  # (N, B_)
    else:
        contact_forces = contact_sensor.data.net_forces_w_history.torch[
            :, :, contact_forces_cfg.body_ids, :
        ]  # (N, T, B_, 3)
        contacts = torch.norm(contact_forces, dim=-1).max(dim=1)[0] > 1.0  # (N, B_)

    rewards = -torch.log(penetration_depth_max + epsilon) * contacts
    return torch.sum(rewards, dim=-1)
