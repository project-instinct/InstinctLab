"""Contact schedule of the reference motion for ACP (Attention-based Contact Planning) trackers.

The motion reference carries the schedule (``ContactMotionReferenceData`` filled by ``TerrainContactMotion``). This
command reads its current frame and provides, per contact body,

- the next contact target: position of the next contact onset at or after the current frame (zero where there is
  none within the motion buffer's onset horizon), in the robot body-yaw frame,
- whether the reference is in contact now,
- a goal: the reference base pose a per-episode lookahead ahead (see ``TerrainContactMotionCfg``), as ACP's
  bounded horizontal direction plus the relative heading.

Mirrored references are not supported (the manager does not mirror contact fields), so a manager with symmetric
augmentation raises.
"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.managers import CommandTerm, CommandTermCfg, SceneEntityCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.configclass import configclass
from isaaclab.utils.noise import UniformNoiseCfg

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors import ContactSensor

    from instinctlab.motion_reference import ContactMotionReferenceData
    from instinctlab.motion_reference.motion_reference_manager import MotionReferenceManagerBase


def heading(quat: torch.Tensor) -> torch.Tensor:
    """Yaw of quaternions in the Isaac Lab convention, from the rotated x axis."""
    x_axis = torch.zeros(quat.shape[:-1] + (3,), device=quat.device, dtype=quat.dtype)
    x_axis[..., 0] = 1.0
    forward = math_utils.quat_apply(quat, x_axis)
    return torch.atan2(forward[..., 1], forward[..., 0])


class ContactScheduleCommand(CommandTerm):
    cfg: ContactScheduleCommandCfg

    def __init__(self, cfg: ContactScheduleCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.motion_reference: MotionReferenceManagerBase = env.scene[cfg.motion_reference.name]
        self.robot: Articulation = env.scene[cfg.asset_name]
        mr_cfg = self.motion_reference.cfg
        if mr_cfg.symmetric_augmentation_joint_mapping is not None:
            raise NotImplementedError("Contact fields are not mirrored; disable symmetric augmentation.")
        self.body_names = list(mr_cfg.contact_body_names)
        if not self.body_names:
            raise ValueError("The motion reference has no contact_body_names.")
        self.robot_body_ids, _ = self.robot.find_bodies(self.body_names, preserve_order=True)
        missing = [n for n in self.body_names if n not in mr_cfg.link_of_interests]
        if missing:
            raise ValueError(f"contact bodies {missing} must be in the motion reference link_of_interests")
        self.reference_link_ids = [list(mr_cfg.link_of_interests).index(n) for n in self.body_names]
        self._target_b = torch.zeros(self.num_envs, len(self.body_names) * 3, device=self.device)

    @property
    def _frame(self) -> ContactMotionReferenceData:
        return self.motion_reference.reference_frame  # type: ignore

    @property
    def target_pos_w(self) -> torch.Tensor:
        """``(N, B, 3)`` next contact target in the world frame (meaningful where :attr:`target_valid`)."""
        return self._frame.next_contact_pos_w[:, 0]

    @property
    def target_valid(self) -> torch.Tensor:
        return self._frame.next_contact_valid[:, 0]

    @property
    def reference_contact(self) -> torch.Tensor:
        """``(N, B)`` whether the reference body is in contact now."""
        return self._frame.contact_flag[:, 0]

    @property
    def command(self) -> torch.Tensor:
        """``(N, B * 3)`` next contact targets in the robot body-yaw frame, zero where invalid."""
        return self._target_b

    @property
    def goal_b(self) -> torch.Tensor:
        """``(N, 3)``: goal direction ``d_xy / max(|d_xy|, goal_radius)`` with ``d_xy`` the horizontal goal displacement
        in the robot body-yaw frame, and the goal heading relative to the robot's, wrapped to [-pi, pi)."""
        root_quat = self.robot.data.root_quat_w.torch
        offset = math_utils.quat_apply_inverse(
            math_utils.yaw_quat(root_quat), self._frame.goal_pos_w[:, 0] - self.robot.data.root_pos_w.torch
        )[:, :2]
        direction = offset / offset.norm(dim=-1, keepdim=True).clamp_min(self.cfg.goal_radius)
        relative_heading = math_utils.wrap_to_pi(heading(self._frame.goal_quat_w[:, 0]) - heading(root_quat))
        return torch.cat([direction, relative_heading.unsqueeze(-1)], dim=-1)

    def _update_metrics(self):
        pass

    def _resample_command(self, env_ids: Sequence[int]):
        pass

    def _update_command(self):
        num_bodies = len(self.body_names)
        root_pos = self.robot.data.root_pos_w.torch
        yaw_quat = math_utils.yaw_quat(self.robot.data.root_quat_w.torch)
        target_b = math_utils.quat_apply_inverse(
            yaw_quat.unsqueeze(1).expand(-1, num_bodies, -1), self.target_pos_w - root_pos.unsqueeze(1)
        )
        self._target_b[:] = (target_b * self.target_valid.unsqueeze(-1)).reshape(self.num_envs, -1)

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "_markers"):
                self._markers = VisualizationMarkers(self.cfg.visualizer_cfg)
            self._markers.set_visibility(True)
        elif hasattr(self, "_markers"):
            self._markers.set_visibility(False)

    def _debug_vis_callback(self, event):
        """Markers: an arrow from each robot contact body to its next contact target, the reference's contacting bodies
        (on the reference robot, shifted by the manager's ``visualizing_robot_offset``) and, with
        ``contact_sensor_name``, where the robot's contact bodies touch the terrain (the sensor's contact points, so it
        must track contact points with the terrain as filter).

        The arrows decode the targets from the computed observation term ``debug_vis_observation`` (group, term), i.e.
        what the policy observes after noise, back to the world frame with the robot's root position and yaw."""
        if not self.robot.is_initialized or not self._env.obs_buf:
            return
        group, term = self.cfg.debug_vis_observation
        num_bodies = len(self.body_names)
        target_b = self._env.obs_buf[group][term].reshape(self.num_envs, num_bodies, 3)
        yaw_quat = math_utils.yaw_quat(self.robot.data.root_quat_w.torch).unsqueeze(1).expand(-1, num_bodies, -1)
        target_w = self.robot.data.root_pos_w.torch.unsqueeze(1) + math_utils.quat_apply(yaw_quat, target_b)
        link_pos = self.robot.data.body_pos_w.torch[:, self.robot_body_ids][self.target_valid]
        arrow = target_w[self.target_valid] - link_pos
        length = arrow.norm(dim=-1)
        pitch = -torch.atan2(arrow[:, 2], arrow[:, :2].norm(dim=-1))
        arrow_quat = math_utils.quat_from_euler_xyz(
            torch.zeros_like(pitch), pitch, torch.atan2(arrow[:, 1], arrow[:, 0])
        )
        # arrow_x.usd spans x in [-0.25, 0.75] at unit scale: shift so the tail sits on the link and the tip on the target
        arrow_pos = link_pos + 0.25 * arrow

        offset = torch.as_tensor(self.motion_reference.cfg.visualizing_robot_offset, device=self.device)
        reference_pos = self._frame.link_pos_w[:, 0, self.reference_link_ids] + offset
        groups = [arrow_pos, reference_pos[self.reference_contact]]
        if self.cfg.contact_sensor_name is not None:
            sensor: ContactSensor = self._env.scene.sensors[self.cfg.contact_sensor_name]
            if sensor.data.contact_pos_w is None:
                raise ValueError(
                    f"sensor {self.cfg.contact_sensor_name!r} must track contact points for the debug view"
                )
            body_ids, _ = sensor.find_bodies(self.body_names, preserve_order=True)
            points = sensor.data.contact_pos_w.torch[:, body_ids].reshape(-1, 3)  # NaN where no contact
            groups.append(points[torch.isfinite(points).all(dim=-1)])
        translations = torch.cat(groups)
        if translations.shape[0] == 0:
            return
        indices = torch.cat([torch.full((len(g),), i, device=self.device) for i, g in enumerate(groups)])
        orientations = torch.zeros(len(translations), 4, device=self.device)
        orientations[:, 0] = 1.0
        orientations[: len(arrow_quat)] = arrow_quat
        scales = torch.ones(len(translations), 3, device=self.device)
        scales[: len(length)] = torch.stack(
            [length, torch.full_like(length, self.cfg.arrow_width), torch.full_like(length, self.cfg.arrow_width)],
            dim=-1,
        )
        self._markers.visualize(
            translations=translations, orientations=orientations, scales=scales, marker_indices=indices
        )


@configclass
class ContactScheduleCommandCfg(CommandTermCfg):
    class_type: type = ContactScheduleCommand

    asset_name: str = "robot"
    motion_reference: SceneEntityCfg = SceneEntityCfg("motion_reference")
    goal_radius: float = 0.3
    """ ACP's r for reference-following goals. """
    resampling_time_range: tuple[float, float] = (1e9, 1e9)
    contact_sensor_name: str | None = None
    """ Debug view only: also mark the contact points of the robot's contact bodies reported by this sensor. """
    debug_vis_observation: tuple[str, str] = ("policy", "contact_targets")
    """ Debug view only: observation (group, term) holding this command's targets, drawn as arrows. """
    arrow_width: float = 0.05
    """ Debug view only: target arrow width, in m. """
    visualizer_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/Command/contact_schedule",
        markers={
            "next_contact_target": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.9, 0.1)),
            ),
            "reference_contact": sim_utils.SphereCfg(
                radius=0.06, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.1, 0.1))
            ),
            "robot_contact": sim_utils.SphereCfg(
                radius=0.06, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.3, 0.9))
            ),
        },
    )
    """ Green: arrows from the robot's contact bodies to their next contact targets; red: reference bodies in contact (on the reference robot); blue: the robot's actual
    contact points with the terrain. """


"""
Observations.
"""


def contact_target_noise(data: torch.Tensor, cfg: UniformNoiseCfg) -> torch.Tensor:
    """Uniform noise on the valid ``(x, y, z)`` targets of a ``(N, B * 3)`` contact-target observation only: invalid
    targets are zero vectors (ACP) and stay zero. Use as ``UniformNoiseCfg(func=contact_target_noise, ...)``."""
    targets = data.reshape(data.shape[0], -1, 3)
    valid = (targets != 0.0).any(dim=-1, keepdim=True)
    noise = torch.rand_like(targets) * (cfg.n_max - cfg.n_min) + cfg.n_min
    return (targets + noise * valid).reshape(data.shape)


def contact_schedule_goal(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """``(N, 3)`` goal direction and relative heading of :class:`ContactScheduleCommand`."""
    return env.command_manager.get_term(command_name).goal_b


def reference_contacts(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """``(N, B)`` 1.0 where the reference body is in contact now."""
    return env.command_manager.get_term(command_name).reference_contact.float()


def scan_height_rel_base(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    miss_value: float = -2.0,
) -> torch.Tensor:
    """``(N, R)`` hit heights relative to the robot base; rays that hit nothing read ``miss_value``."""
    sensor = env.scene.sensors[sensor_cfg.name]
    hit_z = sensor.data.ray_hits_w.torch[..., 2]
    height = hit_z - env.scene[asset_cfg.name].data.root_pos_w.torch[:, 2:3]
    return torch.where(torch.isfinite(height), height, torch.full_like(height, miss_value))


"""
Rewards (ACP Table 3).
"""


def _moving(reference: ContactMotionReferenceData, speed_threshold: float) -> torch.Tensor:
    """``(N,)`` chi_mov: 1.0 where the reference base speed exceeds ``speed_threshold``."""
    return (reference.base_lin_vel_w[:, 0].norm(dim=-1) > speed_threshold).float()


def _in_yaw_frame(points_w: torch.Tensor, base_pos_w: torch.Tensor, base_quat_w: torch.Tensor) -> torch.Tensor:
    """``(N, B, 3)`` world points in the body-yaw frame of the base pose ``(N, 3)``, ``(N, 4)``."""
    yaw_quat = math_utils.yaw_quat(base_quat_w).unsqueeze(1).expand(-1, points_w.shape[1], -1)
    return math_utils.quat_apply_inverse(yaw_quat, points_w - base_pos_w.unsqueeze(1))


def base_pose_gauss(
    env: ManagerBasedRLEnv,
    command_name: str,
    std_pos: float = 0.5,
    std_rot: float = 1.0,
    moving_speed_threshold: float = 0.1,
) -> torch.Tensor:
    """track_base_pos: ``exp(-chi_mov |r - r*|^2 / std_pos^2 - theta^2 / std_rot^2)``, world frame."""
    term: ContactScheduleCommand = env.command_manager.get_term(command_name)
    reference = term.motion_reference.reference_frame
    pos_error = torch.sum(torch.square(term.robot.data.root_pos_w.torch - reference.base_pos_w[:, 0]), dim=-1)
    rot_error = math_utils.quat_error_magnitude(term.robot.data.root_quat_w.torch, reference.base_quat_w[:, 0])
    moving = _moving(reference, moving_speed_threshold)
    return torch.exp(-moving * pos_error / std_pos**2 - torch.square(rot_error) / std_rot**2)


def base_velocity_gauss(
    env: ManagerBasedRLEnv, command_name: str, std_lin: float = 1.0, std_ang: float = 3.0
) -> torch.Tensor:
    """track_base_vel: ``exp(-|v - v*|^2 / std_lin^2 - |w - w*|^2 / std_ang^2)``, world frame."""
    term: ContactScheduleCommand = env.command_manager.get_term(command_name)
    reference = term.motion_reference.reference_frame
    lin_error = torch.sum(torch.square(term.robot.data.root_lin_vel_w.torch - reference.base_lin_vel_w[:, 0]), dim=-1)
    ang_error = torch.sum(torch.square(term.robot.data.root_ang_vel_w.torch - reference.base_ang_vel_w[:, 0]), dim=-1)
    return torch.exp(-lin_error / std_lin**2 - ang_error / std_ang**2)


def contact_label(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float = 1.0
) -> torch.Tensor:
    """track_contact_label: ``mean_b c_meas_b (2 c_ref_b - 1)``, with ``c_meas`` a vertical contact force magnitude
    above ``threshold`` N within the sensor history: +1 per body touching where the reference does, -1 per body
    touching where it does not, 0 for bodies not touching.

    ``sensor_cfg.body_names`` must list the same bodies, in the same order, as the command.
    """
    term: ContactScheduleCommand = env.command_manager.get_term(command_name)
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    force_z = sensor.data.net_forces_w_history.torch[:, :, sensor_cfg.body_ids, 2]
    measured = (force_z.abs().max(dim=1)[0] > threshold).float()
    return (measured * (2.0 * term.reference_contact.float() - 1.0)).mean(dim=-1)


def contact_link_pos_gauss(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float = 1.0,
    in_yaw_frame: bool = False,
    moving_speed_threshold: float | None = None,
) -> torch.Tensor:
    """track_contact_pos_base / _global: ``exp(-chi sum_b |p_b - p*_b|^2 / std^2)`` over all contact bodies.

    Positions are body origins, in the world frame or, with ``in_yaw_frame``, in the body-yaw frame of each base (the
    robot's for the robot, the reference's for the reference). ``chi`` is chi_mov with ``moving_speed_threshold``,
    else 1.
    """
    term: ContactScheduleCommand = env.command_manager.get_term(command_name)
    robot = term.robot
    reference = term.motion_reference.reference_frame
    body_pos = robot.data.body_link_pos_w.torch[:, term.robot_body_ids]
    reference_pos = reference.link_pos_w[:, 0, term.reference_link_ids]
    if in_yaw_frame:
        body_pos = _in_yaw_frame(body_pos, robot.data.root_pos_w.torch, robot.data.root_quat_w.torch)
        reference_pos = _in_yaw_frame(reference_pos, reference.base_pos_w[:, 0], reference.base_quat_w[:, 0])
    error = torch.sum(torch.square(body_pos - reference_pos), dim=(-1, -2))
    if moving_speed_threshold is not None:
        error = _moving(reference, moving_speed_threshold) * error
    return torch.exp(-error / (std * std))
