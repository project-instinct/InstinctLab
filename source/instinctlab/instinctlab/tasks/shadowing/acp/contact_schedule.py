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

import isaaclab.utils.math as math_utils
from isaaclab.managers import CommandTerm, CommandTermCfg, SceneEntityCfg
from isaaclab.utils.configclass import configclass

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
        self._dropout_body_ids = [self.body_names.index(n) for n in cfg.dropout_body_names]
        num_dropout = len(self._dropout_body_ids)
        self._dropped = torch.zeros(self.num_envs, num_dropout, dtype=torch.bool, device=self.device)
        # NaN: no target seen yet, so the first update draws for every body
        self._dropout_target_w = torch.full((self.num_envs, num_dropout, 3), float("nan"), device=self.device)

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
        keep = self.target_valid
        if self.cfg.target_footprint is not None:
            (x_min, x_max), (y_min, y_max) = self.cfg.target_footprint
            inside = (target_b[..., 0] >= x_min) & (target_b[..., 0] <= x_max)
            keep = keep & inside & (target_b[..., 1] >= y_min) & (target_b[..., 1] <= y_max)
        if self.cfg.target_dropout > 0.0:
            ids = self._dropout_body_ids
            target_w = self.target_pos_w[:, ids]
            self._dropped = redraw_dropout(
                self._dropped, target_w, self._dropout_target_w, self.cfg.target_dropout, self.cfg.dropout_redraw_m
            )
            self._dropout_target_w = target_w.clone()
            keep = keep.clone()
            keep[:, ids] &= ~self._dropped
        self._target_b[:] = (target_b * keep.unsqueeze(-1)).reshape(self.num_envs, -1)


@configclass
class ContactScheduleCommandCfg(CommandTermCfg):
    class_type: type = ContactScheduleCommand

    asset_name: str = "robot"
    motion_reference: SceneEntityCfg = SceneEntityCfg("motion_reference")
    goal_radius: float = 0.3
    """ ACP's r for reference-following goals. """
    resampling_time_range: tuple[float, float] = (1e9, 1e9)
    target_footprint: tuple[tuple[float, float], tuple[float, float]] | None = None
    """``((x_min, x_max), (y_min, y_max))`` in the body-yaw frame: targets outside are zero (no target), as the
    stage-2 planner, which selects scan points, can only give targets inside its scan. None keeps every target."""
    target_dropout: float = 0.0
    """Experiment: each target of ``dropout_body_names`` is withheld (zero) with this probability, drawn once per
    target (kept until the target moves by ``dropout_redraw_m``), so the tracker learns to reach contacts the stage-2
    planner misses. 0 disables it."""
    dropout_body_names: list[str] = ["left_ankle_roll_link", "right_ankle_roll_link"]
    dropout_redraw_m: float = 0.01


def redraw_dropout(
    dropped: torch.Tensor, target_w: torch.Tensor, previous_target_w: torch.Tensor, probability: float, redraw_m: float
) -> torch.Tensor:
    """``(N, B)`` dropout mask: redrawn with ``probability`` where the target moved by more than ``redraw_m`` since the
    previous step (a new target; a NaN previous target counts as moved), kept elsewhere."""
    moved = ~((target_w - previous_target_w).norm(dim=-1) <= redraw_m)
    draw = torch.rand(dropped.shape, device=dropped.device) < probability
    return torch.where(moved, draw, dropped)


"""
Observations.
"""


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
Rewards.
"""


def contact_label(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float = 1.0
) -> torch.Tensor:
    """Fraction of contact bodies whose contact state (force above ``threshold`` N) matches the reference.

    ``sensor_cfg.body_names`` must list the same bodies, in the same order, as the command.
    """
    term: ContactScheduleCommand = env.command_manager.get_term(command_name)
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = sensor.data.net_forces_w_history.torch[:, :, sensor_cfg.body_ids, :]
    robot_contact = forces.norm(dim=-1).max(dim=1)[0] > threshold
    return (robot_contact == term.reference_contact).float().mean(dim=-1)


def contact_link_pos_gauss(
    env: ManagerBasedRLEnv, command_name: str, std: float, in_base_frame: bool = False
) -> torch.Tensor:
    """Mean over bodies in reference contact of ``exp(-|p_robot - p_reference|^2 / std^2)``; 0 if none.

    Positions are body origins, in the world frame or, with ``in_base_frame``, relative to each base.
    """
    term: ContactScheduleCommand = env.command_manager.get_term(command_name)
    robot = term.robot
    reference = term.motion_reference.reference_frame
    if in_base_frame:
        root_pos_w, root_quat_w = robot.data.root_pos_w.torch, robot.data.root_quat_w.torch
        inv_pos, inv_quat = math_utils.subtract_frame_transforms(root_pos_w, root_quat_w)
        body_pos = math_utils.transform_points(
            robot.data.body_link_pos_w.torch[:, term.robot_body_ids], inv_pos, inv_quat
        )
        reference_pos = reference.link_pos_b[:, 0, term.reference_link_ids]
    else:
        body_pos = robot.data.body_link_pos_w.torch[:, term.robot_body_ids]
        reference_pos = reference.link_pos_w[:, 0, term.reference_link_ids]
    error = torch.sum(torch.square(body_pos - reference_pos), dim=-1)
    mask = term.reference_contact.float()
    reward = torch.exp(-error / (std * std)) * mask
    return reward.sum(dim=-1) / mask.sum(dim=-1).clamp_min(1.0)
