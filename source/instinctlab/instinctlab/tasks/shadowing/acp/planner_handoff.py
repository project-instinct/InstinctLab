"""Environment side of ACP stage 2: the planner's terrain points, the reference-to-planner hand-off, and the reward
that follows the planner's targets.

- Terrain points: the grid of a yaw-aligned height scanner as (x, y) and height in the robot body-yaw frame, shaped
  like the scan grid (rows, cols) so the planner's CNN reads it as an image.
- Hand-off (paper Eq. S1): ``eta(u) = clamp((u - u_start) / N, 0, 1)`` of the training iteration ``u`` (set by the
  runner through ``env.training_iteration``). The input schedule is a per-env, per-step gate observation (1 = the
  algorithm replaces the reference contact targets by the planner's, with probability ``eta_obs``); the reward
  schedule scales the reference-contact reward terms by ``1 - eta_rew`` and the planner-contact term by
  ``eta_rew``.
- ``track_planned_contact``: reads the planner's targets the algorithm sent before the step
  (``env.policy_feedback``).
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.managers import CurriculumTermCfg, ManagerTermBase, SceneEntityCfg

from .contact_schedule import ContactScheduleCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors import RayCaster
    from isaaclab.sensors.ray_caster.patterns import GridPatternCfg


def linear_ramp(iteration: int, start_iteration: int, ramp_iterations: int, eta_max: float = 1.0) -> float:
    """0 until ``start_iteration``, then linear to ``eta_max`` over ``ramp_iterations`` (paper Eq. S1 reaches 1;
    a smaller ``eta_max`` keeps part of the hand-off with the reference)."""
    if ramp_iterations <= 0:
        raise ValueError(f"ramp_iterations must be positive, got {ramp_iterations}")
    if not 0.0 <= eta_max <= 1.0:
        raise ValueError(f"eta_max must be in [0, 1], got {eta_max}")
    return eta_max * min(max((iteration - start_iteration) / ramp_iterations, 0.0), 1.0)


"""
Observations.
"""


def grid_pattern_shape(pattern_cfg: GridPatternCfg) -> tuple[int, int]:
    """(rows, cols) of the rays of a grid pattern in their flattened order (as Isaac Lab's ``grid_pattern``)."""
    num_x = len(torch.arange(-pattern_cfg.size[0] / 2, pattern_cfg.size[0] / 2 + 1.0e-9, pattern_cfg.resolution))
    num_y = len(torch.arange(-pattern_cfg.size[1] / 2, pattern_cfg.size[1] / 2 + 1.0e-9, pattern_cfg.resolution))
    return (num_y, num_x) if pattern_cfg.ordering == "xy" else (num_x, num_y)


def _scan_points_b(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg, miss_height: float):
    """(N, rows, cols, 3) scan hits in the robot body-yaw frame. A ray that hits nothing reads its start's (x, y)
    and ``miss_height``."""
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    if sensor.cfg.ray_alignment != "yaw":
        raise ValueError(f"the scanner {sensor_cfg.name!r} must be yaw-aligned, got {sensor.cfg.ray_alignment!r}")
    robot = env.scene[asset_cfg.name]
    root_pos = robot.data.root_pos_w.torch
    yaw = math_utils.yaw_quat(robot.data.root_quat_w.torch)
    hits = sensor.data.ray_hits_w.torch
    num_envs, num_rays = hits.shape[:2]
    starts = sensor.data.pos_w.torch.unsqueeze(1) + math_utils.quat_apply(
        math_utils.yaw_quat(sensor.data.quat_w.torch).unsqueeze(1).expand(-1, num_rays, -1), sensor.ray_starts.torch
    )
    missed = ~torch.isfinite(hits).all(dim=-1, keepdim=True)
    points = torch.where(missed, starts, hits) - root_pos.unsqueeze(1)
    points = math_utils.quat_apply_inverse(yaw.unsqueeze(1).expand(-1, num_rays, -1), points)
    points[..., 2:] = torch.where(missed, torch.full_like(points[..., 2:], miss_height), points[..., 2:])
    return points.reshape(num_envs, *grid_pattern_shape(sensor.cfg.pattern_cfg), 3)


def scan_points_xy_b(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """``(N, rows, cols, 2)`` x, y of the scan points in the robot body-yaw frame."""
    return _scan_points_b(env, sensor_cfg, asset_cfg, 0.0)[..., :2]


def scan_points_height_b(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    miss_height: float = -2.0,
) -> torch.Tensor:
    """``(N, rows, cols)`` heights of the scan points relative to the robot base; missed rays read ``miss_height``."""
    return _scan_points_b(env, sensor_cfg, asset_cfg, miss_height)[..., 2]


def planner_gate(
    env: ManagerBasedRLEnv, start_iteration: int, ramp_iterations: int, eta_max: float = 1.0
) -> torch.Tensor:
    """``(N, 1)`` 1.0 with probability ``eta_obs`` of the current training iteration, drawn per env and step."""
    eta = linear_ramp(env.training_iteration, start_iteration, ramp_iterations, eta_max)
    return (torch.rand(env.num_envs, 1, device=env.device) < eta).float()


"""
Curricula.
"""


class RewardWeightHandoff(ManagerTermBase):
    """Moves reward weight from ``reference_terms`` to ``planner_terms`` with ``eta_rew`` of the training iteration:
    each term keeps its configured weight ``w``, scaled to ``w * (1 - eta)`` (reference) or ``w * eta`` (planner).
    """

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        # resolved at the first call: InstinctRlEnv builds its multi-reward manager after the curriculum manager
        self._term_cfgs = None
        self._base_weights = None

    def _resolve_terms(self, env: ManagerBasedRLEnv, names: list[str], group_name: str | None):
        manager = env.reward_manager
        self._term_cfgs = {
            name: manager.get_term_cfg(name, group_name) if group_name else manager.get_term_cfg(name) for name in names
        }
        self._base_weights = {name: term_cfg.weight for name, term_cfg in self._term_cfgs.items()}

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids,
        start_iteration: int,
        ramp_iterations: int,
        reference_terms: list[str],
        planner_terms: list[str],
        group_name: str | None = None,
        eta_max: float = 1.0,
    ) -> dict[str, float]:
        if self._term_cfgs is None:
            self._resolve_terms(env, list(reference_terms) + list(planner_terms), group_name)
        eta = linear_ramp(env.training_iteration, start_iteration, ramp_iterations, eta_max)
        for name in reference_terms:
            self._term_cfgs[name].weight = self._base_weights[name] * (1.0 - eta)
        for name in planner_terms:
            self._term_cfgs[name].weight = self._base_weights[name] * eta
        return {"eta_rew": eta}


"""
Rewards.
"""


def track_planned_contact(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    positions_key: str = "planned_contacts",
    occupancy_key: str = "planned_occupancy",
) -> torch.Tensor:
    """``exp(-sum_b m_b |p_b - p_hat_b|^2 / (std^2 max(1, sum_b m_b)))`` (the paper's ``track_planned_contact``).

    ``p_b`` is the position of contact body b (the command's bodies) and ``p_hat_b`` the planner's target, both in the
    robot body-yaw frame; ``m_b`` the planner's occupancy. The targets come from ``env.policy_feedback`` (sent by the
    algorithm before the step; a missing key raises), in the body-yaw frame of the step's start.
    """
    term: ContactScheduleCommand = env.command_manager.get_term(command_name)
    robot = term.robot
    planned = env.policy_feedback[positions_key].reshape(env.num_envs, -1, 3)
    occupancy = env.policy_feedback[occupancy_key].reshape(env.num_envs, -1)
    root_pos = robot.data.root_pos_w.torch
    yaw = math_utils.yaw_quat(robot.data.root_quat_w.torch)
    body_pos = robot.data.body_link_pos_w.torch[:, term.robot_body_ids] - root_pos.unsqueeze(1)
    body_pos_b = math_utils.quat_apply_inverse(yaw.unsqueeze(1).expand(-1, body_pos.shape[1], -1), body_pos)
    error = (occupancy * (body_pos_b - planned).square().sum(dim=-1)).sum(dim=-1)
    return torch.exp(-error / (std * std * occupancy.sum(dim=-1).clamp_min(1.0)))
