"""Goal-reaching MDP components for ACP stage-3 fine-tuning on procedural terrains.

Contains:
- Env-type helpers: :func:`is_goal_env`, :func:`env_type_obs`, :func:`for_env_type`.
- :class:`GoalReachingCommand` / :class:`GoalReachingCommandCfg`: per-env procedural tile
  assignment, target-patch goal sampling (D4), goal encoding with ``r = 0.5`` m, minimum
  qualifying distance and success/cut-step tracking, and progress-timeout tracking (D6).
- Observations: :func:`combined_goal`, :func:`goal_dist_xy`, :func:`goal_delta_z`,
  :func:`goal_normalized_time`, :func:`planned_contacts_obs`.
- Events & terminations: :func:`reset_goal_envs` (D5), :func:`goal_time_out`,
  :func:`goal_out_of_bounds`, :func:`goal_bad_orientation`, and optional
  :func:`goal_no_progress` (D6).
- Rewards (Table 3 † terms): :func:`move2goal`, :func:`standatgoal`,
  :func:`goal_flat_orientation`, :func:`goal_stumbling`, :func:`goal_slippage`,
  :func:`goal_ang_vel_xy`, with opt-in terrain-mask extension (D7, off by default).
- Curriculum: :class:`GoalTerrainLevels` (EMA 0.2, promote > 0.5, demote > 1.0 m).
"""

from __future__ import annotations

import torch
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.managers import CommandTerm, CommandTermCfg, CurriculumTermCfg, ManagerTermBase, SceneEntityCfg
from isaaclab.utils.configclass import configclass

from instinctlab.terrains.trimesh.mesh_terrains_cfg import MotionMatchedTerrainCfg

from .contact_schedule import ContactScheduleCommand, heading

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors import ContactSensor


# -- env-type helpers ---------------------------------------------------------------------------


def is_goal_env(env: ManagerBasedRLEnv) -> torch.Tensor:
    """``(N,)`` bool mask: True for goal-reaching envs (the last ``cfg.num_goal_envs`` envs)."""
    num_goal = int(getattr(env.cfg, "num_goal_envs", 0))
    if num_goal <= 0:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    return torch.arange(env.num_envs, device=env.device) >= (env.num_envs - num_goal)


def env_type_obs(env: ManagerBasedRLEnv) -> torch.Tensor:
    """``(N, 2)`` one-hot observation: ``[1, 0]`` for reference envs, ``[0, 1]`` for goal envs."""
    goal = is_goal_env(env).float()
    return torch.stack([1.0 - goal, goal], dim=-1)


def for_env_type(
    env: ManagerBasedRLEnv,
    func: Callable[..., torch.Tensor],
    goal: bool = False,
    func_params: dict | None = None,
) -> torch.Tensor:
    """Apply a boolean termination term ``func`` to only reference envs (``goal=False``) or goal envs."""
    params = func_params or {}
    for v in params.values():
        if isinstance(v, SceneEntityCfg) and getattr(v, "body_ids", None) is None:
            v.resolve(env.scene)
    mask = is_goal_env(env) if goal else ~is_goal_env(env)
    return func(env, **params).bool() & mask


# -- opt-in terrain mask extension (D7, disabled by default) ------------------------------------

DEFAULT_TERRAIN_MASKS: dict[str, dict[str, float]] = {
    # foot-dominant terrains: masks on; stairs cap v_max at 0.8 m/s
    "stairs_up": {"T_ori": 1.0, "T_st": 1.0, "T_sl": 1.0, "T_omega": 1.0, "v_max": 0.8},
    "stairs_down": {"T_ori": 1.0, "T_st": 1.0, "T_sl": 1.0, "T_omega": 1.0, "v_max": 0.8},
    "walled_stairs": {"T_ori": 1.0, "T_st": 1.0, "T_sl": 1.0, "T_omega": 1.0, "v_max": 0.8},
    "slope_rough": {"T_ori": 1.0, "T_st": 1.0, "T_sl": 1.0, "T_omega": 1.0, "v_max": 2.0},
    "gaps": {"T_ori": 1.0, "T_st": 1.0, "T_sl": 1.0, "T_omega": 1.0, "v_max": 2.0},
    "double_stones": {"T_ori": 1.0, "T_st": 1.0, "T_sl": 1.0, "T_omega": 1.0, "v_max": 2.0},
    "beam_pallets": {"T_ori": 1.0, "T_st": 1.0, "T_sl": 1.0, "T_omega": 1.0, "v_max": 2.0},
    # whole-body high-platform climbing terrains (up to 1.0 m): masks off
    "box_ascent": {"T_ori": 0.0, "T_st": 0.0, "T_sl": 0.0, "T_omega": 0.0, "v_max": 2.0},
    "box_descent": {"T_ori": 0.0, "T_st": 0.0, "T_sl": 0.0, "T_omega": 0.0, "v_max": 2.0},
    "box_up_down": {"T_ori": 0.0, "T_st": 0.0, "T_sl": 0.0, "T_omega": 0.0, "v_max": 2.0},
    # legacy names kept for unit tests
    "pyramid_stairs": {"T_ori": 1.0, "T_st": 1.0, "T_sl": 1.0, "T_omega": 1.0, "v_max": 0.8},
    "perlin_slope": {"T_ori": 1.0, "T_st": 1.0, "T_sl": 1.0, "T_omega": 1.0, "v_max": 2.0},
}


def _terrain_mask_value(
    env: ManagerBasedRLEnv,
    command_name: str,
    key: str,
    default: float,
    use_terrain_masks: bool,
    terrain_masks: dict[str, dict[str, float]] | None = None,
) -> torch.Tensor:
    """Return ``(N,)`` per-env mask or parameter; constant ``default`` when ``use_terrain_masks`` is False."""
    if not use_terrain_masks:
        return torch.full((env.num_envs,), default, device=env.device)
    cmd: GoalReachingCommand = env.command_manager.get_term(command_name)
    table = DEFAULT_TERRAIN_MASKS if terrain_masks is None else terrain_masks
    vals = torch.tensor(
        [table.get(name, {}).get(key, default) for name in cmd.procedural_type_names],
        dtype=torch.float,
        device=env.device,
    )
    return vals[cmd.terrain_type]


# -- goal command -------------------------------------------------------------------------------


def encode_goal_b(
    root_pos_w: torch.Tensor,
    root_quat_w: torch.Tensor,
    goal_pos_w: torch.Tensor,
    goal_heading: torch.Tensor,
    goal_radius: float = 0.5,
) -> torch.Tensor:
    """``(N, 3)``: ``d_xy / max(|d_xy|, goal_radius)`` in body-yaw frame + relative heading in [-pi, pi)."""
    offset = math_utils.quat_apply_inverse(math_utils.yaw_quat(root_quat_w), goal_pos_w - root_pos_w)[:, :2]
    direction = offset / offset.norm(dim=-1, keepdim=True).clamp_min(goal_radius)
    rel_heading = math_utils.wrap_to_pi(goal_heading - heading(root_quat_w))
    return torch.cat([direction, rel_heading.unsqueeze(-1)], dim=-1)


class GoalReachingCommand(CommandTerm):
    """Manages per-env procedural terrain tiles, target goals, and episode progress for goal envs."""

    cfg: GoalReachingCommandCfg

    def __init__(self, cfg: GoalReachingCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.terrain = env.scene.terrain
        self.num_levels = cfg.num_levels

        origins = self.terrain.terrain_origins  # (num_rows, num_cols, 3)
        self.num_rows, self.num_cols = int(origins.shape[0]), int(origins.shape[1])

        # Identify procedural (non-motion-matched) columns from row 0 of subterrain_specific_cfgs
        sub_cfgs = getattr(self.terrain, "subterrain_specific_cfgs", None)
        proc_cols: list[int] = []
        proc_names: list[str] = []
        sub_terrain_names = (
            list(self.terrain.cfg.terrain_generator.sub_terrains.keys())
            if self.terrain.cfg.terrain_generator is not None
            else []
        )
        if sub_cfgs is not None:
            for col in range(self.num_cols):
                scfg = sub_cfgs[col]
                if not isinstance(scfg, MotionMatchedTerrainCfg):
                    proc_cols.append(col)
                    name = getattr(scfg, "terrain_name", "") or getattr(scfg, "sub_terrain_name", None)
                    if not name:
                        # match by type against generator sub_terrains
                        matched = [
                            k
                            for k, v in self.terrain.cfg.terrain_generator.sub_terrains.items()
                            if type(v) is type(scfg)
                        ]
                        name = matched[0] if matched else f"proc_{len(proc_names)}"
                    proc_names.append(name)
        else:
            proc_cols = list(range(self.num_cols))
            proc_names = sub_terrain_names[: self.num_cols] or [f"proc_{c}" for c in proc_cols]

        if not proc_cols:
            raise ValueError("GoalReachingCommand found no procedural (non-MotionMatched) columns in terrain.")

        self.procedural_col_ids = torch.tensor(proc_cols, dtype=torch.long, device=self.device)
        self.procedural_type_names: list[str] = proc_names
        self.num_procedural_types = len(proc_cols)

        # Flat patches: (num_rows, num_cols, num_patches, 3)
        self.target_patches: torch.Tensor | None = (
            self.terrain.flat_patches.get("target") if hasattr(self.terrain, "flat_patches") else None
        )
        self.spawn_patches: torch.Tensor | None = (
            self.terrain.flat_patches.get("spawn") if hasattr(self.terrain, "flat_patches") else None
        )

        # Tile dimensions for bounds check
        gen = self.terrain.cfg.terrain_generator
        self.tile_size = tuple(gen.size) if gen is not None else (14.0, 12.0)

        # Per-env state
        n = self.num_envs
        self.terrain_type = torch.arange(n, dtype=torch.long, device=self.device) % self.num_procedural_types
        self.terrain_level = torch.randint(0, cfg.max_init_level + 1, (n,), dtype=torch.long, device=self.device)
        self.tile_row = torch.zeros(n, dtype=torch.long, device=self.device)
        self.tile_col = self.procedural_col_ids[self.terrain_type].clone()
        self.tile_origin_w = torch.zeros(n, 3, device=self.device)
        self.spawn_pos_w = torch.zeros(n, 3, device=self.device)
        self.goal_pos_w = torch.zeros(n, 3, device=self.device)
        self.goal_heading = torch.zeros(n, device=self.device)
        self.initial_dist_xy = torch.ones(n, device=self.device)
        self.min_qualifying_dist = torch.full((n,), float("inf"), device=self.device)
        self.reached = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.cut_step = torch.full((n,), -1, dtype=torch.long, device=self.device)
        self.best_dist_xy = torch.full((n,), float("inf"), device=self.device)
        self.time_since_progress_s = torch.zeros(n, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self.goal_b

    @property
    def goal_b(self) -> torch.Tensor:
        return encode_goal_b(
            self.robot.data.root_pos_w.torch,
            self.robot.data.root_quat_w.torch,
            self.goal_pos_w,
            self.goal_heading,
            self.cfg.goal_radius,
        )

    @property
    def dist_xy(self) -> torch.Tensor:
        return (self.goal_pos_w[:, :2] - self.robot.data.root_pos_w.torch[:, :2]).norm(dim=-1)

    @property
    def delta_z(self) -> torch.Tensor:
        return self.goal_pos_w[:, 2] - self.robot.data.root_pos_w.torch[:, 2]

    @property
    def replay_partition(self) -> torch.Tensor:
        """D8: ``terrain_type * num_levels + terrain_level`` in ``[0, K * num_levels)``."""
        return self.terrain_type * self.num_levels + self.terrain_level

    @property
    def replay_group(self) -> torch.Tensor:
        return self.terrain_type

    def _level_to_row(self, levels: torch.Tensor) -> torch.Tensor:
        """Map logical levels ``0..num_levels-1`` to a random row in that level's difficulty band."""
        low = (levels * self.num_rows) // self.num_levels
        high = torch.maximum(((levels + 1) * self.num_rows) // self.num_levels, low + 1).clamp(max=self.num_rows)
        span = (high - low).clamp(min=1)
        offset = torch.minimum((torch.rand(levels.shape, device=self.device) * span.float()).long(), span - 1)
        return (low + offset).clamp(0, self.num_rows - 1)

    def sample_goals(self, env_ids: torch.Tensor) -> None:
        """Sample tile row, spawn point, and target patch for ``env_ids``."""
        if env_ids.numel() == 0:
            return
        rows = self._level_to_row(self.terrain_level[env_ids])
        cols = self.procedural_col_ids[self.terrain_type[env_ids]]
        self.tile_row[env_ids] = rows
        self.tile_col[env_ids] = cols
        origins = self.terrain.terrain_origins[rows, cols]
        self.tile_origin_w[env_ids] = origins

        # Spawn point: from "spawn" flat patches if available, else tile origin
        if self.spawn_patches is not None:
            sp = self.spawn_patches[rows, cols]  # (M, P, 3)
            sp_valid = torch.isfinite(sp).all(dim=-1)
            sp_w = sp_valid.float()
            has_sp = sp_valid.any(dim=-1)
            sp_idx = torch.multinomial(torch.where(has_sp.unsqueeze(-1), sp_w, torch.ones_like(sp_w)), 1).squeeze(-1)
            chosen_sp = sp[torch.arange(len(env_ids), device=self.device), sp_idx]
            spawn = torch.where(has_sp.unsqueeze(-1), chosen_sp, origins)
        else:
            spawn = origins.clone()
        self.spawn_pos_w[env_ids] = spawn

        # Target point: from "target" flat patches filtered by goal_dist_range (D4)
        d_min, d_max = self.cfg.goal_dist_range
        if self.target_patches is not None:
            tp = self.target_patches[rows, cols]  # (M, P, 3)
            finite = torch.isfinite(tp).all(dim=-1)
            dists = (tp[..., :2] - spawn.unsqueeze(1)[..., :2]).norm(dim=-1)
            in_range = finite & (dists >= d_min) & (dists <= d_max)
            use_in_range = in_range.any(dim=-1)
            has_any = finite.any(dim=-1)
            weights = torch.where(use_in_range.unsqueeze(-1), in_range.float(), finite.float())
            weights = torch.where(has_any.unsqueeze(-1), weights, torch.ones_like(weights))
            idx = torch.multinomial(weights, 1).squeeze(-1)
            chosen = tp[torch.arange(len(env_ids), device=self.device), idx]
            fallback = spawn + torch.tensor([0.5 * (d_min + d_max), 0.0, 0.0], device=self.device)
            target = torch.where(has_any.unsqueeze(-1), chosen, fallback)
        else:
            angle = torch.rand(len(env_ids), device=self.device) * (2.0 * torch.pi) - torch.pi
            dist = d_min + torch.rand(len(env_ids), device=self.device) * (d_max - d_min)
            target = spawn.clone()
            target[:, 0] += dist * torch.cos(angle)
            target[:, 1] += dist * torch.sin(angle)

        # z_goal = patch height + default standing base height (D4)
        target = target.clone()
        target[:, 2] = target[:, 2] + self.cfg.default_base_height
        self.goal_pos_w[env_ids] = target

        delta_xy = target[:, :2] - spawn[:, :2]
        self.goal_heading[env_ids] = torch.atan2(delta_xy[:, 1], delta_xy[:, 0])
        init_d = delta_xy.norm(dim=-1)
        self.initial_dist_xy[env_ids] = init_d
        self.min_qualifying_dist[env_ids] = float("inf")
        self.reached[env_ids] = False
        self.cut_step[env_ids] = -1
        self.best_dist_xy[env_ids] = init_d
        self.time_since_progress_s[env_ids] = 0.0

    def _resample_command(self, env_ids: Sequence[int] | slice):
        # Goal sampling is performed inside ``reset_goal_envs`` (event_manager mode="reset") right before
        # teleporting the robot to ``spawn_pos_w``. Do not resample again during ``command_manager.reset``,
        # which runs *after* events and would overwrite ``tile_origin_w`` / ``goal_pos_w`` with a new tile.
        pass

    def _update_metrics(self):
        pass

    def _update_command(self):
        goal_mask = is_goal_env(self._env)
        if not bool(goal_mask.any()):
            return
        d_xy = self.dist_xy
        dz = self.delta_z.abs()
        qualifying = goal_mask & (dz < self.cfg.z_tolerance)
        self.min_qualifying_dist = torch.where(
            qualifying, torch.minimum(self.min_qualifying_dist, d_xy), self.min_qualifying_dist
        )
        self.reached |= qualifying & (d_xy < self.cfg.success_radius)
        # First step within cut_radius (0.2 m) for hindsight labeler truncation
        hit_cut = goal_mask & (self.cut_step < 0) & (d_xy < self.cfg.cut_radius)
        self.cut_step = torch.where(hit_cut, self._env.episode_length_buf, self.cut_step)
        # Progress tracking for optional D6 no-progress termination
        improved = goal_mask & (d_xy < self.best_dist_xy - self.cfg.progress_min_delta_m)
        self.best_dist_xy = torch.where(improved, d_xy, self.best_dist_xy)
        self.time_since_progress_s = torch.where(
            improved, torch.zeros_like(self.time_since_progress_s), self.time_since_progress_s + self._env.step_dt
        )


@configclass
class GoalReachingCommandCfg(CommandTermCfg):
    class_type: type = GoalReachingCommand

    asset_name: str = "robot"
    resampling_time_range: tuple[float, float] = (1e9, 1e9)
    num_levels: int = 10
    max_init_level: int = 0
    default_base_height: float = 0.84
    """Default standing torso_link height of the torso-rooted G1 (init_state.pos[2] = 0.82, mocap ~0.84 m)."""
    spawn_height_clearance: float = 0.04
    """Extra vertical clearance at reset so feet always spawn cleanly above the terrain surface."""
    goal_radius: float = 0.5
    """ACP's r = 0.5 m for goal-reaching commands."""
    goal_dist_range: tuple[float, float] = (2.0, 6.5)
    """Horizontal distance range from spawn to far-side target patch (D4)."""
    success_radius: float = 0.5
    z_tolerance: float = 0.1
    cut_radius: float = 0.2
    """Hindsight labeler cuts admitted goal episodes at their first step within this horizontal distance."""
    goal_episode_length_s: float = 15.0
    progress_min_delta_m: float = 0.25
    """Distance improvement that resets ``time_since_progress_s`` (for optional D6 progress-timeout termination)."""


# -- observations -------------------------------------------------------------------------------


def combined_goal(env: ManagerBasedRLEnv, ref_command_name: str, goal_command_name: str) -> torch.Tensor:
    """``(N, 3)``: reference goal for reference envs, procedural terrain goal for goal envs."""
    ref_goal = env.command_manager.get_term(ref_command_name).goal_b
    goal_goal = env.command_manager.get_term(goal_command_name).goal_b
    return torch.where(is_goal_env(env).unsqueeze(-1), goal_goal, ref_goal)


def goal_dist_xy(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """``(N, 1)`` horizontal distance to the goal (0 for reference envs)."""
    cmd: GoalReachingCommand = env.command_manager.get_term(command_name)
    return (cmd.dist_xy * is_goal_env(env).float()).unsqueeze(-1)


def goal_delta_z(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """``(N, 1)`` ``z_goal - z_base`` (0 for reference envs)."""
    cmd: GoalReachingCommand = env.command_manager.get_term(command_name)
    return (cmd.delta_z * is_goal_env(env).float()).unsqueeze(-1)


def goal_normalized_time(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """``(N, 1)`` elapsed time divided by ``goal_episode_length_s``."""
    cmd: GoalReachingCommand = env.command_manager.get_term(command_name)
    t = (env.episode_length_buf.float() * env.step_dt) / cmd.cfg.goal_episode_length_s
    return t.clamp(0.0, 1.0).unsqueeze(-1)


def planned_contacts_obs(
    env: ManagerBasedRLEnv, num_bodies: int = 9, positions_key: str = "planned_contacts"
) -> torch.Tensor:
    """``(N, num_bodies * 3)`` planned contacts from ``env.policy_feedback`` (zeros before first act).

    Under D13, ``PlannerAlgoMixin`` overwrites this slot in ``critic_goal`` with fresh ``p_hat``
    at every ``act()`` and ``compute_returns()``.
    """
    fb = getattr(env, "policy_feedback", None)
    if fb is not None and positions_key in fb:
        return fb[positions_key].reshape(env.num_envs, num_bodies * 3)
    return torch.zeros(env.num_envs, num_bodies * 3, device=env.device)


# -- events & terminations ----------------------------------------------------------------------


def reset_goal_envs(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | slice,
    command_name: str = "goal_reaching",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    pos_z_range: tuple[float, float] = (0.0, 0.05),
    yaw_range: tuple[float, float] = (-0.4, 0.4),
    joint_pos_range: tuple[float, float] = (-0.2, 0.2),
) -> None:
    """Reset goal envs standing on their procedural tile, facing start->goal + U(yaw_range) (D5)."""
    if isinstance(env_ids, slice):
        ids = torch.arange(env.num_envs, device=env.device)[env_ids]
    else:
        ids = torch.as_tensor(env_ids, dtype=torch.long, device=env.device)
    goal_ids = ids[is_goal_env(env)[ids]]
    if goal_ids.numel() == 0:
        return
    cmd: GoalReachingCommand = env.command_manager.get_term(command_name)
    cmd.sample_goals(goal_ids)
    robot: Articulation = env.scene[asset_cfg.name]

    root_state = robot.data.default_root_state.torch[goal_ids].clone()
    root_state[:, :3] = cmd.spawn_pos_w[goal_ids]
    dz = pos_z_range[0] + torch.rand(len(goal_ids), device=env.device) * (pos_z_range[1] - pos_z_range[0])
    root_state[:, 2] += cmd.cfg.default_base_height + cmd.cfg.spawn_height_clearance + dz
    yaw = cmd.goal_heading[goal_ids] + (
        yaw_range[0] + torch.rand(len(goal_ids), device=env.device) * (yaw_range[1] - yaw_range[0])
    )
    zeros = torch.zeros_like(yaw)
    root_state[:, 3:7] = math_utils.quat_from_euler_xyz(zeros, zeros, yaw)
    root_state[:, 7:13] = 0.0
    robot.write_root_state_to_sim(root_state, env_ids=goal_ids)

    joint_pos = robot.data.default_joint_pos.torch[goal_ids].clone()
    joint_pos += joint_pos_range[0] + torch.rand_like(joint_pos) * (joint_pos_range[1] - joint_pos_range[0])
    soft_limits = robot.data.soft_joint_pos_limits.torch[goal_ids]
    joint_pos = torch.clamp(joint_pos, soft_limits[..., 0], soft_limits[..., 1])
    joint_vel = torch.zeros_like(joint_pos)
    robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=goal_ids)


def goal_time_out(env: ManagerBasedRLEnv, max_episode_length_s: float = 15.0) -> torch.Tensor:
    """Goal-env timeout at ``max_episode_length_s`` (15 s)."""
    max_steps = int(round(max_episode_length_s / env.step_dt))
    return (env.episode_length_buf >= max_steps) & is_goal_env(env)


def goal_out_of_bounds(
    env: ManagerBasedRLEnv,
    command_name: str = "goal_reaching",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    margin: float = 0.25,
    min_z_offset: float = -3.0,
) -> torch.Tensor:
    """Goal-env failure: robot base leaves its assigned course corridor or falls below tile floor (D6)."""
    cmd: GoalReachingCommand = env.command_manager.get_term(command_name)
    pos_w = env.scene[asset_cfg.name].data.root_pos_w.torch
    # Center bounds on midpoint of (spawn, goal) so directional tiles whose origin is offset in -x work cleanly
    center_xy = 0.5 * (cmd.spawn_pos_w[:, :2] + cmd.goal_pos_w[:, :2])
    rel = (pos_w[:, :2] - center_xy).abs()
    half_x = 0.5 * cmd.tile_size[0] - margin
    half_y = 0.5 * cmd.tile_size[1] - margin
    min_floor_z = torch.minimum(cmd.spawn_pos_w[:, 2], cmd.goal_pos_w[:, 2] - cmd.cfg.default_base_height)
    out = (rel[:, 0] > half_x) | (rel[:, 1] > half_y) | (pos_w[:, 2] < min_floor_z + min_z_offset)
    return out & is_goal_env(env)


def goal_bad_orientation(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    limit: float = 0.8,
) -> torch.Tensor:
    """Goal-env failure: torso tilt ``|g_torso,xy| > limit`` (D6)."""
    pg_xy = env.scene[asset_cfg.name].data.projected_gravity_b.torch[:, :2].norm(dim=-1)
    return (pg_xy > limit) & is_goal_env(env)


def goal_no_progress(
    env: ManagerBasedRLEnv,
    command_name: str = "goal_reaching",
    timeout_s: float = 8.0,
) -> torch.Tensor:
    """Optional D6 failure: goal env has not reduced goal distance for ``timeout_s`` seconds."""
    cmd: GoalReachingCommand = env.command_manager.get_term(command_name)
    return (cmd.time_since_progress_s > timeout_s) & (~cmd.reached) & is_goal_env(env)


# -- rewards (Table 3 † terms; D7 terrain masks opt-in, off by default) -------------------------


def move2goal(
    env: ManagerBasedRLEnv,
    command_name: str = "goal_reaching",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    v_max: float = 2.0,
    use_terrain_masks: bool = False,
    terrain_masks: dict[str, dict[str, float]] | None = None,
) -> torch.Tensor:
    """Paper Table 3 ``move2goal``: ``(1 - g_xy) g_v + g_xy g_z``."""
    cmd: GoalReachingCommand = env.command_manager.get_term(command_name)
    robot: Articulation = env.scene[asset_cfg.name]
    diff_xy = cmd.goal_pos_w[:, :2] - robot.data.root_pos_w.torch[:, :2]
    d_g = diff_xy.norm(dim=-1)
    dir_xy = diff_xy / d_g.clamp_min(1e-6).unsqueeze(-1)
    vel_xy = robot.data.root_lin_vel_w.torch[:, :2]
    v_toward = (vel_xy * dir_xy).sum(dim=-1)
    v_speed = vel_xy.norm(dim=-1)
    v_cap = _terrain_mask_value(env, command_name, "v_max", v_max, use_terrain_masks, terrain_masks)
    g_xy = (d_g < cmd.cfg.success_radius).float()
    g_v = ((v_toward > 0.1) & (v_speed <= v_cap)).float()
    g_z = 1.0
    return (1.0 - g_xy) * g_v + g_xy * g_z


def standatgoal(
    env: ManagerBasedRLEnv,
    command_name: str = "goal_reaching",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    foot_body_names: Sequence[str] = ("left_ankle_roll_link", "right_ankle_roll_link"),
) -> torch.Tensor:
    """Paper Table 3 ``standatgoal``: ``g_xy g_h g_psi exp(-(d_foot + d_tilt + d_q) / 3)``."""
    cmd: GoalReachingCommand = env.command_manager.get_term(command_name)
    robot: Articulation = env.scene[asset_cfg.name]
    d_g = cmd.dist_xy
    dz = cmd.delta_z.abs()
    d_psi = math_utils.wrap_to_pi(cmd.goal_heading - heading(robot.data.root_quat_w.torch)).abs()

    g_xy = (d_g < cmd.cfg.success_radius).float()
    g_h = (dz < 0.2).float()
    g_psi = (d_psi < 0.5).float()

    foot_ids = getattr(cmd, "_standatgoal_foot_ids", None)
    if foot_ids is None:
        foot_ids, _ = robot.find_bodies(list(foot_body_names), preserve_order=True)
        cmd._standatgoal_foot_ids = foot_ids
    feet_xy = robot.data.body_link_pos_w.torch[:, foot_ids, :2]
    d_foot = (feet_xy - cmd.goal_pos_w[:, None, :2]).norm(dim=-1).mean(dim=-1)
    d_tilt = robot.data.projected_gravity_b.torch[:, :2].norm(dim=-1)
    d_q = (robot.data.joint_pos.torch - robot.data.default_joint_pos.torch).abs().mean(dim=-1)
    return g_xy * g_h * g_psi * torch.exp(-(d_foot + d_tilt + d_q) / 3.0)


def goal_flat_orientation(
    env: ManagerBasedRLEnv,
    command_name: str = "goal_reaching",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    use_terrain_masks: bool = False,
    terrain_masks: dict[str, dict[str, float]] | None = None,
) -> torch.Tensor:
    """Paper Table 3 ``flat_orientation``: ``T_ori |g_torso,xy|^2``."""
    pg_sq = robot_pg_sq(env, asset_cfg)
    mask = _terrain_mask_value(env, command_name, "T_ori", 1.0, use_terrain_masks, terrain_masks)
    return mask * pg_sq


def robot_pg_sq(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    return env.scene[asset_cfg.name].data.projected_gravity_b.torch[:, :2].square().sum(dim=-1)


def goal_stumbling(
    env: ManagerBasedRLEnv,
    command_name: str = "goal_reaching",
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=["left_ankle_roll_link", "right_ankle_roll_link"]),
    use_terrain_masks: bool = False,
    terrain_masks: dict[str, dict[str, float]] | None = None,
) -> torch.Tensor:
    """Paper Table 3 ``stumbling``: ``T_st sum_f 1[F_xy^max > 1.1 F_z^max + 1]``."""
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = sensor.data.net_forces_w_history.torch[:, :, sensor_cfg.body_ids, :]  # (N, H, F, 3)
    f_xy_max = forces[..., :2].norm(dim=-1).max(dim=1).values
    f_z_max = forces[..., 2].abs().max(dim=1).values
    stumbled = (f_xy_max > 1.1 * f_z_max + 1.0).float().sum(dim=-1)
    mask = _terrain_mask_value(env, command_name, "T_st", 1.0, use_terrain_masks, terrain_masks)
    return mask * stumbled


def goal_slippage(
    env: ManagerBasedRLEnv,
    command_name: str = "goal_reaching",
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=["left_ankle_roll_link", "right_ankle_roll_link"]),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["left_ankle_roll_link", "right_ankle_roll_link"]),
    contact_time_threshold: float = 0.026,
    vel_threshold: float = 0.5,
    use_terrain_masks: bool = False,
    terrain_masks: dict[str, dict[str, float]] | None = None,
) -> torch.Tensor:
    """Paper Table 3 ``slippage``: ``T_sl sum_f 1[t_contact > 0.026] * 1[|v_f| > 0.5]``."""
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    robot: Articulation = env.scene[asset_cfg.name]
    contact_time = sensor.data.current_contact_time.torch[:, sensor_cfg.body_ids]
    foot_vel = robot.data.body_lin_vel_w.torch[:, asset_cfg.body_ids, :2].norm(dim=-1)
    slipped = ((contact_time > contact_time_threshold) & (foot_vel > vel_threshold)).float().sum(dim=-1)
    mask = _terrain_mask_value(env, command_name, "T_sl", 1.0, use_terrain_masks, terrain_masks)
    return mask * slipped


def goal_ang_vel_xy(
    env: ManagerBasedRLEnv,
    command_name: str = "goal_reaching",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    use_terrain_masks: bool = False,
    terrain_masks: dict[str, dict[str, float]] | None = None,
) -> torch.Tensor:
    """Paper Table 3 ``ang_vel_xy``: ``T_omega |omega_torso,xy|^2``."""
    ang_sq = env.scene[asset_cfg.name].data.root_ang_vel_b.torch[:, :2].square().sum(dim=-1)
    mask = _terrain_mask_value(env, command_name, "T_omega", 1.0, use_terrain_masks, terrain_masks)
    return mask * ang_sq


# -- curriculum (L6) ----------------------------------------------------------------------------


class GoalTerrainLevels(ManagerTermBase):
    """Per-env procedural terrain difficulty curriculum (ACP paper L1147-1152, L1178-1184).

    After each goal-env episode:
      ``s <- (1 - ema_alpha) * s + ema_alpha * I_success``
      promote if ``success and s > promote_threshold`` (random level at top)
      demote if ``not success and min_qualifying_dist > demote_distance``
    """

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.success_ema = torch.zeros(env.num_envs, device=env.device)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int] | torch.Tensor | slice,
        command_name: str = "goal_reaching",
        ema_alpha: float = 0.2,
        promote_threshold: float = 0.5,
        demote_distance: float = 1.0,
        reset_ema_on_level_change: bool = True,
    ) -> dict[str, float]:
        if isinstance(env_ids, slice):
            ids = torch.arange(env.num_envs, device=env.device)[env_ids]
        else:
            ids = torch.as_tensor(env_ids, dtype=torch.long, device=env.device)
        goal_mask = is_goal_env(env)
        goal_ids = ids[goal_mask[ids]]
        ep_len = getattr(env, "episode_length_buf", None)
        if ep_len is not None and goal_ids.numel() > 0:
            goal_ids = goal_ids[ep_len[goal_ids] > 0]
        cmd: GoalReachingCommand = env.command_manager.get_term(command_name)
        if goal_ids.numel() > 0:
            terminated = env.termination_manager.terminated[goal_ids]
            success = cmd.reached[goal_ids] & (~terminated)
            self.success_ema[goal_ids] = (1.0 - ema_alpha) * self.success_ema[goal_ids] + ema_alpha * success.float()

            promote = success & (self.success_ema[goal_ids] > promote_threshold)
            demote = (~success) & (cmd.min_qualifying_dist[goal_ids] > demote_distance)

            levels = cmd.terrain_level[goal_ids]
            new_levels = levels + promote.long() - demote.long()
            at_top = new_levels >= cmd.num_levels
            if bool(at_top.any()):
                rand_levels = torch.randint(0, cmd.num_levels, (int(at_top.sum()),), device=env.device)
                new_levels[at_top] = rand_levels
            new_levels = new_levels.clamp(0, cmd.num_levels - 1)
            cmd.terrain_level[goal_ids] = new_levels
            if reset_ema_on_level_change:
                changed = promote | demote
                self.success_ema[goal_ids[changed]] = 0.0

        all_goal = goal_mask.nonzero().squeeze(-1)
        if all_goal.numel() == 0:
            return {"mean_level": 0.0, "mean_success_ema": 0.0}
        return {
            "mean_level": float(cmd.terrain_level[all_goal].float().mean()),
            "mean_success_ema": float(self.success_ema[all_goal].mean()),
        }
