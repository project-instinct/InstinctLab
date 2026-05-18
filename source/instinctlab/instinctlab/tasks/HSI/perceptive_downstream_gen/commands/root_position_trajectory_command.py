"""Flat-patch waypoint root trajectory (world XY, axis-aligned segments) for downstream_gen."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm
from isaaclab.terrains import TerrainImporter

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from .commands_cfg import RootPositionTrajectoryCommandCfg


def _normalize_env_indices(env_ids, num_envs: int, device: torch.device) -> torch.Tensor:
    if isinstance(env_ids, slice):
        return torch.arange(num_envs, device=device, dtype=torch.long)
    return torch.as_tensor(env_ids, device=device, dtype=torch.long).reshape(-1)


class RootPositionTrajectoryCommand(CommandTerm):
    """Nearest-origin + greedy-NN patch chain; integrate world-X then world-Y toward each waypoint."""

    cfg: RootPositionTrajectoryCommandCfg

    def __init__(self, cfg: RootPositionTrajectoryCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]
        self.terrain: TerrainImporter = env.scene["terrain"]

        patch_key = cfg.target_patch_key
        if patch_key not in self.terrain.flat_patches:
            raise RuntimeError(
                f"Root position trajectory expects flat patches under '{patch_key}'. "
                f"Found keys: {list(self.terrain.flat_patches.keys())}"
            )
        self._patch_tensor = self.terrain.flat_patches[patch_key]

        store = cfg.max_waypoints_stored
        self._waypoints_w = torch.zeros(self.num_envs, store, 3, device=self.device)
        self._num_waypoints = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        self._interp_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._waypoint_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._axis_phase = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._finished = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self._root_vel_command_w = torch.zeros(self.num_envs, 3, device=self.device)

        self.metrics["error_pos_xy"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["track_pos_xy_exp"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        """World XYZ root position reference (updated each physics step)."""
        return self._interp_pos_w

    @property
    def root_vel_command_w(self) -> torch.Tensor:
        """WORLD-frame velocity implied by discrete axis moves (sparse XY commands)."""
        return self._root_vel_command_w

    def __str__(self) -> str:
        return (
            "<RootPositionTrajectoryCommand>\n"
            f"\tpatch_key={self.cfg.target_patch_key}\tspeed={self.cfg.speed:.3g}\t"
            f"num_waypoints={self.cfg.num_waypoints}"
        )

    def _gather_active_targets(self) -> torch.Tensor:
        env_i = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        last_i = torch.clamp(self._num_waypoints - 1, min=0)
        wi = torch.clamp(self._waypoint_idx, max=last_i)
        return self._waypoints_w[env_i, wi]

    def _build_waypoints_for_tile(self, patches: torch.Tensor, origin_xy: torch.Tensor) -> torch.Tensor:
        valid = ~(patches.abs().sum(dim=-1) < 1.0e-9)
        p = patches[valid]
        if p.shape[0] == 0:
            raise RuntimeError("No usable flat-patch samples for RootPositionTrajectoryCommand.")

        take = min(int(self.cfg.num_waypoints), int(p.shape[0]))
        d0 = torch.norm(p[:, :2] - origin_xy.unsqueeze(0), dim=-1)
        cur_idx = int(torch.argmin(d0))

        indices: list[int] = [cur_idx]
        unused_mask = torch.ones(p.shape[0], dtype=torch.bool, device=p.device)
        unused_mask[cur_idx] = False

        for _ in range(take - 1):
            if not unused_mask.any():
                break
            rem_idx = unused_mask.nonzero(as_tuple=False).view(-1)
            last_xy = p[cur_idx, :2].unsqueeze(0)
            dist_xy = torch.norm(p[rem_idx, :2] - last_xy, dim=-1)
            sel_i = int(rem_idx[int(torch.argmin(dist_xy))])
            indices.append(sel_i)
            unused_mask[sel_i] = False
            cur_idx = sel_i

        return torch.stack([p[j] for j in indices], dim=0)

    @staticmethod
    def _step_axis_aligned(
        cur: torch.Tensor,
        tgt: torch.Tensor,
        mask: torch.Tensor,
        *,
        eps: float,
        stride: float,
        speed: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        new_c = cur.clone()
        vx = torch.zeros_like(cur)
        dm = tgt - cur
        snapped_ini = mask & (torch.abs(dm) <= eps)
        new_c[snapped_ini] = tgt[snapped_ini]
        mv_msk = mask & (~snapped_ini)
        mv = torch.sign(dm[mv_msk]) * torch.minimum(
            torch.abs(dm[mv_msk]),
            torch.full_like(dm[mv_msk], stride),
        )
        new_c[mv_msk] = cur[mv_msk] + mv
        snapped_fin = mv_msk & (torch.abs(tgt - new_c) <= eps)
        new_c[snapped_fin] = tgt[snapped_fin]
        vx[mv_msk] = torch.sign(dm[mv_msk]) * speed
        vx[snapped_ini] = 0.0
        vx[snapped_fin] = 0.0
        vx[~mask] = 0.0
        return new_c, vx

    def _refresh_command_z(self) -> None:
        dz = float(self.cfg.command_z_offset_from_root)
        self._interp_pos_w[:, 2] = self.robot.data.root_pos_w[:, 2] + dz

    def _update_metrics(self) -> None:
        denom = float(self.cfg.resampling_time_range[1]) / max(self._env.step_dt, 1.0e-9)
        err = torch.sum(
            torch.square(self.robot.data.root_pos_w[:, :2] - self._interp_pos_w[:, :2]),
            dim=-1,
        )
        self.metrics["error_pos_xy"] += torch.sqrt(torch.clamp(err, min=0.0)) / max(denom, 1.0)
        pos_std = float(self.cfg.pos_metrics_std)
        self.metrics["track_pos_xy_exp"] += torch.exp(-err / (pos_std**2)) / float(self._env.max_episode_length)

    def _resample_command(self, env_ids: Sequence[int] | slice) -> None:
        ids_t = _normalize_env_indices(env_ids, self.num_envs, self.device)
        origins = torch.as_tensor(self.terrain.terrain_origins, dtype=torch.float32, device=self.device)
        dz_slot = float(self.cfg.command_z_offset_from_root)

        for e in ids_t.tolist():
            row = int(self.terrain.terrain_levels[e])
            col = int(self.terrain.terrain_types[e])

            pts = self._patch_tensor[row, col].to(device=self.device, dtype=torch.float32)
            ori_xy = origins[row, col, :2]
            wp_full = self._build_waypoints_for_tile(pts, ori_xy)
            k_eff = min(int(wp_full.shape[0]), int(self.cfg.max_waypoints_stored))
            wp = wp_full[:k_eff]

            self._num_waypoints[e] = k_eff
            if k_eff > 0:
                self._waypoints_w[e, :k_eff].copy_(wp)
                if k_eff < self.cfg.max_waypoints_stored:
                    pad = wp[k_eff - 1].unsqueeze(0).expand(self.cfg.max_waypoints_stored - k_eff, 3)
                    self._waypoints_w[e, k_eff:].copy_(pad)

            self._interp_pos_w[e, :3].copy_(self.robot.data.root_pos_w[e, :3])
            self._interp_pos_w[e, 2] = self.robot.data.root_pos_w[e, 2] + dz_slot
            self._waypoint_idx[e] = 0
            self._axis_phase[e] = 0
            self._finished[e] = k_eff == 0
            self._root_vel_command_w[e].zero_()

    def _update_command(self) -> None:
        stride = float(self.cfg.speed) * float(self._env.step_dt)
        eps_xy = float(self.cfg.arrival_tolerance)

        tgt = self._gather_active_targets()
        self._root_vel_command_w.zero_()

        active = (~self._finished) & (self._num_waypoints > 0)

        ix = self._interp_pos_w[:, 0].clone()
        iy = self._interp_pos_w[:, 1].clone()

        phase0 = active & (self._axis_phase == 0)
        ix_new, vxc = self._step_axis_aligned(
            ix, tgt[:, 0], phase0, eps=eps_xy, stride=stride, speed=float(self.cfg.speed)
        )
        arrived_x = phase0 & (torch.abs(ix_new - tgt[:, 0]) <= eps_xy)
        self._axis_phase[arrived_x] = 1

        phase1 = active & (~self._finished) & (self._axis_phase == 1)
        iy_new, vyc = self._step_axis_aligned(
            iy, tgt[:, 1], phase1, eps=eps_xy, stride=stride, speed=float(self.cfg.speed)
        )

        nx = ix.clone()
        ny = iy.clone()
        nx[phase0] = ix_new[phase0]
        ny[phase1] = iy_new[phase1]
        self._interp_pos_w[:, 0] = nx
        self._interp_pos_w[:, 1] = ny

        vx_out = torch.zeros_like(vxc)
        vy_out = torch.zeros_like(vyc)
        vx_out[phase0] = vxc[phase0]
        vy_out[phase1] = vyc[phase1]
        self._root_vel_command_w[:, 0] = vx_out
        self._root_vel_command_w[:, 1] = vy_out

        arrived_y = phase1 & (torch.abs(iy_new - tgt[:, 1]) <= eps_xy)
        last_anchor = torch.clamp(self._num_waypoints - 1, min=0)
        on_final = arrived_y & (self._waypoint_idx >= last_anchor)
        on_inner = arrived_y & (self._waypoint_idx < last_anchor)

        self._finished[on_final] = True
        self._waypoint_idx[on_inner] = self._waypoint_idx[on_inner] + 1
        self._axis_phase[on_inner] = 0

        self._refresh_command_z()

    def _set_debug_vis_impl(self, debug_vis: bool):
        raise NotImplementedError("Debug visualization is not implemented for RootPositionTrajectoryCommand.")
