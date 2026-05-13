from __future__ import annotations

import os
import torch
import yaml
from typing import TYPE_CHECKING, Sequence

import joblib

from instinctlab.motion_reference import MotionSequence
from instinctlab.motion_reference.motion_files.amass_motion import AmassMotion

if TYPE_CHECKING:
    from isaaclab.scene import InteractiveScene

    from .terrain_motion_cfg import TerrainMotionCfg


class TerrainMotion(AmassMotion):
    """Processing Terrain motion files, which are typically terrain-dependent.
    NOTE: Multi-processing and accepting enable_trajectories() are not supported for now.
    """

    cfg: TerrainMotionCfg

    def enable_trajectories(self, traj_ids: torch.Tensor | slice | None = None) -> None:
        if not traj_ids is None:
            self._all_motion_terrain_ids = self._all_motion_terrain_ids[traj_ids]
            self._all_motion_origins_in_scene = self._all_motion_origins_in_scene[traj_ids]
            self._all_motion_num_selectable_origins = self._all_motion_num_selectable_origins[traj_ids]
        super().enable_trajectories(traj_ids)

    def _refresh_motion_file_list(self):
        """Refresh the list of motion files based on the terrain configuration."""
        with open(self.cfg.metadata_yaml) as file:
            self.yaml_data = yaml.safe_load(file)

        self._all_motion_files = [os.path.join(self.cfg.path, f["motion_file"]) for f in self.yaml_data["motion_files"]]
        self._motion_weights = torch.tensor(
            [float(f.get("weight", 1.0)) for f in self.yaml_data["motion_files"]],
            dtype=torch.float,
            device=self.buffer_device,
        )
        # new motion-specific attributes w.r.t amass_motion.py
        self._all_motion_terrain_ids = torch.tensor(
            [int(f["terrain_id"]) for f in self.yaml_data["motion_files"]],
            dtype=torch.int,
            device=self.buffer_device,
        )
        self._all_motion_origins_in_scene = (
            torch.ones(
                len(self._all_motion_files),
                self.cfg.max_origins_per_motion,
                3,
                dtype=torch.float,
                device=self.output_device,
            )
            * torch.nan
        )
        self._all_motion_num_selectable_origins = (
            torch.ones(len(self._all_motion_files), dtype=torch.int, device=self.output_device) * torch.nan
        )  # type: ignore

    def match_scene(self, scene: InteractiveScene) -> None:
        terrain = scene.terrain
        subterrain_specific_cfgs = scene.terrain.subterrain_specific_cfgs
        terrain_id_to_origins = {t["terrain_id"]: [] for t in self.yaml_data["terrains"]}
        # Collect all terrain origins in case some subterrains are from the same terrain_file.
        for row_idx in range(terrain.terrain_origins.shape[0]):
            for col_idx in range(terrain.terrain_origins.shape[1]):
                subterrain_cfg = subterrain_specific_cfgs[row_idx * terrain.terrain_origins.shape[1] + col_idx]
                difficulty = subterrain_cfg.difficulty
                terrain_idx = int(
                    min(max(difficulty * len(self.yaml_data["terrains"]), 0), len(self.yaml_data["terrains"]) - 1)
                )
                terrain_id_to_origins[terrain_idx].append(terrain.terrain_origins[row_idx, col_idx])

        # Set the origins for each motion by _all_motion_terrain_ids.
        for terrain_id, origins in terrain_id_to_origins.items():
            num_motion_matched = (self._all_motion_terrain_ids == terrain_id).sum().item()
            num_origins = len(origins)
            if num_origins == 0:
                print(f"Warning: Terrain {terrain_id} has no origins. Skipping.")
                self._all_motion_num_selectable_origins[self._all_motion_terrain_ids == terrain_id] = 0
                continue
            stacked_origins = torch.stack(origins)  # (num_origins, 3)
            if num_origins > self.cfg.max_origins_per_motion:
                # sample the origins subset for each matched motion (num_motion_matched, cfg.max_origins_per_motion, 3)
                sample_i = torch.randint(0, num_origins, (num_motion_matched, self.cfg.max_origins_per_motion))
                sampled_origins = stacked_origins[sample_i]
                self._all_motion_origins_in_scene[self._all_motion_terrain_ids == terrain_id] = sampled_origins
                self._all_motion_num_selectable_origins[self._all_motion_terrain_ids == terrain_id] = (
                    self.cfg.max_origins_per_motion
                )
            else:
                self._all_motion_origins_in_scene[self._all_motion_terrain_ids == terrain_id, :num_origins] = (
                    stacked_origins
                )
                self._all_motion_num_selectable_origins[self._all_motion_terrain_ids == terrain_id] = num_origins

    def _sample_assigned_env_starting_stub(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> None:
        """Also sample the origins for the given env_ids."""
        super()._sample_assigned_env_starting_stub(env_ids)
        assigned_ids = self.env_ids_to_assigned_ids(env_ids).to(self.buffer_device)

        if not hasattr(self, "_assigned_env_origins"):
            self._assigned_env_origins = torch.zeros(
                self.num_assigned_envs,
                3,
                dtype=torch.float,
                device=self.output_device,
            )

        self._safe_motion_resampling(assigned_ids)

        self._sample_assigned_env_origins(assigned_ids)

    def _safe_motion_resampling(self, assigned_ids: Sequence[int] | torch.Tensor) -> None:
        """Resample motions that have no terrain origins onto motions that do.

        Replacing `_assigned_env_motion_selection` without updating `_motion_buffer_start_time_s` would leave start
        times from the previous clip; `fill_init_reference_state` then indexes the new clip and can assert out of
        range. After any replacement, we resample start times (or zero them when no sampler applies).
        """
        if self._all_motion_num_selectable_origins.isnan().any():
            # terrain is not fully initialized, skip safe resampling
            return
        motion_ids = self._assigned_env_motion_selection[assigned_ids]
        no_origins_mask = self._all_motion_num_selectable_origins[motion_ids] == 0
        all_motion_mask_with_origins = self._all_motion_num_selectable_origins > 0
        all_motion_ids_with_origins = torch.where(all_motion_mask_with_origins)[0]
        if no_origins_mask.any():
            re_sample_ids = assigned_ids[no_origins_mask]
            resampled_motion_ids = torch.multinomial(
                self._motion_weights[all_motion_ids_with_origins],
                len(re_sample_ids),
                replacement=True,
            ).to(self.buffer_device)
            self._assigned_env_motion_selection[re_sample_ids] = all_motion_ids_with_origins[resampled_motion_ids]

            can_resample_start_times = self.cfg.motion_start_from_middle_range[1] > 0.0 or (
                self.cfg.motion_bin_length_s is not None and hasattr(self, "_motion_bin_weights")
            )
            if can_resample_start_times:
                self._sample_env_motion_start_time(re_sample_ids)
            else:
                self._motion_buffer_start_time_s[re_sample_ids] = 0.0

    def _sample_assigned_env_origins(self, assigned_ids: Sequence[int] | torch.Tensor) -> None:
        """Sample the origins for the assigned envs."""
        assert assigned_ids is not None  # type: ignore
        if self._all_motion_num_selectable_origins.isnan().any():
            # terrain is not fully initialized, skip safe resampling
            return
        motion_ids = self._assigned_env_motion_selection[assigned_ids]
        sample_i = (
            (torch.rand(len(assigned_ids)).to(self.output_device) * self._all_motion_num_selectable_origins[motion_ids])
            .floor()
            .long()
        )
        self._assigned_env_origins[assigned_ids] = self._all_motion_origins_in_scene[motion_ids, sample_i]

    def _get_motion_based_origin(self, env_origins: torch.Tensor, env_ids: Sequence[int] | torch.Tensor):
        """Get the origin for a specific motion based on its terrain ID.
        NOTE: env_origins is ignored in this case since the terrain motion is terrain-dependent.
        """
        assigned_ids = self.env_ids_to_assigned_ids(env_ids).to(self.buffer_device)
        return self._assigned_env_origins[assigned_ids]
