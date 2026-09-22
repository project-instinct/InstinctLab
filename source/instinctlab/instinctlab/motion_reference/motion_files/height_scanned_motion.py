from __future__ import annotations

import torch
import torch.nn.functional as F
from collections.abc import Sequence
from typing import TYPE_CHECKING

from instinctlab.motion_reference.motion_files.amass_motion import AmassMotion

if TYPE_CHECKING:
    from isaaclab.scene import InteractiveScene

    from instinctlab.motion_reference.motion_reference_data import MotionReferenceData, MotionReferenceState

    from .height_scanned_motion_cfg import HeightScannedMotionCfg


class HeightScannedMotion(AmassMotion):
    """Height-scanned AMASS motion buffer.

    The source motion is assumed to be authored on the z=0 ground plane. This
    buffer lifts the reference base/link positions by the terrain height sampled
    from a normal ray-caster height scanner.
    """

    cfg: HeightScannedMotionCfg

    def match_scene(self, scene: InteractiveScene) -> None:
        """Keep a reference to the configured ray-caster height scanner."""
        if self.cfg.height_scanner_name not in scene.sensors:
            raise KeyError(
                f"Height scanner '{self.cfg.height_scanner_name}' not found in scene sensors. "
                f"Available sensors: {list(scene.sensors.keys())}"
            )
        self.height_scanner = scene.sensors[self.cfg.height_scanner_name]

    def fill_init_reference_state(
        self,
        env_ids: Sequence[int] | torch.Tensor,
        env_origins: torch.Tensor,
        state_buffer: MotionReferenceState,
    ) -> None:
        super().fill_init_reference_state(env_ids, env_origins, state_buffer)

        if not hasattr(self, "height_scanner"):
            return

        base_pos_w = state_buffer.base_pos_w[env_ids]
        query_xy = base_pos_w[..., :2].unsqueeze(1)  # (B, 1, 2)
        sampled_z = self._sample_heights(query_xy, env_ids).squeeze(1)  # (B,)
        height_offset = sampled_z - env_origins[env_ids][..., 2] + self.cfg.height_offset  # (B,)
        if self.cfg.debug_height_offset:
            print(
                "[HeightScannedMotion] init height_offset: "
                f"mean={height_offset.mean().item():.4f}, "
                f"min={height_offset.min().item():.4f}, "
                f"max={height_offset.max().item():.4f}"
            )
        state_buffer.base_pos_w[env_ids] = base_pos_w + torch.stack(
            [torch.zeros_like(height_offset), torch.zeros_like(height_offset), height_offset], dim=-1
        )

    def fill_motion_data(
        self,
        env_ids: Sequence[int] | torch.Tensor,
        sample_timestamp: torch.Tensor,
        env_origins: torch.Tensor,
        data_buffer: MotionReferenceData,
    ) -> None:
        super().fill_motion_data(env_ids, sample_timestamp, env_origins, data_buffer)

        if not hasattr(self, "height_scanner"):
            return

        base_pos_w = data_buffer.base_pos_w[env_ids]  # (B, num_frames, 3)
        query_xy = base_pos_w[..., :2]  # (B, num_frames, 2)
        sampled_z = self._sample_heights(query_xy, env_ids)  # (B, num_frames)
        height_offset = (
            sampled_z - env_origins[env_ids][..., 2].unsqueeze(-1) + self.cfg.height_offset
        )  # (B, num_frames)
        if self.cfg.debug_height_offset:
            print(
                "[HeightScannedMotion] motion height_offset: "
                f"mean={height_offset.mean().item():.4f}, "
                f"min={height_offset.min().item():.4f}, "
                f"max={height_offset.max().item():.4f}"
            )

        base_pos_w = base_pos_w.clone()
        base_pos_w[..., 2] += height_offset
        data_buffer.base_pos_w[env_ids] = base_pos_w

        link_pos_w = data_buffer.link_pos_w[env_ids].clone()  # (B, num_frames, num_links, 3)
        link_pos_w[..., 2] += height_offset.unsqueeze(-1)
        data_buffer.link_pos_w[env_ids] = link_pos_w

    def _sample_heights(self, query_xy: torch.Tensor, env_ids: Sequence[int] | torch.Tensor) -> torch.Tensor:
        """Sample terrain height at query XY positions.

        Args:
            query_xy: Query positions in world XY, shape ``(B, Q, 2)``.
            env_ids: Environment indices to sample.

        Returns:
            Terrain-height offsets, shape ``(B, Q)``.
        """
        hits = self.height_scanner.data.ray_hits_w.torch[env_ids]  # (B, R, 3)
        valid = torch.isfinite(hits).all(dim=-1)  # (B, R)
        hits_xy = hits[..., :2]
        hits_z = hits[..., 2]

        process_func = getattr(self, f"process_height_{self.cfg.height_sampling_mode}")
        return process_func(query_xy, hits_xy, hits_z, valid)

    def process_height_mean(
        self,
        query_xy: torch.Tensor,
        hits_xy: torch.Tensor,
        hits_z: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        # Mean valid scanned height per environment, repeated along the query axis.
        count = valid.sum(dim=-1).clamp_min(1)
        mean_height = torch.where(valid, hits_z, torch.zeros_like(hits_z)).sum(dim=-1) / count
        return mean_height.unsqueeze(-1).expand(-1, query_xy.shape[1])

    def process_height_nearest(
        self,
        query_xy: torch.Tensor,
        hits_xy: torch.Tensor,
        hits_z: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        dist = torch.cdist(query_xy, hits_xy)  # (B, Q, R)
        dist = dist.masked_fill(~valid.unsqueeze(1), torch.inf)
        nearest = dist.argmin(dim=-1)  # (B, Q)
        sampled = torch.gather(hits_z, 1, nearest)  # (B, Q)

        # If an environment has no valid hits, argmin can still select an inf
        # ray-hit. Fall back to zero offset for those environments.
        has_valid = valid.any(dim=-1, keepdim=True)  # (B, 1)
        sampled = torch.where(has_valid, sampled, torch.zeros_like(sampled))
        sampled = torch.nan_to_num(sampled, nan=0.0, posinf=0.0, neginf=0.0)

        window = max(int(self.cfg.height_smoothing_window), 1)
        if window > 1 and sampled.shape[1] > 1:
            pad = window // 2
            padded = F.pad(sampled, (pad, window - pad - 1), mode="replicate")
            sampled = padded.unfold(-1, window, 1).mean(dim=-1)

        return sampled
