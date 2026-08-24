from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import warp as wp

import isaaclab.utils.math as math_utils
import isaaclab.utils.string as string_utils
from isaaclab.markers import VisualizationMarkers
from isaaclab.sensors.sensor_base import SensorBase
from isaaclab.utils.warp import ProxyArray

from instinctlab.sensors.volume_points.volume_points_data import VolumePointsData
from instinctlab.utils.backend_dispatch import create_backend_component

if TYPE_CHECKING:
    from instinctlab.sensors.volume_points.volume_points_cfg import VolumePointsCfg


class VolumePointsBase(SensorBase):
    """Backend-neutral volume-points sampling and virtual-obstacle queries."""

    cfg: VolumePointsCfg

    def __init__(self, cfg: VolumePointsCfg):
        super().__init__(cfg)

    @property
    def data(self) -> VolumePointsData:
        self._update_outdated_buffers()
        return self._data

    @property
    def num_bodies(self) -> int:
        """Number of bodies with volume points attached."""
        return self._num_bodies

    @property
    def body_names(self) -> list[str]:
        """Ordered names of bodies with volume points attached."""
        return self._body_names

    def register_virtual_obstacles(self, virtual_obstacles: dict[str, Any]) -> None:
        """Register terrain-derived obstacles used for penetration queries."""
        self._virtual_obstacles.update(virtual_obstacles)

    def find_bodies(self, name_keys: str | Sequence[str], preserve_order: bool = False) -> tuple[list[int], list[str]]:
        """Find sensed bodies matching one or more regular expressions."""
        return string_utils.resolve_matching_names(name_keys, self.body_names, preserve_order)

    def _initialize_impl(self):
        super()._initialize_impl()
        self._initialize_backend_impl()

        self._volume_points_pattern = self.cfg.points_generator.func(self.cfg.points_generator).to(self.device)
        self._data = VolumePointsData.make_zero(
            num_envs=self._num_envs,
            num_bodies=self._num_bodies,
            point_num_each_body=self._volume_points_pattern.shape[0],
            device=self.device,
        )
        self._virtual_obstacles: dict[str, Any] = {}

    def _initialize_backend_impl(self) -> None:
        """Initialize backend-owned rigid-body state access."""
        raise NotImplementedError

    def _update_buffers_impl(self, env_mask: wp.array):
        env_ids = wp.to_torch(env_mask).nonzero(as_tuple=False).squeeze(-1)
        self._refresh_body_state(env_mask, env_ids)
        self._refresh_volume_points(env_ids)
        self._refresh_penetration_offset(env_ids)

    def _refresh_body_state(self, env_mask: wp.array, env_ids: torch.Tensor) -> None:
        """Update body pose and velocity buffers from the active backend."""
        raise NotImplementedError

    def _refresh_volume_points(self, env_ids: torch.Tensor) -> None:
        """Transform the local sampling pattern and compute point velocities."""
        pos_w = self._data.pos_w.torch
        quat_w = self._data.quat_w.torch
        vel_w = self._data.vel_w.torch
        ang_vel_w = self._data.ang_vel_w.torch
        points_pos_w_buffer = self._data.points_pos_w.torch
        points_vel_w_buffer = self._data.points_vel_w.torch

        num_env_bodies = pos_w[env_ids].shape[0] * pos_w[env_ids].shape[1]
        points_pos_w = math_utils.transform_points(
            self._volume_points_pattern.unsqueeze(0).expand(num_env_bodies, -1, -1),
            pos_w[env_ids].flatten(0, 1),
            quat_w[env_ids].flatten(0, 1),
        ).reshape(*pos_w[env_ids].shape[:2], self._data.point_num_each_body, 3)
        points_pos_w_buffer[env_ids] = points_pos_w

        points_vel_w = vel_w[env_ids].unsqueeze(-2).expand_as(points_pos_w).clone()
        points_vel_w += torch.linalg.cross(
            ang_vel_w[env_ids].unsqueeze(-2),
            points_pos_w - pos_w[env_ids].unsqueeze(-2),
            dim=-1,
        )
        points_vel_w_buffer[env_ids] = points_vel_w

    def _refresh_penetration_offset(self, env_ids: torch.Tensor) -> None:
        points_pos_w = self._data.points_pos_w.torch
        penetration_offset_buffer = self._data.penetration_offset.torch
        penetration_offset_buf = penetration_offset_buffer[env_ids]
        penetration_offset_buf.zero_()
        penetration_depth_buf = torch.zeros_like(penetration_offset_buf[..., 0])

        if env_ids.numel() == self._num_envs:
            points_query = self._data.points_pos_w
        else:
            points_query = ProxyArray(wp.from_torch(points_pos_w[env_ids].contiguous(), dtype=wp.vec3f))

        for virtual_obstacle in self._virtual_obstacles.values():
            penetration_offset = virtual_obstacle.get_points_penetration_offset(points_query).torch
            penetration_depth = torch.norm(penetration_offset, dim=-1)
            mask = penetration_depth > penetration_depth_buf
            penetration_depth_buf[mask] = penetration_depth[mask]
            penetration_offset_buf[mask] = penetration_offset[mask]

        penetration_offset_buffer[env_ids] = penetration_offset_buf

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "points_visualizer"):
                self.points_visualizer = VisualizationMarkers(self.cfg.visualizer_cfg)
            self.points_visualizer.set_visibility(True)
        elif hasattr(self, "points_visualizer"):
            self.points_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self._body_state_is_valid():
            return

        points = self._data.points_pos_w.torch.view(-1, 3)
        penetrated = torch.norm(self._data.penetration_offset.torch.view(-1, 3), dim=-1) > 0.0
        if not torch.any(penetrated):
            points = torch.cat([points, torch.zeros_like(points[:1])], dim=0)
            penetrated = torch.cat([penetrated, torch.tensor([True], device=self.device)], dim=0)

        self.points_visualizer.visualize(translations=points, marker_indices=penetrated.long())

    def _body_state_is_valid(self) -> bool:
        """Return whether backend-owned rigid-body state can be read."""
        raise NotImplementedError

    def _invalidate_initialize_callback(self, event):
        super()._invalidate_initialize_callback(event)
        if hasattr(self, "points_visualizer"):
            del self.points_visualizer
        self._invalidate_backend_impl()

    def _invalidate_backend_impl(self) -> None:
        """Invalidate backend-owned rigid-body state."""
        raise NotImplementedError


class VolumePoints:
    """Construct the volume-points sensor for the active physics backend."""

    def __new__(cls, cfg: VolumePointsCfg):
        return create_backend_component(
            cfg,
            {
                "physx": "instinctlab.sensors.volume_points.physx:PhysxVolumePoints",
                "newton": "instinctlab.sensors.volume_points.newton:NewtonVolumePoints",
            },
        )
