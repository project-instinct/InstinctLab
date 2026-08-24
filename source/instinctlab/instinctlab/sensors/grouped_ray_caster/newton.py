from __future__ import annotations

import re
from typing import TYPE_CHECKING

import warp as wp
from isaaclab_newton.physics import NewtonManager
from isaaclab_newton.sensors.ray_caster import MultiMeshRayCaster, MultiMeshRayCasterCamera

from instinctlab.sensors.grouped_ray_caster.flat_target_prim_registry import FlatTargetPrimRegistryMixin
from instinctlab.sensors.grouped_ray_caster.grouped_ray_caster import GroupedRayCasterKernelMixin
from instinctlab.sensors.grouped_ray_caster.grouped_ray_caster_camera import GroupedRayCasterCameraKernelMixin

if TYPE_CHECKING:
    from instinctlab.sensors.grouped_ray_caster.grouped_ray_caster_camera_cfg import GroupedRayCasterCameraCfg
    from instinctlab.sensors.grouped_ray_caster.grouped_ray_caster_cfg import GroupedRayCasterCfg


class NewtonGroupedRayCasterBackendMixin:
    """Newton site registration and tracked-target transform access."""

    def _register_sites_for_expr(self, prim_expr: str) -> list[str]:
        attach_expr = prim_expr
        if prim_expr.rsplit("/", 1)[-1].lower() in ("camera", "raycaster"):
            attach_expr = prim_expr.rsplit("/", 1)[0]
        body_pattern = re.sub(r"env_\.\*", "env_0", attach_expr)
        if body_pattern.startswith("/World/envs/env_0/"):
            identity = wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat(0.0, 0.0, 0.0, 1.0))
            return [NewtonManager.cl_register_site(body_pattern, identity)]
        return super()._register_sites_for_expr(prim_expr)

    def _register_target_sites_for_exprs(self, owner_exprs: list[str]) -> list[str]:
        identity = wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat(0.0, 0.0, 0.0, 1.0))
        patterns = [re.sub(r"env_(?:\.\*|\*)", "env_0", owner_expr) for owner_expr in owner_exprs]
        return [NewtonManager.cl_register_site(pattern, identity) for pattern in patterns]

    def _create_tracked_target_view(self, target_prim_path: str | list[str]):
        target_exprs = target_prim_path if isinstance(target_prim_path, list) else [target_prim_path]
        lookup_key = tuple(re.sub(r"env_\.\*", "env_*", expr) for expr in target_exprs)
        labels = self._tracked_site_labels_by_target[lookup_key]
        site_indices = self._resolve_site_indices(labels, str(target_prim_path), self._num_envs)
        return wp.array(site_indices, dtype=wp.int32, device=self._device)

    @staticmethod
    def _tracked_target_count(view) -> int:
        return int(view.shape[0])

    @staticmethod
    def _tracked_target_world_ids(view) -> None:
        return None

    def _tracked_target_transforms_wp(self, view) -> wp.array:
        view_count = self._tracked_target_count(view)
        transforms_wp = wp.empty(view_count, dtype=wp.transformf, device=self._device)
        self._update_newton_site_transforms(
            view,
            transforms_wp,
            wp.empty(view_count, dtype=wp.vec3f, device=self._device),
            wp.empty(view_count, dtype=wp.quatf, device=self._device),
        )
        return transforms_wp


class NewtonGroupedRayCaster(
    GroupedRayCasterKernelMixin,
    FlatTargetPrimRegistryMixin,
    NewtonGroupedRayCasterBackendMixin,
    MultiMeshRayCaster,
):
    """Newton ray caster over flat mesh entities grouped by fixed world IDs."""

    cfg: GroupedRayCasterCfg


class NewtonGroupedRayCasterCamera(
    GroupedRayCasterCameraKernelMixin,
    FlatTargetPrimRegistryMixin,
    NewtonGroupedRayCasterBackendMixin,
    MultiMeshRayCasterCamera,
):
    """Newton grouped ray-caster camera."""

    cfg: GroupedRayCasterCameraCfg
